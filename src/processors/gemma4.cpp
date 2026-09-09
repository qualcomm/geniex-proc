// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/processors/gemma4.cpp — Gemma4 VLM processor implementation.
//
// Compiled only when GENIEXPROC_ENABLE_VISION is ON.
//
// Ported from transformers' Gemma4ImageProcessorPil. Validated numerically
// against it (see tests/gemma4.cpp).

#include "geniex-proc/gemma4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "vision/vision.h"

namespace geniex::gemma4 {

Gemma4Processor::~Gemma4Processor() = default;

namespace {

// Gemma4 only ships graphs for these soft-token budgets.
constexpr int kSupportedSoftTokens[] = {70, 140, 280, 560, 1120};

// Turn framing. Unlike ChatML's <|im_start|>, Gemma's BOS is a real
// beginning-of-sequence token, separate from the turn delimiters.
static const std::string BOS_TOKEN  = "<bos>";
static const std::string TURN_OPEN  = "<|turn>";
static const std::string TURN_CLOSE = "<turn|>";

// Gemma names the assistant role "model".
const char* gemmaRoleName(geniex::Role role) {
    return role == geniex::Role::Assistant ? "model" : geniex::role_to_string(role);
}

// Strips leading and trailing whitespace, leaving the interior untouched.
//
// Required for parity, not cosmetic: Gemma's published Jinja template pipes
// every message body through `| trim` before emitting it
// ({{- message['content'] | trim -}}), so a builder that skips this renders a
// different prompt than upstream for any body with surrounding whitespace —
// and the surrounding turn markers already supply the newlines, so leaving it
// in would put stray blank lines inside the turn.
//
// Edges only, matching Jinja's filter (Python str.strip()): "  a b  \n" becomes
// "a b", while "a   b" and "a\n\nb" are passed through verbatim. The character
// set is str.strip()'s default whitespace.
std::string trim(const std::string& s) {
    const char* ws    = " \t\n\r\f\v";
    const auto  first = s.find_first_not_of(ws);
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(ws) - first + 1);
}

// Aspect-ratio-preserving target size: the largest (h, w) that produces at most
// `max_patches` patches and is divisible by pooling_kernel_size * patch_size.
// Mirrors get_aspect_ratio_preserving_size() upstream.
std::pair<int, int> aspectRatioPreservingSize(
    int height, int width, int patch_size, int max_patches, int pooling_kernel_size) {
    const double total_px  = static_cast<double>(height) * static_cast<double>(width);
    const double target_px = static_cast<double>(max_patches) * patch_size * patch_size;
    const double factor    = std::sqrt(target_px / total_px);

    const double ideal_height = factor * height;
    const double ideal_width  = factor * width;
    const int    side_mult    = pooling_kernel_size * patch_size;

    int target_height = static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;
    int target_width  = static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;

    if (target_height == 0 && target_width == 0) {
        throw std::runtime_error(
            "geniex::gemma4: image resizes to 0x0; height/width must be divisible by "
            "pooling_kernel_size * patch_size = " +
            std::to_string(side_mult));
    }

    const int max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (target_height == 0) {
        target_height = side_mult;
        target_width  = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side_mult,
            max_side_length);
    } else if (target_width == 0) {
        target_width  = side_mult;
        target_height = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side_mult,
            max_side_length);
    }

    if (static_cast<double>(target_height) * target_width > target_px) {
        throw std::runtime_error("geniex::gemma4: resized image exceeds the patch budget");
    }
    return {target_height, target_width};
}

}  // namespace

// ============================================================
// Pimpl
// ============================================================

struct Gemma4Processor::Impl {
    std::unique_ptr<geniex::Tokenizer> tokenizer_;
    Gemma4Config                       config_;

    Impl(const std::string& tokenizer_path, const std::string& tokenizer_config_path, const Gemma4Config& config)
        : config_(config) {
        // The tokenizer is optional: process_images() is pure image preprocessing
        // and is used on its own to drive the VEG graph and to diff against the
        // Python reference. Only the text paths require one.
        if (!tokenizer_path.empty()) {
            tokenizer_ = geniex::Tokenizer::from_file(tokenizer_path, tokenizer_config_path);
            if (!tokenizer_) {
                throw std::runtime_error("geniex::gemma4: failed to load tokenizer from " + tokenizer_path);
            }
        }
        const auto* end = kSupportedSoftTokens + std::size(kSupportedSoftTokens);
        if (std::find(kSupportedSoftTokens, end, config_.max_soft_tokens) == end) {
            throw std::runtime_error(
                "geniex::gemma4: max_soft_tokens must be one of {70,140,280,560,1120}, got " +
                std::to_string(config_.max_soft_tokens));
        }
    }

    // Gemma4 turn framing, hand-rolled as in Qwen2VL / InternVL. Byte-identical
    // to the bundled Jinja template for the message shapes supported here:
    //
    //   <bos>  then per message  <|turn>ROLE\n <markers> <trimmed> <turn|>\n
    //   then, if add_generation_prompt,  <|turn>model\n
    //
    // BOS lands once at the very start, not per turn as in ChatML — so a
    // multi-turn caller reusing a KV cache must feed only the new turn's delta
    // and re-emit the previous turn's `<turn|>` itself.
    //
    // Narrower than the Jinja it replaces: tool calls and the thinking channel
    // are not rendered. Nothing drives either today; see the follow-up to move
    // every VLM processor back onto Jinja.
    std::string build_template_text(const std::vector<geniex::ChatMessage>& messages, bool add_generation_prompt,
                                    std::string_view image_marker) const {
        std::string out = BOS_TOKEN;
        for (const auto& msg : messages) {
            // A literal marker here would break positional marker-to-image
            // pairing in process().
            if (!image_marker.empty() && msg.content.find(image_marker) != std::string::npos) {
                GENIEXPROC_THROW("ChatMessage::content contains the reserved image_marker '" +
                                 std::string(image_marker) + "'");
            }

            out += TURN_OPEN;
            out += gemmaRoleName(msg.role);
            out += '\n';

            // Markers lead the text, matching Gemma's upstream ordering.
            for (size_t i = 0; i < msg.mm_content.size(); ++i) {
                out += image_marker;
            }

            out += trim(msg.content);
            out += TURN_CLOSE;
            out += '\n';
        }

        if (add_generation_prompt) {
            out += TURN_OPEN;
            out += gemmaRoleName(geniex::Role::Assistant);
            out += '\n';
        }
        return out;
    }

    // Preprocesses one image into `max_patches` rows of `patch_dim` floats plus
    // its (x, y) position ids. Returns the soft-token count for this image.
    int preprocessOne(const std::string& path, float* pixels_out, int32_t* positions_out) const {
        const int patch = config_.patch_size;
        const int maxp  = config_.max_patches();
        const int pdim  = config_.patch_dim();

        // Step 0: load as HWC uint8 RGB.
        xt::xtensor<uint8_t, 3> img = geniex::vision::load_image(path);

        // Step 0b: optional square pre-resize that pins the graph's soft-token
        // count. See Gemma4Config::force_square_size.
        if (config_.force_square_size > 0 &&
            (static_cast<int>(img.shape(0)) != config_.force_square_size ||
                static_cast<int>(img.shape(1)) != config_.force_square_size)) {
            img = geniex::vision::resize_image(img, config_.force_square_size, config_.force_square_size);
        }

        // Step 1: aspect-ratio-preserving resize onto the patch budget. A no-op
        // when force_square_size already lands on the target (768 does).
        const int h = static_cast<int>(img.shape(0));
        const int w = static_cast<int>(img.shape(1));
        const auto [th, tw] = aspectRatioPreservingSize(h, w, patch, maxp, config_.pooling_kernel_size);
        if (th != h || tw != w) {
            img = geniex::vision::resize_image(img, th, tw);
        }

        const int nph = static_cast<int>(img.shape(0)) / patch;
        const int npw = static_cast<int>(img.shape(1)) / patch;
        const int n   = nph * npw;
        if (n > maxp) {
            throw std::runtime_error("geniex::gemma4: " + std::to_string(n) + " patches exceeds budget " +
                                     std::to_string(maxp));
        }

        // Steps 2-4: rescale to [0,1] (no mean/std — Gemma4 trains on raw [0,1])
        // and patchify. Upstream reshapes (C,H,W) -> (C,nph,p,npw,p), transposes
        // to (nph,npw,p,p,C) and flattens, so within a patch the layout is
        // row-major over (patch_row, patch_col, channel) — channel fastest.
        const float scale = config_.rescale_factor;
        for (int pr = 0; pr < nph; ++pr) {
            for (int pc = 0; pc < npw; ++pc) {
                float* dst = pixels_out + static_cast<size_t>(pr * npw + pc) * pdim;
                for (int y = 0; y < patch; ++y) {
                    for (int x = 0; x < patch; ++x) {
                        for (int c = 0; c < 3; ++c) {
                            *dst++ = static_cast<float>(img(pr * patch + y, pc * patch + x, c)) * scale;
                        }
                    }
                }
            }
        }

        // Step 5: position ids. meshgrid(indexing="xy") then row-major flatten
        // gives (x, y) = (column, row) for patch index pr*npw+pc.
        for (int pr = 0; pr < nph; ++pr) {
            for (int pc = 0; pc < npw; ++pc) {
                int32_t* dst = positions_out + static_cast<size_t>(pr * npw + pc) * 2;
                dst[0]       = pc;  // x
                dst[1]       = pr;  // y
            }
        }

        // Step 6: pad — zero pixels, position id -1.
        std::fill(pixels_out + static_cast<size_t>(n) * pdim, pixels_out + static_cast<size_t>(maxp) * pdim, 0.0f);
        std::fill(positions_out + static_cast<size_t>(n) * 2, positions_out + static_cast<size_t>(maxp) * 2, -1);

        return n / (config_.pooling_kernel_size * config_.pooling_kernel_size);
    }
};

// ============================================================
// Construction
// ============================================================

Gemma4Processor::Gemma4Processor(std::unique_ptr<Impl> impl, std::string image_marker_override)
    : VisionProcessor(std::move(image_marker_override)), impl_(std::move(impl)) {}

std::unique_ptr<Gemma4Processor> Gemma4Processor::create(const std::string& tokenizer_path,
    const std::string& tokenizer_config_path, const Gemma4Config& config, std::string image_marker_override) {
    auto impl = std::make_unique<Impl>(tokenizer_path, tokenizer_config_path, config);
    return std::unique_ptr<Gemma4Processor>(
        new Gemma4Processor(std::move(impl), std::move(image_marker_override)));
}

geniex::Tokenizer& Gemma4Processor::tokenizer() {
    if (!impl_->tokenizer_) {
        throw std::runtime_error("geniex::gemma4: processor was created without a tokenizer");
    }
    return *impl_->tokenizer_;
}

const Gemma4Config& Gemma4Processor::config() const { return impl_->config_; }

std::string Gemma4Processor::apply_chat_template(
    const std::vector<geniex::ChatMessage>& messages, const geniex::ApplyChatTemplateOptions& opts) const {
    // No tokenizer needed: the framing comes from the message list alone.
    // Hand-rolled template: only add_generation_prompt is honored; tools are ignored.
    return impl_->build_template_text(messages, opts.add_generation_prompt, image_marker());
}

// ============================================================
// Image-only preprocessing
// ============================================================

BatchFeatures Gemma4Processor::process_images(const std::vector<std::string>& image_paths) {
    BatchFeatures out;
    if (image_paths.empty()) return out;

    const size_t n_img = image_paths.size();
    const size_t maxp  = static_cast<size_t>(impl_->config_.max_patches());
    const size_t pdim  = static_cast<size_t>(impl_->config_.patch_dim());

    out.pixel_values       = xt::zeros<float>({n_img, maxp, pdim});
    out.image_position_ids = xt::zeros<int32_t>({n_img, maxp, size_t{2}});
    out.num_soft_tokens_per_image.resize(n_img);

    for (size_t i = 0; i < n_img; ++i) {
        out.num_soft_tokens_per_image[i] = impl_->preprocessOne(image_paths[i],
            out.pixel_values.data() + i * maxp * pdim,
            out.image_position_ids.data() + i * maxp * 2);
    }
    return out;
}

// ============================================================
// Full (text + image) processing
// ============================================================

BatchFeatures Gemma4Processor::process(
    const std::string& formatted_text, const std::vector<std::string>& image_paths) {
    BatchFeatures out = process_images(image_paths);

    // Expand each marker into boi + N x image_token + eoi, N being that image's
    // soft-token count. Mirrors Gemma4Processor.replace_image_token() upstream.
    const std::string& marker = image_marker();
    std::string        text;
    text.reserve(formatted_text.size());

    size_t pos = 0, found = 0, img_i = 0;
    while ((found = formatted_text.find(marker, pos)) != std::string::npos) {
        if (img_i >= image_paths.size()) {
            throw std::runtime_error("geniex::gemma4: more image markers than images (" +
                                     std::to_string(image_paths.size()) + ")");
        }
        text.append(formatted_text, pos, found - pos);
        text.append(impl_->config_.boi_token);
        for (int k = 0; k < out.num_soft_tokens_per_image[img_i]; ++k) {
            text.append(impl_->config_.image_token);
        }
        text.append(impl_->config_.eoi_token);
        pos = found + marker.size();
        ++img_i;
    }
    text.append(formatted_text, pos, std::string::npos);

    if (img_i != image_paths.size()) {
        throw std::runtime_error("geniex::gemma4: " + std::to_string(image_paths.size()) + " images but only " +
                                 std::to_string(img_i) + " markers in the prompt");
    }

    if (!impl_->tokenizer_) {
        throw std::runtime_error("geniex::gemma4: process() needs a tokenizer");
    }
    out.text      = text;
    out.input_ids = impl_->tokenizer_->encode(text, /*add_special_tokens=*/false);
    return out;
}

}  // namespace geniex::gemma4
