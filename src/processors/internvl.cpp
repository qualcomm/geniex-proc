// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/processors/internvl.cpp — InternVL VLM processor implementation.
//
// Compiled only when GENIEXPROC_ENABLE_VISION is ON.

#include "geniex-proc/internvl.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "vision/vision.h"

namespace geniex::internvl {

InternVLProcessor::~InternVLProcessor() = default;

// ============================================================
// Special tokens (ChatML, shared with the Qwen family)
// ============================================================

static const std::string BOS_TOKEN = "<|im_start|>";
static const std::string EOS_TOKEN = "<|im_end|>";

// ============================================================
// Pimpl
// ============================================================

struct InternVLProcessor::Impl {
    std::unique_ptr<geniex::Tokenizer> tokenizer_;
    InternVLConfig config_;

    Impl(const std::string& tokenizer_path, const InternVLConfig& config) : config_(config) {
        // Optional: process_images() is pure image preprocessing and needs no
        // tokenizer. Only the text paths require one.
        if (!tokenizer_path.empty()) {
            tokenizer_ = geniex::Tokenizer::from_file(tokenizer_path);
            if (!tokenizer_) {
                GENIEXPROC_THROW("geniex::internvl: failed to load tokenizer from " + tokenizer_path);
            }
        }
        if (config_.image_size <= 0 || config_.patch_size <= 0 || config_.spatial_merge_size <= 0) {
            GENIEXPROC_THROW("geniex::internvl: image_size / patch_size / spatial_merge_size must be positive");
        }
        if (config_.image_size % (config_.patch_size * config_.spatial_merge_size) != 0) {
            GENIEXPROC_THROW("geniex::internvl: image_size must be divisible by patch_size * spatial_merge_size");
        }
        if (config_.image_mean.size() != 3 || config_.image_std.size() != 3) {
            GENIEXPROC_THROW("geniex::internvl: image_mean / image_std must each have 3 entries");
        }
    }

    // Preprocesses one image into a planar CHW float32 tile at `pixels_out`,
    // which must have room for 3 * image_size * image_size floats.
    void preprocessOne(const std::string& path, float* pixels_out) const {
        const int s = config_.image_size;

        // Load as HWC uint8 RGB.
        xt::xtensor<uint8_t, 3> img = geniex::vision::load_image(path);

        // Resize straight to the square tile the graph was traced at. Aspect
        // ratio is intentionally not preserved, matching the reference
        // preprocessor when dynamic tiling is disabled (max_num_tiles = 1).
        if (static_cast<int>(img.shape(0)) != s || static_cast<int>(img.shape(1)) != s) {
            img = geniex::vision::resize_image(img, s, s);
        }

        // Rescale to [0,1], normalize per channel, and transpose HWC -> planar
        // CHW in one pass. The graph takes [1, 3, S, S].
        const float scale = config_.rescale_factor;
        for (int c = 0; c < 3; ++c) {
            const float mean = config_.image_mean[static_cast<size_t>(c)];
            const float inv_std = 1.0f / config_.image_std[static_cast<size_t>(c)];
            float* dst = pixels_out + static_cast<size_t>(c) * s * s;
            for (int y = 0; y < s; ++y) {
                for (int x = 0; x < s; ++x) {
                    const float v = static_cast<float>(img(y, x, c)) * scale;
                    *dst++ = (v - mean) * inv_std;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Chat template — InternVL uses the ChatML format
    // ------------------------------------------------------------------

    // Renders ChatML for `messages`, hand-rolled because the bundle's
    // tokenizer_config.json ships no chat_template.
    //
    // Multi-turn callers reusing a KV cache pass only the new messages each turn,
    // and must re-emit the previous assistant turn's `<|im_end|>\n` themselves.
    std::string build_template_text(const std::vector<geniex::ChatMessage>& messages, bool add_generation_prompt,
                                    std::string_view image_marker) const {
        std::string out;
        for (const auto& msg : messages) {
            // Reject a literal marker inside user content: it would break
            // positional replacement during process().
            if (!image_marker.empty() && msg.content.find(image_marker) != std::string::npos) {
                GENIEXPROC_THROW("ChatMessage::content contains the reserved image_marker '" +
                                 std::string(image_marker) + "'");
            }

            out += BOS_TOKEN;
            out += role_to_string(msg.role);
            out += '\n';

            for (size_t i = 0; i < msg.mm_content.size(); ++i) {
                out += image_marker;
            }

            out += msg.content;
            out += EOS_TOKEN;
            out += '\n';
        }

        if (add_generation_prompt) {
            out += BOS_TOKEN;
            out += role_to_string(geniex::Role::Assistant);
            out += '\n';
        }

        return out;
    }
};

// ============================================================
// Construction
// ============================================================

InternVLProcessor::InternVLProcessor(std::unique_ptr<Impl> impl, std::string image_marker_override)
    : VisionProcessor(std::move(image_marker_override)), impl_(std::move(impl)) {}

std::unique_ptr<InternVLProcessor> InternVLProcessor::create(const std::string& tokenizer_path,
                                                             const InternVLConfig& config,
                                                             std::string image_marker_override) {
    auto impl = std::make_unique<Impl>(tokenizer_path, config);
    return std::unique_ptr<InternVLProcessor>(new InternVLProcessor(std::move(impl), std::move(image_marker_override)));
}

geniex::Tokenizer& InternVLProcessor::tokenizer() {
    if (!impl_->tokenizer_) {
        GENIEXPROC_THROW("geniex::internvl: processor was created without a tokenizer");
    }
    return *impl_->tokenizer_;
}

const InternVLConfig& InternVLProcessor::config() const { return impl_->config_; }

std::string InternVLProcessor::apply_chat_template(const std::vector<geniex::ChatMessage>& messages,
                                                   bool add_generation_prompt) const {
    return impl_->build_template_text(messages, add_generation_prompt, image_marker());
}

// ============================================================
// Image-only preprocessing
// ============================================================

BatchFeatures InternVLProcessor::process_images(const std::vector<std::string>& image_paths) {
    BatchFeatures out;
    if (image_paths.empty()) return out;

    const size_t n_img = image_paths.size();
    const size_t s = static_cast<size_t>(impl_->config_.image_size);
    const size_t g = static_cast<size_t>(impl_->config_.grid_side());

    out.pixel_values = xt::zeros<float>({n_img, size_t{3}, s, s});

    // Must be filled even though this graph's geometry is fixed: the runtime
    // keys off it. (T, H, W) = (1, grid, grid).
    out.image_grid_thw = xt::zeros<size_t>({n_img, size_t{3}});

    for (size_t i = 0; i < n_img; ++i) {
        impl_->preprocessOne(image_paths[i], out.pixel_values.data() + i * 3 * s * s);
        out.image_grid_thw(i, 0) = 1;
        out.image_grid_thw(i, 1) = g;
        out.image_grid_thw(i, 2) = g;
    }
    return out;
}

// ============================================================
// Full (text + image) processing
// ============================================================

BatchFeatures InternVLProcessor::process(const std::string& formatted_text,
                                         const std::vector<std::string>& image_paths) {
    BatchFeatures out = process_images(image_paths);

    if (!impl_->tokenizer_) {
        GENIEXPROC_THROW("geniex::internvl: process() needs a tokenizer");
    }

    // Expand each marker into vision_start + N x image_token + vision_end.
    // N is fixed by the traced graph geometry (448/14/2)^2 = 256.
    const std::string& marker = image_marker();
    const int n_tok = impl_->config_.num_image_tokens();

    std::string text;
    text.reserve(formatted_text.size());

    size_t pos = 0, found = 0, img_i = 0;
    while ((found = formatted_text.find(marker, pos)) != std::string::npos) {
        if (img_i >= image_paths.size()) {
            GENIEXPROC_THROW("geniex::internvl: more image markers than images (" + std::to_string(image_paths.size()) +
                             ")");
        }
        text.append(formatted_text, pos, found - pos);
        text.append(impl_->config_.vision_start_token);
        for (int k = 0; k < n_tok; ++k) {
            text.append(impl_->config_.image_token);
        }
        text.append(impl_->config_.vision_end_token);
        pos = found + marker.size();
        ++img_i;
    }
    text.append(formatted_text, pos, std::string::npos);

    if (img_i != image_paths.size()) {
        GENIEXPROC_THROW("geniex::internvl: fewer image markers than images (" + std::to_string(image_paths.size()) +
                         ")");
    }

    out.text = text;
    out.input_ids = impl_->tokenizer_->encode(text, /*add_special_tokens=*/false);
    return out;
}

}  // namespace geniex::internvl
