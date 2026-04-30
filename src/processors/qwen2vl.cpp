// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/processors/qwen2vl.cpp — Qwen2-VL processor implementation.
//
// Compiled only when GENIEXPROC_BUILD_QWEN2VL is ON.
//
// Architecture:
//   Qwen2VLProcessor owns a pimpl (Impl) that holds the Tokenizer and all
//   image processing state. Consumers never see image processor internals.

#include "geniex-proc/qwen2vl.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "vision/vision.h"

namespace geniex::qwen2vl {

Qwen2VLProcessor::~Qwen2VLProcessor() = default;

// ============================================================
// Special tokens (Qwen2-VL)
// ============================================================

static const std::string BOS_TOKEN = "<|im_start|>";
static const std::string EOS_TOKEN = "<|im_end|>";

// Special token IDs for Qwen2-VL / Qwen2.5-Omni
static constexpr int32_t VISION_START_ID = 151652;
static constexpr int32_t VISION_END_ID   = 151653;
static constexpr int32_t IMAGE_PAD_ID    = 151655;

// ============================================================
// Pimpl
// ============================================================

struct Qwen2VLProcessor::Impl {
    std::unique_ptr<geniex::Tokenizer> tokenizer_;
    Qwen2VLConfig config_;

    // Cached xtensor views of mean/std for per-channel normalization
    xt::xtensor<float, 1> image_mean_;
    xt::xtensor<float, 1> image_std_;

    explicit Impl(const std::string& tokenizer_path, const Qwen2VLConfig& config)
        : tokenizer_(geniex::Tokenizer::from_file(tokenizer_path)),
          config_(config),
          image_mean_(xt::adapt(config.image_mean)),
          image_std_(xt::adapt(config.image_std)) {}

    // ------------------------------------------------------------------
    // Image preprocessing — mirrors qwen2vl-proc.cpp / qwen3-vl.cpp
    // ------------------------------------------------------------------

    /// Preprocess a single HWC uint8 image into flattened patches + grid dims.
    /// Returns: (patches [grid_t*grid_h*grid_w, C*T*P*P], grid_thw [1, 3])
    std::tuple<xt::xarray<float>, xt::xarray<size_t>>
    preprocess_single_image(const xt::xtensor<uint8_t, 3>& image) const {
        int height = static_cast<int>(image.shape(0));
        int width  = static_cast<int>(image.shape(1));
        int resized_height = height;
        int resized_width  = width;

        // 1. Resize
        xt::xtensor<uint8_t, 3> processed = image;
        if (config_.fixed_height > 0 && config_.fixed_width > 0) {
            // Fixed-shape mode: ignore aspect ratio and resize to the exact fixed size.
            resized_height = config_.fixed_height;
            resized_width  = config_.fixed_width;
        } else {
            std::tie(resized_height, resized_width) = geniex::vision::smart_resize(
                height, width,
                config_.patch_size * config_.merge_size,
                static_cast<int>(config_.min_pixels),
                static_cast<int>(config_.max_pixels));
        }
        processed = geniex::vision::resize_image(image, resized_height, resized_width);

        // 2. Cast to float, rescale, normalize
        xt::xtensor<float, 3> img_float = xt::cast<float>(processed);
        img_float = img_float * config_.rescale_factor;

        for (int c = 0; c < 3; ++c) {
            xt::view(img_float, xt::all(), xt::all(), c) =
                (xt::view(img_float, xt::all(), xt::all(), c) - image_mean_(c)) / image_std_(c);
        }

        // 3. HWC → CHW, add temporal dimension, tile
        xt::xtensor<float, 3> img_chw = xt::transpose(img_float, {2, 0, 1});
        // shape: [1, C, H, W]
        xt::xarray<float> patches = xt::view(img_chw, xt::newaxis(), xt::all(), xt::all(), xt::all());
        // tile to [temporal_patch_size, C, H, W] — xt::repeat repeats the single axis-0 element T times
        patches = xt::eval(xt::repeat(patches, static_cast<size_t>(config_.temporal_patch_size), 0));

        size_t channels = patches.shape(1);
        size_t grid_t   = patches.shape(0) / config_.temporal_patch_size;
        size_t grid_h   = static_cast<size_t>(resized_height) / config_.patch_size;
        size_t grid_w   = static_cast<size_t>(resized_width)  / config_.patch_size;

        // 4. Reshape + transpose for patch extraction
        // [grid_t, T, C, grid_h/ms, ms, P, grid_w/ms, ms, P]
        patches = patches.reshape({
            grid_t,
            static_cast<size_t>(config_.temporal_patch_size),
            channels,
            grid_h / config_.merge_size,
            static_cast<size_t>(config_.merge_size),
            static_cast<size_t>(config_.patch_size),
            grid_w / config_.merge_size,
            static_cast<size_t>(config_.merge_size),
            static_cast<size_t>(config_.patch_size),
        });
        patches = xt::transpose(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});

        // 5. Flatten to [n_patches, C*T*P*P]
        xt::xarray<float> flat = patches.reshape({
            grid_t * grid_h * grid_w,
            channels * config_.temporal_patch_size * config_.patch_size * config_.patch_size,
        });

        xt::xarray<size_t> grid_thw = xt::adapt(
            std::vector<size_t>{grid_t, grid_h, grid_w}, {1UL, 3UL});

        return {flat, grid_thw};
    }

    // ------------------------------------------------------------------
    // Chat template — Qwen2-VL format
    // ------------------------------------------------------------------


    std::string build_template_text(
        const std::vector<geniex::ChatMessage>& messages,
        bool add_generation_prompt,
        std::string_view image_marker) const
    {
        std::string out;
        for (const auto& msg : messages) {
            // Defensive: reject a literal marker inside user content — would
            // break positional replacement during process().
            if (!image_marker.empty() &&
                msg.content.find(image_marker) != std::string::npos) {
                GENIEXPROC_THROW(
                    "ChatMessage::content contains the reserved image_marker '" +
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

    /// Tokenize `formatted_text` while splicing a vision pad block in place of
    /// each occurrence of `image_marker`. The i-th marker is paired
    /// positionally with `image_patch_counts[i]`.
    ///
    /// @throws std::runtime_error if marker count != image_patch_counts.size().
    std::vector<int32_t> build_input_ids_from_text(
        const std::string& formatted_text,
        const std::vector<size_t>& image_patch_counts,
        std::string_view image_marker) const
    {
        const size_t merge_length =
            static_cast<size_t>(config_.merge_size) * config_.merge_size;
        GENIEXPROC_CHECK(!image_marker.empty());

        std::vector<int32_t> ids;
        size_t cursor   = 0;
        size_t img_idx  = 0;

        while (cursor < formatted_text.size()) {
            size_t hit = formatted_text.find(image_marker, cursor);
            if (hit == std::string::npos) {
                // Tail segment — encode and finish.
                auto seg = tokenizer_->encode(
                    formatted_text.substr(cursor), false);
                ids.insert(ids.end(), seg.begin(), seg.end());
                break;
            }

            // Encode text before the marker.
            if (hit > cursor) {
                auto seg = tokenizer_->encode(
                    formatted_text.substr(cursor, hit - cursor), false);
                ids.insert(ids.end(), seg.begin(), seg.end());
            }

            // Splice vision block for this image.
            if (img_idx >= image_patch_counts.size()) {
                GENIEXPROC_THROW(
                    "formatted_text has more '" + std::string(image_marker) +
                    "' markers than supplied images");
            }
            ids.push_back(VISION_START_ID);
            size_t n_pads = image_patch_counts[img_idx++] / merge_length;
            ids.insert(ids.end(), n_pads, IMAGE_PAD_ID);
            ids.push_back(VISION_END_ID);

            cursor = hit + image_marker.size();
        }

        if (img_idx != image_patch_counts.size()) {
            GENIEXPROC_THROW(
                "formatted_text has fewer '" + std::string(image_marker) +
                "' markers than supplied images");
        }

        return ids;
    }
};

// ============================================================
// Qwen2VLProcessor public API
// ============================================================

Qwen2VLProcessor::Qwen2VLProcessor(std::unique_ptr<Impl> impl,
                                   std::string image_marker_override)
    : geniex::VisionProcessor(std::move(image_marker_override)),
      impl_(std::move(impl)) {}

/*static*/
std::unique_ptr<Qwen2VLProcessor> Qwen2VLProcessor::create(
    const std::string& tokenizer_path,
    const Qwen2VLConfig& config,
    std::string image_marker_override)
{
    auto impl = std::make_unique<Impl>(tokenizer_path, config);
    // Can't use make_unique because ctor is private — use raw new via unique_ptr
    return std::unique_ptr<Qwen2VLProcessor>(
        new Qwen2VLProcessor(std::move(impl), std::move(image_marker_override)));
}

geniex::Tokenizer& Qwen2VLProcessor::tokenizer() {
    return *impl_->tokenizer_;
}

std::string Qwen2VLProcessor::apply_chat_template(
    const std::vector<geniex::ChatMessage>& messages,
    bool add_generation_prompt) const
{
    return impl_->build_template_text(messages, add_generation_prompt, image_marker());
}

BatchFeatures Qwen2VLProcessor::process(
    const std::string& formatted_text,
    const std::vector<std::string>& image_paths)
{
    // 1. Load and preprocess each image; accumulate pixel_values and grid_thw
    xt::xarray<float>  pixel_values;
    xt::xarray<size_t> image_grid_thw;
    std::vector<size_t> image_patch_counts;  // grid_t * grid_h * grid_w per image

    for (const auto& path : image_paths) {
        auto image = geniex::vision::load_image(path);
        auto [patches, grid_thw] = impl_->preprocess_single_image(image);

        // grid_thw shape [1, 3]: (T, H, W)
        size_t grid_t = grid_thw(0, 0);
        size_t grid_h = grid_thw(0, 1);
        size_t grid_w = grid_thw(0, 2);
        image_patch_counts.push_back(grid_t * grid_h * grid_w);

        if (pixel_values.shape().size() == 0) {
            pixel_values    = patches;
            image_grid_thw  = grid_thw;
        } else {
            pixel_values   = xt::concatenate(std::make_tuple(pixel_values, patches), 0);
            image_grid_thw = xt::concatenate(std::make_tuple(image_grid_thw, grid_thw), 0);
        }
    }

    // 2. Tokenize formatted_text while splicing vision pad blocks at each marker.
    //    Validates that marker count matches image_paths.size().
    std::vector<int32_t> input_ids =
        impl_->build_input_ids_from_text(formatted_text, image_patch_counts, image_marker());

    // 3. Assemble BatchFeatures
    BatchFeatures features;
    features.text      = formatted_text;
    features.input_ids = std::move(input_ids);
    if (!image_paths.empty()) {
        features.pixel_values   = std::move(pixel_values);
        features.image_grid_thw = std::move(image_grid_thw);
    }
    return features;
}

}  // namespace geniex::qwen2vl
