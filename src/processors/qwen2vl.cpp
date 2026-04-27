// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0
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

// ============================================================
// Special tokens (Qwen2-VL)
// ============================================================

static const std::string BOS_TOKEN = "<|im_start|>";
static const std::string EOS_TOKEN = "<|im_end|>";

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

    /// Build input_ids directly, inserting special token IDs for vision blocks
    /// without relying on the tokenizer to recognize them.
    ///
    /// For each message:
    ///   encode("<|im_start|>{role}\n")
    ///   [VISION_START_ID, IMAGE_PAD_ID×n, VISION_END_ID for each image]
    ///   encode("{content}<|im_end|>\n")
    ///
    /// image_patch_counts[i] = grid_t * grid_h * grid_w for the i-th image
    std::vector<int32_t> build_input_ids(
        const std::vector<geniex::ChatMessage>& messages,
        const std::vector<size_t>& image_patch_counts,
        bool add_generation_prompt) const
    {
        // Special token IDs for Qwen2-VL / Qwen2.5-Omni
        static constexpr int32_t VISION_START_ID = 151652;
        static constexpr int32_t VISION_END_ID   = 151653;
        static constexpr int32_t IMAGE_PAD_ID    = 151655;

        const size_t merge_length = static_cast<size_t>(config_.merge_size) * config_.merge_size;

        std::vector<int32_t> ids;
        size_t image_idx = 0;

        for (const auto& msg : messages) {
            // Encode message header: "<|im_start|>{role}\n"
            auto header_ids = tokenizer_->encode(
                BOS_TOKEN + msg.role + "\n", false);
            ids.insert(ids.end(), header_ids.begin(), header_ids.end());

            // Insert vision blocks for each image in this message.
            // n_pads = grid_t*grid_h*grid_w / merge_size² because the vision
            // encoder merges merge_size² patches into one embedding.
            for (size_t i = 0; i < msg.mm_content_paths.size(); ++i) {
                if (image_idx >= image_patch_counts.size()) break;
                ids.push_back(VISION_START_ID);
                size_t n_pads = image_patch_counts[image_idx++] / merge_length;
                ids.insert(ids.end(), n_pads, IMAGE_PAD_ID);
                ids.push_back(VISION_END_ID);
            }

            // Encode message content + close: "{content}<|im_end|>\n"
            auto content_ids = tokenizer_->encode(
                msg.content + EOS_TOKEN + "\n", false);
            ids.insert(ids.end(), content_ids.begin(), content_ids.end());
        }

        if (add_generation_prompt) {
            auto gen_ids = tokenizer_->encode(
                BOS_TOKEN + "assistant\n", false);
            ids.insert(ids.end(), gen_ids.begin(), gen_ids.end());
        }

        return ids;
    }
};

// ============================================================
// Qwen2VLProcessor public API
// ============================================================

Qwen2VLProcessor::Qwen2VLProcessor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

/*static*/
std::unique_ptr<Qwen2VLProcessor> Qwen2VLProcessor::create(
    const std::string& tokenizer_path,
    const Qwen2VLConfig& config)
{
    auto impl = std::make_unique<Impl>(tokenizer_path, config);
    // Can't use make_unique because ctor is private — use raw new via unique_ptr
    return std::unique_ptr<Qwen2VLProcessor>(new Qwen2VLProcessor(std::move(impl)));
}

geniex::Tokenizer& Qwen2VLProcessor::tokenizer() {
    return *impl_->tokenizer_;
}

BatchFeatures Qwen2VLProcessor::process(const geniex::VisionProcessorInput& input) {
    const auto& messages             = input.messages;
    const bool  add_generation_prompt = input.add_generation_prompt;

    // 1. Collect all image paths across all messages (in order)
    std::vector<std::string> all_image_paths;
    for (const auto& msg : messages) {
        for (const auto& path : msg.mm_content_paths) {
            all_image_paths.push_back(path);
        }
    }

    // 2. Load and preprocess each image; accumulate pixel_values and grid_thw
    xt::xarray<float>  pixel_values;
    xt::xarray<size_t> image_grid_thw;
    std::vector<size_t> image_patch_counts;  // grid_t * grid_h * grid_w per image

    for (const auto& path : all_image_paths) {
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

    // 3. Build input_ids directly (special tokens inserted as IDs, not text)
    std::vector<int32_t> input_ids = impl_->build_input_ids(messages, image_patch_counts, add_generation_prompt);

    // 4. Assemble BatchFeatures
    BatchFeatures features;
    features.input_ids = std::move(input_ids);
    if (!all_image_paths.empty()) {
        features.pixel_values   = std::move(pixel_values);
        features.image_grid_thw = std::move(image_grid_thw);
    }
    return features;
}

}  // namespace geniex::qwen2vl
