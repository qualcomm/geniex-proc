// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/internvl.h — InternVL vision-language processor.
//
// Image preprocessing for the single-tile (no dynamic tiling) export: resize to
// a fixed square, rescale to [0,1], normalize with ImageNet mean/std, then
// transpose HWC -> planar CHW.
//
// Produces the single input the exported ViT graph takes:
//   pixel_values [n_images, 3, image_size, image_size]  float32
//
// The exported graph is traced at a fixed tile count of one, so dynamic-preprocess
// tiling (a variable number of 448x448 tiles plus an optional thumbnail) is not
// implemented. Every image is resampled exactly once to image_size x image_size.

#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/processor.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex::internvl {

// ============================================================
// Configuration
// ============================================================

/// Image processor parameters. Defaults match the InternVL3.5 checkpoint (448
/// square, patch 14, downsample ratio 0.5 -> spatial_merge_size 2, ImageNet
/// mean/std).
///
/// The mean/std are ImageNet values, NOT the CLIP values used by the Qwen-VL
/// family. Feeding CLIP constants here silently degrades accuracy.
struct GENIEXPROC_API InternVLConfig {
    int image_size = 448;
    int patch_size = 14;
    int spatial_merge_size = 2;  // pixel-unshuffle factor (1 / downsample_ratio)
    float rescale_factor = 1.0f / 255.0f;

    std::vector<float> image_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> image_std = {0.229f, 0.224f, 0.225f};

    /// Special tokens delimiting the image span: Qwen-style vision delimiters
    /// wrapping a run of InternVL's own <IMG_CONTEXT> placeholders.
    std::string vision_start_token = "<|vision_start|>";
    std::string image_token = "<IMG_CONTEXT>";
    std::string vision_end_token = "<|vision_end|>";

    /// Patch-grid side length (image_size / patch_size), e.g. 448/14 = 32.
    int grid_side() const { return image_size / patch_size; }

    /// Visual tokens emitted per image after pixel-unshuffle, e.g. 32^2/2^2 = 256.
    int num_image_tokens() const {
        const int g = grid_side() / spatial_merge_size;
        return g * g;
    }
};

// ============================================================
// Processor
// ============================================================

/// Processor for InternVL VLMs. Composes a Tokenizer, image preprocessing, and
/// the InternVL (ChatML-derived) chat template.
///
/// process() expands each image marker into `vision_start_token` +
/// N x `image_token` + `vision_end_token`, where N is
/// InternVLConfig::num_image_tokens(). The resulting image-token run is
/// contiguous, letting the runtime splice the ViT output into `inputs_embeds`
/// by position.
class GENIEXPROC_API InternVLProcessor : public geniex::VisionProcessor {
   public:
    /// @param tokenizer_path path to tokenizer.json
    /// @throws std::runtime_error if the tokenizer cannot be loaded.
    static std::unique_ptr<InternVLProcessor> create(const std::string& tokenizer_path,
                                                     const InternVLConfig& config = {},
                                                     std::string image_marker_override = {});

    ~InternVLProcessor() override;

    InternVLProcessor(const InternVLProcessor&) = delete;
    InternVLProcessor& operator=(const InternVLProcessor&) = delete;

    /// Access the underlying tokenizer (encode / decode / is_eog).
    geniex::Tokenizer& tokenizer();

    const InternVLConfig& config() const;

    std::string apply_chat_template(const std::vector<geniex::ChatMessage>& messages,
                                    bool add_generation_prompt = true) const override;

    /// @throws std::runtime_error if the marker count does not match image_paths.
    BatchFeatures process(const std::string& formatted_text, const std::vector<std::string>& image_paths) override;

    /// Preprocess images only — no text. Fills pixel_values [n, 3, S, S] and
    /// image_grid_thw [n, 3].
    BatchFeatures process_images(const std::vector<std::string>& image_paths);

   protected:
    InternVLProcessor() = default;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    InternVLProcessor(std::unique_ptr<Impl> impl, std::string image_marker_override);
};

}  // namespace geniex::internvl
