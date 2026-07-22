// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/gemma4.h — Gemma4 vision-language processor.
//
// Mirrors transformers' Gemma4ImageProcessorPil: aspect-ratio-preserving resize
// onto a patch budget, rescale to [0,1] (no mean/std — Gemma4 was trained on raw
// [0,1] pixels), patchify, then pad to a fixed patch count with position id -1.
//
// Produces the two inputs the exported VEG (Visual Embedding Generator) graph
// takes:
//   pixel_values       [n_images, max_patches, patch_size^2 * 3]  float32
//   image_position_ids [n_images, max_patches, 2]                 int32  (x, y)

#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/processor.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex::gemma4 {

// ============================================================
// Configuration
// ============================================================

/// Image processor parameters. Defaults match the Gemma4 checkpoint's
/// processor_config.json (patch 16, pooling 3, 280 max soft tokens, rescale
/// 1/255, do_normalize=false).
struct GENIEXPROC_API Gemma4Config {
    int   patch_size          = 16;
    int   max_soft_tokens     = 280;  // max_patches = max_soft_tokens * pooling^2 = 2520
    int   pooling_kernel_size = 3;
    float rescale_factor      = 1.0f / 255.0f;

    /// Resize every image to this square size before the aspect-preserving step.
    ///
    /// The exported VEG graph is traced at a FIXED soft-token count (256 for the
    /// v73 export), which only holds for a square image: 768x768 -> 48x48 patches
    /// -> 2304/9 = 256. Feeding a non-square image would change the soft-token
    /// count and no longer match the graph's output shape.
    ///
    /// 768 is deliberate: the aspect-preserving resize *targets* 768x768 for any
    /// square input, so at 768 it is a no-op and the image is resampled exactly
    /// once, from the original pixels. Pre-squaring to 448 instead (as the
    /// original reference script did) makes the processor upscale straight back
    /// to 768 — a second resample that only blurs detail.
    ///
    /// Set to 0 to disable and let the true aspect-preserving path run.
    int force_square_size = 768;

    /// Special tokens delimiting the image span. Defaults are Gemma4's.
    std::string boi_token   = "<|image>";   // begin-of-image
    std::string image_token = "<|image|>";  // one per soft token
    std::string eoi_token   = "<image|>";   // end-of-image

    int max_patches() const { return max_soft_tokens * pooling_kernel_size * pooling_kernel_size; }
    int patch_dim() const { return patch_size * patch_size * 3; }
};

// ============================================================
// Processor
// ============================================================

/// Processor for Gemma4 VLM. Composes a Tokenizer, image preprocessing, and the
/// Gemma4 chat template.
///
/// process() expands each image marker into `boi_token` + N x `image_token` +
/// `eoi_token`, where N is the soft-token count implied by that image's geometry
/// — exactly what Gemma4Processor.replace_image_token() does upstream. The
/// resulting image-token run is contiguous, which is what lets the runtime splice
/// the VEG output into `inputs_embeds` by position.
class GENIEXPROC_API Gemma4Processor : public geniex::VisionProcessor {
public:
    /// @param tokenizer_path      path to tokenizer.json
    /// @param tokenizer_config_path path to tokenizer_config.json (chat template);
    ///        when empty, apply_chat_template() throws.
    /// @throws std::runtime_error if the tokenizer cannot be loaded.
    static std::unique_ptr<Gemma4Processor> create(
        const std::string& tokenizer_path,
        const std::string& tokenizer_config_path = {},
        const Gemma4Config& config = {},
        std::string image_marker_override = {});

    ~Gemma4Processor() override;

    Gemma4Processor(const Gemma4Processor&)            = delete;
    Gemma4Processor& operator=(const Gemma4Processor&) = delete;

    /// Access the underlying tokenizer (encode / decode / is_eog).
    geniex::Tokenizer& tokenizer();

    const Gemma4Config& config() const;

    std::string apply_chat_template(
        const std::vector<geniex::ChatMessage>& messages,
        bool add_generation_prompt = true) const override;

    /// @throws std::runtime_error if the marker count does not match image_paths.
    BatchFeatures process(
        const std::string& formatted_text,
        const std::vector<std::string>& image_paths) override;

    /// Preprocess images only — no text. Useful for driving the VEG graph
    /// directly, and for numerical comparison against the Python reference.
    /// Fills pixel_values / image_position_ids / num_soft_tokens_per_image.
    BatchFeatures process_images(const std::vector<std::string>& image_paths);

protected:
    Gemma4Processor() = default;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Gemma4Processor(std::unique_ptr<Impl> impl, std::string image_marker_override);
};

}  // namespace geniex::gemma4
