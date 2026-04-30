// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/qwen2vl.h — Qwen2-VL vision-language processor.

#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/processor.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex::qwen2vl {

// ============================================================
// Configuration
// ============================================================

/// Image processor parameters. All fields have HuggingFace-compatible defaults
/// for the Qwen2-VL model family (patch_size=14, CLIP mean/std, etc.).
struct GENIEXPROC_API Qwen2VLConfig {
    int64_t min_pixels        = 4 * 28 * 28;       // 3136
    int64_t max_pixels        = 1280 * 28 * 28;    // 1003520
    int     patch_size        = 14;
    int     temporal_patch_size = 2;
    int     merge_size        = 2;
    float   rescale_factor    = 1.0f / 255.0f;
    std::vector<float> image_mean = {0.48145466f, 0.4578275f,  0.40821073f};
    std::vector<float> image_std  = {0.26862954f, 0.26130258f, 0.27577711f};

    // When both are non-zero, smart_resize() is skipped and every image is
    // resized to exactly (fixed_height x fixed_width), ignoring aspect ratio.
    int     fixed_height      = 0;
    int     fixed_width       = 0;
};

// ============================================================
// Processor
// ============================================================

/// Processor for Qwen2-VL. Composes a Tokenizer, image preprocessing
/// (smart_resize → rescale → normalize → patch), and the Qwen2-VL chat template.
class GENIEXPROC_API Qwen2VLProcessor : public geniex::VisionProcessor {
public:
    /// Create a Qwen2VLProcessor from a tokenizer.json path and optional config.
    ///
    /// @param image_marker_override Optional override for the text sentinel
    ///        emitted per image by apply_chat_template(). Leave empty to use
    ///        geniex::kDefaultImageMarker.
    /// @throws std::runtime_error if the tokenizer file cannot be loaded.
    static std::unique_ptr<Qwen2VLProcessor> create(
        const std::string& tokenizer_path,
        const Qwen2VLConfig& config = {},
        std::string image_marker_override = {});

    virtual ~Qwen2VLProcessor();

    // Non-copyable
    Qwen2VLProcessor(const Qwen2VLProcessor&)            = delete;
    Qwen2VLProcessor& operator=(const Qwen2VLProcessor&) = delete;

    /// Access the underlying tokenizer (encode / decode / is_eog).
    geniex::Tokenizer& tokenizer();

    std::string apply_chat_template(
        const std::vector<geniex::ChatMessage>& messages,
        bool add_generation_prompt = true) const override;

    /// @throws std::runtime_error if the marker count does not match.
    BatchFeatures process(
        const std::string& formatted_text,
        const std::vector<std::string>& image_paths) override;

protected:
    Qwen2VLProcessor() = default;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Qwen2VLProcessor(std::unique_ptr<Impl> impl,
                     std::string image_marker_override);
};

}  // namespace geniex::qwen2vl
