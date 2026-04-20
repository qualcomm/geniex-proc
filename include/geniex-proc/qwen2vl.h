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
};

// ============================================================
// Processor
// ============================================================

/// Processor for Qwen2-VL. Composes a Tokenizer, image preprocessing
/// (smart_resize → rescale → normalize → patch), and the Qwen2-VL chat template.
class GENIEXPROC_API Qwen2VLProcessor : public geniex::VisionProcessor {
public:
    /// Create a Qwen2VLProcessor from a tokenizer.json path and optional config.
    /// @throws std::runtime_error if the tokenizer file cannot be loaded.
    static std::unique_ptr<Qwen2VLProcessor> create(const std::string& tokenizer_path,
                                                    const Qwen2VLConfig& config = {});

    virtual ~Qwen2VLProcessor() = default;

    // Non-copyable
    Qwen2VLProcessor(const Qwen2VLProcessor&)            = delete;
    Qwen2VLProcessor& operator=(const Qwen2VLProcessor&) = delete;

    /// Access the underlying tokenizer (encode / decode / is_eog).
    geniex::Tokenizer& tokenizer();

    /// Process a conversation with optional images into model-ready inputs.
    /// Images are loaded from ChatMessage::mm_content_paths in message order.
    BatchFeatures process(const geniex::VisionProcessorInput& input) override;

protected:
    Qwen2VLProcessor() = default;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit Qwen2VLProcessor(std::unique_ptr<Impl> impl);
};

}  // namespace geniex::qwen2vl
