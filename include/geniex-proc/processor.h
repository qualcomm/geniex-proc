#pragma once

// geniex-proc/processor.h — Abstract processor interfaces.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/types.h"

namespace geniex {

// ============================================================
// VisionProcessorInput
// ============================================================

/// Input to a vision processor.
/// Images are supplied via ChatMessage::mm_content_paths per message.
struct GENIEXPROC_API VisionProcessorInput {
    /// Conversation history.
    std::vector<ChatMessage> messages;

    /// If true, appends the assistant generation prompt after the last message.
    bool add_generation_prompt = true;

    // Future optional fields:
    // std::optional<int>  max_image_tokens;
    // std::optional<bool> do_resize;
};

// ============================================================
// VisionProcessor
// ============================================================

/// Abstract base for vision-language processors.
/// Concrete implementations: Qwen2VLProcessor, ...
class GENIEXPROC_API VisionProcessor {
public:
    VisionProcessor()                                  = default;
    virtual ~VisionProcessor()                         = default;
    VisionProcessor(const VisionProcessor&)            = delete;
    VisionProcessor& operator=(const VisionProcessor&) = delete;

    /// Process a conversation with optional images into model-ready inputs.
    virtual BatchFeatures process(const VisionProcessorInput& input) = 0;

    /// Instantiate the appropriate VisionProcessor from a model directory.
    /// Inspects the config file in model_dir to determine the processor type.
    /// @throws std::runtime_error — not yet implemented.
    static std::unique_ptr<VisionProcessor> from_config(const std::string& model_dir);
};

} // namespace geniex
