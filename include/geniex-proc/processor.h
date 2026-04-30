// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

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
// VisionProcessor
// ============================================================

inline constexpr const char* kDefaultImageMarker = "<__image__>";

/// Abstract base for vision-language processors.
///
/// Two-stage API:
///   1. apply_chat_template(messages, ...)     — pure text, emits a formatted
///      prompt with `image_marker()` inserted
///   2. process(formatted_text, image_paths)   — loads and preprocesses images,
///      then tokenizes the formatted string
///
/// Marker-to-image pairing is strictly positional: the i-th occurrence of
/// `image_marker()` in the formatted string corresponds to `image_paths[i]`.
///
/// Concrete implementations: Qwen2VLProcessor, ...
class GENIEXPROC_API VisionProcessor {
public:
    virtual ~VisionProcessor()                         = default;
    VisionProcessor(const VisionProcessor&)            = delete;
    VisionProcessor& operator=(const VisionProcessor&) = delete;

    /// The text sentinel emitted by apply_chat_template() in place of each
    /// image, and consumed positionally by process().
    const std::string& image_marker() const { return image_marker_; }

    /// Apply the model-specific chat template to `messages` and return the
    /// formatted prompt string.
    virtual std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true) const = 0;

    /// Tokenize `formatted_text` and preprocess each image in `image_paths`
    /// into model-ready inputs. The formatted string must contain exactly
    /// `image_paths.size()` occurrences of `image_marker()`.
    ///
    /// @throws std::runtime_error if the marker count does not match.
    virtual BatchFeatures process(
        const std::string& formatted_text,
        const std::vector<std::string>& image_paths) = 0;

    /// Instantiate the appropriate VisionProcessor from a model directory.
    /// Inspects the config file in model_dir to determine the processor type.
    /// @throws std::runtime_error — not yet implemented.
    static std::unique_ptr<VisionProcessor> from_config(const std::string& model_dir);

protected:
    /// Derived classes may pass an override; if empty, `kDefaultImageMarker`
    /// is used.
    explicit VisionProcessor(std::string image_marker_override = {})
        : image_marker_(image_marker_override.empty()
                            ? std::string(kDefaultImageMarker)
                            : std::move(image_marker_override)) {}

private:
    std::string image_marker_;
};

} // namespace geniex
