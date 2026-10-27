// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/processors/qwen2vl.cpp — Qwen2-VL processor implementation.
//
// Compiled only when GENIEXPROC_ENABLE_VISION is ON.
//
// Architecture:
//   Qwen2VLProcessor owns a pimpl (Impl) that holds the Tokenizer and all
//   image processing state. Consumers never see image processor internals.

#include "geniex-proc/qwen2vl.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include "processors/tools.h"
#include "vision/preprocess_device.h"
#include "vision/vision.h"

namespace geniex::qwen2vl {

// ============================================================
// Special tokens (Qwen2-VL)
// ============================================================

static const std::string BOS_TOKEN = "<|im_start|>";
static const std::string EOS_TOKEN = "<|im_end|>";

// Special token IDs for Qwen2-VL / Qwen2.5-Omni
static constexpr int32_t VISION_START_ID = 151652;
static constexpr int32_t VISION_END_ID   = 151653;
static constexpr int32_t IMAGE_PAD_ID    = 151655;

namespace {
using json = nlohmann::ordered_json;
}  // namespace

// ============================================================
// Pimpl
// ============================================================

struct Qwen2VLProcessor::Impl {
    std::unique_ptr<geniex::Tokenizer> tokenizer_;
    Qwen2VLConfig config_;

    geniex::vision::PreprocessDevice preprocess_device_;

    explicit Impl(const std::string& tokenizer_path, const Qwen2VLConfig& config)
        : tokenizer_(geniex::Tokenizer::from_file(tokenizer_path)),
          config_(config),
          preprocess_device_(config.preprocess_device) {}

    // ------------------------------------------------------------------
    // Image preprocessing — mirrors qwen2vl-proc.cpp / qwen3-vl.cpp
    // ------------------------------------------------------------------

    /// Preprocess a single HWC uint8 image into patches and grid dimensions.
    std::tuple<xt::xarray<float>, xt::xarray<size_t>>
    preprocess_single_image(const xt::xtensor<uint8_t, 3>& image) const {
        return geniex::vision::preprocess_qwen2vl(preprocess_device_, image, config_);
    }

    // ------------------------------------------------------------------
    // Chat template — Qwen2-VL format
    // ------------------------------------------------------------------

    std::string build_template_text(const std::vector<geniex::ChatMessage>& messages,
                                    const geniex::ApplyChatTemplateOptions& opts, std::string_view image_marker) const {
        const json tools = geniex::internal::parse_tools<json>(opts);
        if (!tools.is_array()) GENIEXPROC_THROW("tools_json must be a JSON array");
        std::string out;
        if (!tools.empty()) {
            out += BOS_TOKEN + "system\n";
            if (!messages.empty() && messages.front().role == geniex::Role::System) {
                out += messages.front().content + "\n\n";
            }
            out +=
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within XML tags:\n ";
            for (const auto& tool : tools) {
                if (!tool.is_object() || !tool.contains("function") || !tool["function"].is_object() ||
                    !tool["function"].contains("name") || !tool["function"]["name"].is_string()) {
                    GENIEXPROC_THROW("tools must contain named functions");
                }
                const auto definition = tool.dump();
                if (!image_marker.empty() && definition.find(image_marker) != std::string::npos) {
                    GENIEXPROC_THROW("tools contain the reserved image_marker");
                }
                out += '\n' + definition;
            }
            out +=
                "\n \n\nFor each function call, return a json object with function name and arguments within "
                "<tool_call></tool_call> XML tags:\n<tool_call>\n"
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
            out += EOS_TOKEN + "\n";
        }
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            // Defensive: reject a literal marker inside user content — would
            // break positional replacement during process().
            if (!image_marker.empty() && msg.content.find(image_marker) != std::string::npos) {
                GENIEXPROC_THROW("ChatMessage::content contains the reserved image_marker '" +
                                 std::string(image_marker) + "'");
            }
            if (!tools.empty() && i == 0 && msg.role == geniex::Role::System) continue;

            if (msg.role == geniex::Role::Tool) {
                if (i == 0 || messages[i - 1].role != geniex::Role::Tool) out += BOS_TOKEN + "user";
                out += "\n<tool_response>\n";
                for (size_t j = 0; j < msg.mm_content.size(); ++j) out += image_marker;
                out += msg.content + "\n</tool_response>";
                if (i + 1 == messages.size() || messages[i + 1].role != geniex::Role::Tool) out += EOS_TOKEN + "\n";
                continue;
            }

            out += BOS_TOKEN;
            out += role_to_string(msg.role);
            out += '\n';

            for (size_t j = 0; j < msg.mm_content.size(); ++j) out += image_marker;
            out += msg.content;

            if (msg.role == geniex::Role::Assistant) {
                bool first_call = true;
                for (const auto& call : msg.tool_calls) {
                    const auto args = call.arguments_json.empty() ? json::object() : json::parse(call.arguments_json);
                    if (!msg.content.empty() || !first_call) out += '\n';
                    out += "<tool_call>\n{\"name\": " + json(call.name).dump() + ", \"arguments\": " + args.dump() +
                           "}\n</tool_call>";
                    first_call = false;
                }
            }
            out += EOS_TOKEN;
            out += '\n';
        }

        if (opts.add_generation_prompt) {
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

Qwen2VLProcessor::Qwen2VLProcessor() = default;

Qwen2VLProcessor::~Qwen2VLProcessor() = default;

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

std::string Qwen2VLProcessor::apply_chat_template(const std::vector<geniex::ChatMessage>& messages,
                                                  const geniex::ApplyChatTemplateOptions& opts) const {
    return impl_->build_template_text(messages, opts, image_marker());
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
        xt::xarray<float> patches;
        xt::xarray<size_t> grid_thw;
        std::tie(patches, grid_thw) = impl_->preprocess_single_image(image);

        // grid_thw shape [1, 3]: (T, H, W)
        size_t grid_t = grid_thw(0, 0);
        size_t grid_h = grid_thw(0, 1);
        size_t grid_w = grid_thw(0, 2);
        image_patch_counts.push_back(grid_t * grid_h * grid_w);

        if (pixel_values.shape().size() == 0 && image_grid_thw.shape().size() == 0) {
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
