// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// geniex::Tokenizer implementation
// Wraps tokenizers-cpp behind a pimpl to prevent tokenizers_cpp.h from leaking
// into consumer translation units.

#include "geniex-proc/tokenizer.h"

#include <stdexcept>
#include <string>

// Internal includes — never exposed in public headers
#include "src/internal/token_bytes.h"
#include "src/internal/utils.h"
#include "src/tokenizer/tokenizer_config.h"

#include <tokenizers_cpp.h>

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include "geniex-sampling/sampling-vocab.h"

namespace geniex {

// ============================================================
// Tokenizer::Impl
// ============================================================

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;

    // Vocab interface for use by Sampler/Grammar — created lazily on first access
    mutable std::unique_ptr<geniex_vocab_interface> vocab;

    // Chat-template support. Populated only when tokenizer_config_path
    // is non-empty AND the file contains a non-empty `chat_template` field.
    std::unique_ptr<minja::chat_template> chat_template;

    Impl(const std::string& tokenizer_path,
         const std::string& tokenizer_config_path) {
        tok = tokenizers::Tokenizer::FromBlobJSON(read_file_to_string(tokenizer_path));
        if (!tok) {
            throw std::runtime_error("geniex::Tokenizer: failed to load tokenizer from: " + tokenizer_path);
        }
        // Build vocab immediately so special tokens are cached
        vocab = std::unique_ptr<geniex_vocab_interface>(
            create_geniex_vocab_tokenizers(tok.get())
        );

        if (!tokenizer_config_path.empty()) {
            auto cfg = geniex::internal::load_tokenizer_config(tokenizer_config_path);
            if (!cfg.chat_template.empty()) {
                // minja's ctor runs a capability-detection pass that renders
                // the template several times against dummy contexts; do it
                // once at load time and cache the result.
                chat_template = std::make_unique<minja::chat_template>(
                    cfg.chat_template,
                    cfg.bos_token,
                    cfg.eos_token);
            }
        }
    }

    geniex_vocab_interface* get_vocab() const {
        return vocab.get();
    }
};

// ============================================================
// Tokenizer public API
// ============================================================

Tokenizer::Tokenizer(const std::string& tokenizer_path,
                     const std::string& tokenizer_config_path)
    : impl_(std::make_unique<Impl>(tokenizer_path, tokenizer_config_path)) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void* Tokenizer::get_vocab_ptr() const {
    return impl_->get_vocab();
}

std::unique_ptr<Tokenizer> Tokenizer::from_file(const std::string& tokenizer_path,
                                                const std::string& tokenizer_config_path) {
    return std::unique_ptr<Tokenizer>(new Tokenizer(tokenizer_path, tokenizer_config_path));
}

std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_special_tokens) const {
    return impl_->tok->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                               bool skip_special_tokens) const {
    return impl_->tok->Decode(ids);
}

std::string Tokenizer::decode_token(int32_t token_id, bool stream_safe_utf8) const {
    if (stream_safe_utf8) {
        return geniex::internal::token_id_to_raw_bytes(impl_->tok.get(), token_id);
    }
    return impl_->tok->Decode({token_id});
}

bool Tokenizer::has_chat_template() const noexcept {
    return impl_->chat_template != nullptr;
}

namespace {

using ordered_json = nlohmann::ordered_json;

// Parses a (possibly empty) caller-supplied JSON string. `field_name` is
// used in the error message so a bad input names itself.
ordered_json parse_optional_json(const std::string& s, const char* field_name) {
    if (s.empty()) {
        return ordered_json();
    }
    try {
        return ordered_json::parse(s);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("geniex::Tokenizer::apply_chat_template: failed to parse ")
            + field_name + ": " + e.what());
    }
}

// Resolves the two-form tool input on ApplyChatTemplateOptions into the
// JSON value minja's chat_template_inputs::tools expects. `tools_json`
// wins when non-empty so callers that already hold OpenAI-shaped JSON
// (FFI / HTTP) skip the round trip through ChatTool.
ordered_json build_tools_json(const ApplyChatTemplateOptions& opts) {
    if (!opts.tools_json.empty()) {
        return parse_optional_json(opts.tools_json, "tools_json");
    }
    if (opts.tools.empty()) {
        return ordered_json();
    }
    ordered_json out = ordered_json::array();
    for (const auto& t : opts.tools) {
        ordered_json fn;
        fn["name"]        = t.name;
        fn["description"] = t.description;
        if (t.parameters_json.empty()) {
            fn["parameters"] = ordered_json::object();
        } else {
            try {
                fn["parameters"] = ordered_json::parse(t.parameters_json);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "geniex::Tokenizer::apply_chat_template: tools[" + t.name
                    + "].parameters_json failed to parse: " + e.what());
            }
        }
        out.push_back({{"type", "function"}, {"function", std::move(fn)}});
    }
    return out;
}

// Converts one ChatMessage into the dict shape HF chat templates index into:
// `role`, `content`, and the tool/reasoning fields when set.
ordered_json message_to_json(const ChatMessage& m, std::size_t index) {
    ordered_json msg;
    msg["role"]    = role_to_string(m.role);
    msg["content"] = m.content;

    if (!m.tool_calls.empty()) {
        ordered_json tool_calls = ordered_json::array();
        for (const auto& tc : m.tool_calls) {
            ordered_json args;
            if (tc.arguments_json.empty()) {
                args = ordered_json::object();
            } else {
                try {
                    args = ordered_json::parse(tc.arguments_json);
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        "geniex::Tokenizer::apply_chat_template: messages["
                        + std::to_string(index) + "].tool_calls[" + tc.name
                        + "].arguments_json failed to parse: " + e.what());
                }
            }
            ordered_json fn;
            fn["name"]      = tc.name;
            fn["arguments"] = std::move(args);

            ordered_json call;
            if (!tc.id.empty()) call["id"] = tc.id;
            call["type"]     = "function";
            call["function"] = std::move(fn);

            tool_calls.push_back(std::move(call));
        }
        msg["tool_calls"] = std::move(tool_calls);
    }
    if (!m.tool_call_id.empty())      msg["tool_call_id"]      = m.tool_call_id;
    if (!m.name.empty())              msg["name"]              = m.name;
    if (!m.reasoning_content.empty()) msg["reasoning_content"] = m.reasoning_content;

    return msg;
}

}  // namespace

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    const ApplyChatTemplateOptions& opts) const
{
    if (!impl_->chat_template) {
        throw std::runtime_error(
            "geniex::Tokenizer::apply_chat_template: no chat template loaded "
            "(pass tokenizer_config_path to from_file())");
    }

    ordered_json messages_json = ordered_json::array();
    for (std::size_t i = 0; i < messages.size(); ++i) {
        messages_json.push_back(message_to_json(messages[i], i));
    }

    minja::chat_template_inputs inputs;
    inputs.messages              = std::move(messages_json);
    inputs.tools                 = build_tools_json(opts);
    inputs.add_generation_prompt = opts.add_generation_prompt;

    // extra_context_json provides the base; typed fields take precedence.
    ordered_json extra = opts.extra_context_json.empty()
        ? ordered_json::object()
        : parse_optional_json(opts.extra_context_json, "extra_context_json");
    extra["enable_thinking"] = opts.enable_thinking;
    inputs.extra_context = std::move(extra);

    try {
        if (!opts.chat_template_override.empty()) {
            minja::chat_template tmpl(
                opts.chat_template_override,
                impl_->chat_template->bos_token(),
                impl_->chat_template->eos_token());
            return tmpl.apply(inputs);
        }
        return impl_->chat_template->apply(inputs);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("geniex::Tokenizer::apply_chat_template: render failed: ") + e.what());
    }
}

bool Tokenizer::is_eog(int32_t token_id) const {
    return impl_->get_vocab()->is_eog(token_id);
}

int32_t Tokenizer::vocab_size() const {
    return impl_->get_vocab()->n_tokens();
}

std::string Tokenizer::id_to_piece(int32_t token_id) const {
    return impl_->get_vocab()->token_to_piece_str(token_id);
}

int32_t Tokenizer::piece_to_id(const std::string& piece) const {
    return impl_->tok->TokenToId(piece);
}

int32_t Tokenizer::token_bos() const {
    return impl_->get_vocab()->token_bos();
}

int32_t Tokenizer::token_eos() const {
    return impl_->get_vocab()->token_eos();
}

int32_t Tokenizer::token_pad() const {
    return impl_->get_vocab()->token_pad();
}

} // namespace geniex
