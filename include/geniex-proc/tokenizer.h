// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/tokenizer.h
//
// Public Tokenizer wrapper. Hides tokenizers_cpp.h behind a pimpl.
//
// For LLM use, create directly via from_file().
// For VLM/Omni use, obtain via Processor::tokenizer().


#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/types.h"

namespace geniex {

// Per-call options for Tokenizer::apply_chat_template(), mirroring HuggingFace's
// PreTrainedTokenizerBase.apply_chat_template(). JSON fields are kept as strings
// so this header stays free of nlohmann::json.
//
// At namespace scope rather than nested in Tokenizer: the `= {}` default on
// apply_chat_template() needs the NSDMI for `add_generation_prompt`, which
// Clang rejects while the enclosing class is still incomplete.
struct GENIEXPROC_API ApplyChatTemplateOptions {
    // Append the assistant header so the model can start its reply.
    bool add_generation_prompt = true;

    // Overrides the template loaded from tokenizer_config.json; empty = use
    // the loaded template. Loaded bos/eos strings are still used.
    std::string chat_template_override;

    // OpenAI-style tools array, JSON-serialized; empty = no tools.
    //   [{"type":"function","function":{"name":...,"description":...,
    //     "parameters":{...JSON Schema...}}}, ...]
    std::string tools_json;

    // JSON-serialized Jinja context for model-specific switches such as
    // `enable_thinking` (Qwen3) or `reasoning_effort` (Mistral4).
    std::string extra_context_json;
};

class GENIEXPROC_API Tokenizer {
public:
    // Compatibility alias for callers spelled `Tokenizer::ApplyChatTemplateOptions`.
    using ApplyChatTemplateOptions = ::geniex::ApplyChatTemplateOptions;

    /**
     * @brief Create a Tokenizer from a tokenizer.json file path.
     * @param tokenizer_path Path to tokenizer.json (HuggingFace format).
     * @param tokenizer_config_path Optional path to tokenizer_config.json.
     *        When provided, the chat template (and bos/eos token strings)
     *        are loaded so apply_chat_template() can render prompts.
     *        When empty, apply_chat_template() throws.
     * @throws std::runtime_error if a provided file cannot be loaded or parsed.
     */
    static std::unique_ptr<Tokenizer> from_file(
        const std::string& tokenizer_path,
        const std::string& tokenizer_config_path = "");

    ~Tokenizer();

    // Non-copyable, movable
    Tokenizer(const Tokenizer&)            = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    /**
     * @brief Encode text to token IDs.
     * @param text Input text.
     * @param add_special_tokens Whether to add BOS/EOS tokens.
     */
    std::vector<int32_t> encode(const std::string& text,
                                bool add_special_tokens = false) const;

    /**
     * @brief Decode token IDs back to text.
     * @param ids Token ID sequence.
     * @param skip_special_tokens Whether to skip special tokens in output.
     */
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special_tokens = true) const;

    /**
     * @brief Decode a single token for use in a per-token streaming loop.
     *
     * @param token_id  The token id to decode.
     * @param stream_safe_utf8  When true (default): returns the *raw byte
     *        sequence* contributed by this token.
     *
     *        When false: byte-identical to `decode({token_id})`.
     */
    std::string decode_token(int32_t token_id, bool stream_safe_utf8 = true) const;

    // True iff a tokenizer_config.json with a non-empty `chat_template`
    // field was loaded.
    bool has_chat_template() const noexcept;

    // Render the loaded chat template against `messages` and return the
    // formatted prompt string.
    //
    // Throws std::runtime_error when:
    //   - has_chat_template() is false,
    //   - tools_json / extra_context_json / a ToolCall::arguments_json
    //     fails to parse,
    //   - the underlying Jinja template fails to render.
    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        const ApplyChatTemplateOptions& opts = {}) const;

    /**
     * @brief Check if a token ID is an end-of-generation token.
     *
     * Covers EOS, EOT, EOM and other model-specific stop tokens.
     */
    bool is_eog(int32_t token_id) const;

    /** @brief Vocabulary size. */
    int32_t vocab_size() const;

    /** @brief Convert a single token ID to its string piece (raw, not decoded). */
    std::string id_to_piece(int32_t token_id) const;

    /** @brief Convert a string piece to its token ID (-1 if not found). */
    int32_t piece_to_id(const std::string& piece) const;

    // Special token IDs (return GENIEX_TOKEN_NULL = -1 if not present)
    int32_t token_bos() const;
    int32_t token_eos() const;
    int32_t token_pad() const;

private:
    Tokenizer(const std::string& tokenizer_path,
              const std::string& tokenizer_config_path);

    // pimpl — hides tokenizers_cpp.h and geniex_vocab_interface
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Returns the internal geniex_vocab_interface* as void*.
    // Only Grammar uses this (via friend access).
    void* get_vocab_ptr() const;

    friend class Grammar;
};

} // namespace geniex
