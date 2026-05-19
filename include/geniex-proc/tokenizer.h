// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/tokenizer.h
//
// Public Tokenizer wrapper. Hides tokenizers_cpp.h behind a pimpl.
//
// For LLM use, create directly via from_file().
// For VLM/Omni use, obtain via Processor::tokenizer().
//
// Usage:
//   auto tokenizer = geniex::Tokenizer::from_file("tokenizer.json");
//   auto ids  = tokenizer->encode("Hello world");
//   auto text = tokenizer->decode(ids);

#include <memory>
#include <string>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/types.h"

namespace geniex {

class GENIEXPROC_API Tokenizer {
public:
    /**
     * @brief Create a Tokenizer from a tokenizer.json file path.
     * @param path Path to tokenizer.json (HuggingFace format).
     * @throws std::runtime_error if the file cannot be loaded.
     */
    static std::unique_ptr<Tokenizer> from_file(const std::string& path);

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

    /**
     * @brief Apply the model's chat template and return the formatted string.
     *
     * NOTE: Not yet supported on the base Tokenizer. Each model has its own
     * template logic; use the model-specific Processor instead, or format
     * the prompt string manually.
     *
     * @throws std::runtime_error always — not yet implemented.
     */
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt = true) const;

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
    explicit Tokenizer(const std::string& path);

    // pimpl — hides tokenizers_cpp.h and geniex_vocab_interface
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Returns the internal geniex_vocab_interface* as void*.
    // Only Grammar uses this (via friend access).
    void* get_vocab_ptr() const;

    friend class Grammar;
};

} // namespace geniex
