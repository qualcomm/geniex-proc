// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp)
// Original work Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License (https://opensource.org/licenses/MIT)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "geniex-proc/types.h"

struct geniex_vocab_interface {
    virtual ~geniex_vocab_interface() = default;

    // Core tokenization functions
    virtual int token_to_piece(geniex_token token, char* buf, int32_t length, bool special = false) = 0;
    virtual std::vector<geniex_token> tokenize(const std::string& text, bool add_special = false,
                                             bool parse_special = false) = 0;

    // Convenience function for grammar - returns string directly
    virtual std::string token_to_piece_str(geniex_token token) = 0;

    // Detokenization
    virtual int32_t detokenize(const geniex_token* tokens, int32_t n_tokens, char* text, int32_t text_len_max,
                               bool remove_special = false, bool unparse_special = false) = 0;
    virtual std::string detokenize(const std::vector<geniex_token>& tokens, bool special = false) = 0;

    // number of tokens in the vocabulary
    virtual int32_t n_tokens() = 0;

    // Token classification functions
    virtual bool is_eog(geniex_token token) = 0;
    virtual bool is_control(geniex_token token) = 0;
    virtual bool is_byte(geniex_token token) = 0;
    virtual bool is_normal(geniex_token token) = 0;
    virtual bool is_unknown(geniex_token token) = 0;
    virtual bool is_user_defined(geniex_token token) = 0;
    virtual bool is_unused(geniex_token token) = 0;

    // Special token functions
    virtual geniex_token token_bos() = 0;
    virtual geniex_token token_eos() = 0;
    virtual geniex_token token_eot() = 0;
    virtual geniex_token token_eom() = 0;
    virtual geniex_token token_unk() = 0;
    virtual geniex_token token_sep() = 0;
    virtual geniex_token token_nl() = 0;
    virtual geniex_token token_pad() = 0;

    // Fill-in-the-middle tokens
    virtual geniex_token token_prefix() = 0;
    virtual geniex_token token_middle() = 0;
    virtual geniex_token token_suffix() = 0;
    virtual geniex_token token_fim_pre() = 0;
    virtual geniex_token token_fim_suf() = 0;
    virtual geniex_token token_fim_mid() = 0;
    virtual geniex_token token_fim_pad() = 0;
    virtual geniex_token token_fim_rep() = 0;
    virtual geniex_token token_fim_sep() = 0;
};

// Forward declaration to avoid including tokenizers_cpp.h here
namespace tokenizers { class Tokenizer; }

// Internal factory functions — not part of the public API
geniex_vocab_interface* create_geniex_vocab_tokenizers(const std::string& vocab_path);
geniex_vocab_interface* create_geniex_vocab_tokenizers(tokenizers::Tokenizer* tokenizer);