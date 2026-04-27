// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp)
// Original work Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License (https://opensource.org/licenses/MIT)

#include "sampling-vocab.h"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>

struct geniex_vocab_tokenizers : public geniex_vocab_interface {
    tokenizers::Tokenizer* tokenizer;

    // Cache for special tokens
    mutable geniex_token special_bos_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_eos_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_eot_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_eom_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_unk_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_pad_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_nl_id = GENIEX_TOKEN_NULL;
    mutable bool special_tokens_cached = false;

    // set of all tokens that cause "end of generation"
    mutable std::set<geniex_token> special_eog_ids;

    // fim tokens
    mutable geniex_token special_fim_pre_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_fim_suf_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_fim_mid_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_fim_pad_id = GENIEX_TOKEN_NULL;
    mutable geniex_token special_fim_rep_id = GENIEX_TOKEN_NULL;  // repo
    mutable geniex_token special_fim_sep_id = GENIEX_TOKEN_NULL;  // file separator

    geniex_vocab_tokenizers(tokenizers::Tokenizer* tok) : tokenizer(tok) { cache_special_tokens(); }

    ~geniex_vocab_tokenizers() override = default;

    // Core tokenization functions
    int token_to_piece(geniex_token token, char* buf, int32_t length, bool special = false) override {
        if (!tokenizer || !buf || length <= 0) return 0;

        std::string piece = tokenizer->IdToToken(token);
        if (piece.empty()) return 0;

        int copy_len = std::min(static_cast<int>(piece.length()), length - 1);
        std::memcpy(buf, piece.c_str(), copy_len);
        buf[copy_len] = '\0';
        return copy_len;
    }

    std::string token_to_piece_str(geniex_token token) override { return detokenize({token}); }

    std::vector<geniex_token> tokenize(const std::string& text, bool add_special = false,
                                     bool parse_special = false) override {
        if (!tokenizer) return {};

        return tokenizer->Encode(text);
    }

    // Detokenization
    int32_t detokenize(const geniex_token* tokens, int32_t n_tokens, char* text, int32_t text_len_max,
                       bool remove_special = false, bool unparse_special = false) override {
        if (!tokenizer || !tokens || !text || text_len_max <= 0) return 0;

        std::vector<int32_t> ids;
        ids.reserve(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            ids.push_back(tokens[i]);
        }

        std::string decoded = tokenizer->Decode(ids);

        int32_t copy_len = std::min(static_cast<int32_t>(decoded.length()), text_len_max - 1);
        std::memcpy(text, decoded.c_str(), copy_len);
        text[copy_len] = '\0';
        return copy_len;
    }

    std::string detokenize(const std::vector<geniex_token>& tokens, bool special = false) override {
        if (!tokenizer) return "";

        return tokenizer->Decode(tokens);
    }

    int32_t n_tokens() override {
        if (!tokenizer) return 0;
        return static_cast<int32_t>(tokenizer->GetVocabSize());
    }

    // Token classification functions
    bool is_eog(geniex_token token) override {
        cache_special_tokens();
        return token != GENIEX_TOKEN_NULL && special_eog_ids.count(token) > 0;
    }

    bool is_control(geniex_token token) override {
        // Check if token represents a control character
        std::string piece = tokenizer->IdToToken(token);
        if (piece.empty()) return false;

        // Check for common control patterns
        if (piece.length() >= 2 && piece[0] == '<' && piece.back() == '>') {
            return true;
        }

        // Check for actual control characters
        for (char c : piece) {
            if (c < 32 && c != '\t' && c != '\n' && c != '\r') {
                return true;
            }
        }
        return false;
    }

    bool is_byte(geniex_token token) override {
        std::string piece = tokenizer->IdToToken(token);
        // Check if it's a byte token (common pattern: <0x??> or similar)
        return piece.length() >= 4 && piece.substr(0, 3) == "<0x";
    }

    bool is_normal(geniex_token token) override {
        return !is_control(token) && !is_byte(token) && !is_unknown(token) && !is_unused(token);
    }

    bool is_unknown(geniex_token token) override {
        cache_special_tokens();
        return token == special_unk_id;
    }

    bool is_user_defined(geniex_token token) override {
        // This would require additional metadata that's not available from tokenizers
        return false;
    }

    bool is_unused(geniex_token token) override {
        // Check if token is valid
        if (token < 0 || token >= n_tokens()) return true;
        std::string piece = tokenizer->IdToToken(token);
        return piece.empty();
    }

    // Special token functions
    geniex_token token_bos() override {
        cache_special_tokens();
        return special_bos_id;
    }

    geniex_token token_eos() override {
        cache_special_tokens();
        return special_eos_id;
    }

    geniex_token token_eot() override {
        cache_special_tokens();
        return special_eot_id;
    }

    geniex_token token_eom() override {
        cache_special_tokens();
        return special_eom_id;
    }

    geniex_token token_unk() override {
        cache_special_tokens();
        return special_unk_id;
    }

    geniex_token token_sep() override {
        auto id = tokenizer->TokenToId("[SEP]");
        if (id != GENIEX_TOKEN_NULL) return id;
        id = tokenizer->TokenToId("<sep>");
        if (id != GENIEX_TOKEN_NULL) return id;
        return -1;
    }

    geniex_token token_nl() override {
        cache_special_tokens();
        return special_nl_id;
    }

    geniex_token token_pad() override {
        cache_special_tokens();
        return special_pad_id;
    }

    // Fill-in-the-middle tokens
    geniex_token token_prefix() override {
        auto id = tokenizer->TokenToId("<|fim_prefix|>");
        if (id != GENIEX_TOKEN_NULL) return id;
        return tokenizer->TokenToId("<|prefix|>");
    }

    geniex_token token_middle() override {
        auto id = tokenizer->TokenToId("<|fim_middle|>");
        if (id != GENIEX_TOKEN_NULL) return id;
        return tokenizer->TokenToId("<|middle|>");
    }

    geniex_token token_suffix() override {
        auto id = tokenizer->TokenToId("<|fim_suffix|>");
        if (id != GENIEX_TOKEN_NULL) return id;
        return tokenizer->TokenToId("<|suffix|>");
    }

    geniex_token token_fim_pre() override { return special_fim_pre_id; }

    geniex_token token_fim_suf() override { return special_fim_suf_id; }

    geniex_token token_fim_mid() override { return special_fim_mid_id; }

    geniex_token token_fim_pad() override { return special_fim_pad_id; }

    geniex_token token_fim_rep() override { return special_fim_rep_id; }

    geniex_token token_fim_sep() override { return special_fim_sep_id; }

    void cache_special_tokens() const {
        if (special_tokens_cached) return;

        // Common special token strings to try
        std::vector<std::string> bos_candidates = {"<|im_start|>", "<s>", "<bos>", "<|startoftext|>",
                                                   "<|begin_of_text|>"};
        std::vector<std::string> eos_candidates = {"<|im_end|>", "</s>", "<eos>", "<end_of_turn>"};
        std::vector<std::string> eot_candidates = {"<|endoftext|>", "<|end_of_text|>", "<|eot_id|>",
                                                   "<|end|>",       "<EOT>",           "_<EOT>"};
        std::vector<std::string> eom_candidates = {"<|eom_id|>"};
        std::vector<std::string> unk_candidates = {"<unk>", "<UNK>", "[UNK]"};
        std::vector<std::string> pad_candidates = {"<pad>", "<PAD>", "[PAD]"};
        std::vector<std::string> nl_candidates = {"\n", "\\n"};

        // FIM tokens
        std::vector<std::string> fim_pre_candidates = {
            "<|fim_prefix|>",  // Qwen
            "<fim-prefix>",
            "<fim_prefix>",    // Granite
            "<｜fim▁begin｜>"  // DeepSeek
            "<PRE>",
            "▁<PRE>"  // CodeLlama
        };
        std::vector<std::string> fim_suf_candidates = {
            "<|fim_suffix|>",  // Qwen
            "<fim-suffix>",
            "<fim_suffix>",    // Granite
            "<｜fim▁hole｜>",  // DeepSeek
            "<SUF>",
            "▁<SUF>",  // CodeLlama
        };
        std::vector<std::string> fim_mid_candidates = {
            "<|fim_middle|>",  // Qwen
            "<fim-middle>",
            "<fim_middle>",   // Granite
            "<｜fim▁end｜>",  // DeepSeek
            "<MID>",
            "▁<MID>"  // CodeLlama
        };
        std::vector<std::string> fim_pad_candidates = {
            "<|fim_pad|>",  // Qwen
            "<fim-pad>",
            "<fim_pad>",  // Granite
            "<PAD>",
        };
        std::vector<std::string> fim_rep_candidates = {
            "<|fim_repo|>",  // Qwen
            "<|repo_name|>", "<fim-repo>", "<REPO>",
            "<reponame>"  // Granite
        };
        std::vector<std::string> fim_sep_candidates = {
            "<|fim_sep|>",  // Qwen
        };

        // Try to find BOS token
        for (const auto& candidate : bos_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_bos_id = id;
                break;
            }
        }

        // Try to find EOS token
        for (const auto& candidate : eos_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_eos_id = id;
                break;
            }
        }

        // Try to find EOT token
        for (const auto& candidate : eot_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_eot_id = id;
                break;
            }
        }

        // Try to find EOM token
        for (const auto& candidate : eom_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_eom_id = id;
                break;
            }
        }

        // Try to find UNK token
        for (const auto& candidate : unk_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_unk_id = id;
                break;
            }
        }

        // Try to find PAD token
        for (const auto& candidate : pad_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_pad_id = id;
                break;
            }
        }

        // Try to find newline token
        for (const auto& candidate : nl_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_nl_id = id;
                break;
            }
        }

        // Try to find FIM tokens
        for (const auto& candidate : fim_pre_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_fim_pre_id = id;
                break;
            }
        }

        for (const auto& candidate : fim_suf_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_fim_suf_id = id;
                break;
            }
        }

        for (const auto& candidate : fim_mid_candidates) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_fim_mid_id = id;
                break;
            }
        }

        // maintain a list of tokens that cause end-of-generation
        special_eog_ids.clear();

        if (special_fim_pad_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_fim_pad_id) == 0) {
            special_eog_ids.insert(special_fim_pad_id);
        }

        if (special_fim_rep_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_fim_rep_id) == 0) {
            special_eog_ids.insert(special_fim_rep_id);
        }

        if (special_fim_sep_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_fim_sep_id) == 0) {
            special_eog_ids.insert(special_fim_sep_id);
        }

        for (const auto& candidate : {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
                                      "<|eom_id|>", "<EOT>", "_<EOT>", "<|end_of_text|>", "</s>"}) {
            auto id = tokenizer->TokenToId(candidate);
            if (id != GENIEX_TOKEN_NULL) {
                special_eog_ids.insert(id);
            }
        }

        // sanity checks
        if (special_eos_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_eos_id) == 0) {
            special_eog_ids.insert(special_eos_id);
            std::cerr << "special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect"
                      << std::endl;
        }

        if (special_eot_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_eot_id) == 0) {
            special_eog_ids.insert(special_eot_id);
            std::cerr << "special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect"
                      << std::endl;
        }

        if (special_eom_id != GENIEX_TOKEN_NULL && special_eog_ids.count(special_eom_id) == 0) {
            special_eog_ids.insert(special_eom_id);
            std::cerr << "special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect"
                      << std::endl;
        }

        special_tokens_cached = true;
    }
};

geniex_vocab_interface* create_geniex_vocab_tokenizers(const std::string& vocab_path) {
    try {
        auto tokenizer = tokenizers::Tokenizer::FromJSON(vocab_path);
        if (!tokenizer) {
            return nullptr;
        }
        return new geniex_vocab_tokenizers(tokenizer.release());
    } catch (const std::exception& e) {
        return nullptr;
    }
}

geniex_vocab_interface* create_geniex_vocab_tokenizers(tokenizers::Tokenizer* tokenizer) {
    if (!tokenizer) {
        return nullptr;
    }
    return new geniex_vocab_tokenizers(tokenizer);
}