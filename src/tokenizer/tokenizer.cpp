// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0
//
// geniex::Tokenizer implementation
// Wraps tokenizers-cpp behind a pimpl to prevent tokenizers_cpp.h from leaking
// into consumer translation units.

#include "geniex-proc/tokenizer.h"

#include <fstream>
#include <stdexcept>
#include <string>

// Internal includes — never exposed in public headers
#include <tokenizers_cpp.h>

#include "geniex-sampling/sampling-vocab.h"

namespace geniex {

// ============================================================
// Tokenizer::Impl
// ============================================================

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;

    // Vocab interface for use by Sampler/Grammar — created lazily on first access
    mutable std::unique_ptr<geniex_vocab_interface> vocab;

    explicit Impl(const std::string& path) {
        tok = tokenizers::Tokenizer::FromJSON(path);
        if (!tok) {
            throw std::runtime_error("geniex::Tokenizer: failed to load tokenizer from: " + path);
        }
        // Build vocab immediately so special tokens are cached
        vocab = std::unique_ptr<geniex_vocab_interface>(
            create_geniex_vocab_tokenizers(tok.get())
        );
    }

    geniex_vocab_interface* get_vocab() const {
        return vocab.get();
    }
};

// ============================================================
// Tokenizer public API
// ============================================================

Tokenizer::Tokenizer(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void* Tokenizer::get_vocab_ptr() const {
    return impl_->get_vocab();
}

std::unique_ptr<Tokenizer> Tokenizer::from_file(const std::string& path) {
    return std::unique_ptr<Tokenizer>(new Tokenizer(path));
}

std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_special_tokens) const {
    return impl_->tok->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                               bool skip_special_tokens) const {
    return impl_->tok->Decode(ids);
}

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt) const
{
    // Chat template application is not yet supported.
    // Each model family has its own Jinja template which requires a model-specific
    // Processor to apply correctly. Use the model-specific Processor instead.
    throw std::runtime_error(
        "geniex::Tokenizer::apply_chat_template is not supported. "
        "Use the model-specific Processor (e.g. Qwen3VLProcessor) to apply "
        "the chat template, or format the prompt string manually."
    );
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
