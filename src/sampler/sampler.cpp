// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// geniex::Sampler and geniex::Grammar implementations
// These wrap the internal geniex_sampler_context and grammar machinery.
// All internal types (geniex_sampler_context, geniex_vocab_interface, etc.)
// are hidden behind pimpl and never exposed in public headers.

#include "geniex-proc/sampler.h"
#include "geniex-proc/tokenizer.h"

// Internal includes — never exposed in public headers
#include "geniex-sampling/geniex-sampling.h"
#include "geniex-sampling/sampling-vocab.h"

#include <algorithm>
#include <stdexcept>

namespace geniex {

// ============================================================
// Grammar::Impl
// ============================================================

struct Grammar::Impl {
    // Raw grammar sampler — owned by this struct
    struct geniex_sampler* grammar_sampler = nullptr;

    // Pointer to the vocab (non-owning, lifetime managed by Tokenizer)
    geniex_vocab_interface* vocab = nullptr;

    std::string grammar_str;
    std::string grammar_root;

    Impl(const std::string& grammar_str_,
         geniex_vocab_interface* vocab_,
         const std::string& grammar_root_)
        : vocab(vocab_)
        , grammar_str(grammar_str_)
        , grammar_root(grammar_root_)
    {
        if (!vocab) {
            throw std::runtime_error("geniex::Grammar: tokenizer vocab must not be null");
        }
        grammar_sampler = geniex_sampler_init_grammar(
            vocab, grammar_str.c_str(), grammar_root.c_str());
        if (!grammar_sampler) {
            throw std::runtime_error("geniex::Grammar: failed to initialise grammar sampler");
        }
    }

    ~Impl() {
        if (grammar_sampler) {
            geniex_sampler_free(grammar_sampler);
        }
    }
};

// ============================================================
// Grammar public API
// ============================================================

Grammar::Grammar(const std::string& grammar_str,
                 Tokenizer& tokenizer,
                 const std::string& grammar_root)
    : impl_(std::make_unique<Impl>(grammar_str,
                                   static_cast<geniex_vocab_interface*>(tokenizer.get_vocab_ptr()),
                                   grammar_root))
{}

Grammar::~Grammar() = default;
Grammar::Grammar(Grammar&&) noexcept = default;
Grammar& Grammar::operator=(Grammar&&) noexcept = default;

// ============================================================
// Sampler::Impl
// ============================================================

struct Sampler::Impl {
    geniex_sampler_params params;

    // EOG token set — populated from params.eog_tokens at construction
    std::vector<geniex_token> eog_tokens;

    // Grammar (optional, owned via set_grammar)
    std::unique_ptr<Grammar> grammar;

    // Sampler chain context
    struct geniex_sampler_context* sctx = nullptr;

    explicit Impl(const geniex_sampler_params& params_)
        : params(params_)
        , eog_tokens(params_.eog_tokens)
    {
        rebuild_context();
    }

    ~Impl() {
        geniex_sampler_context_free(sctx);
        sctx = nullptr;
    }

    void destroy_context() {
        geniex_sampler_context_free(sctx);
        sctx = nullptr;
    }

    void rebuild_context() {
        destroy_context();
        sctx = geniex_sampler_init_context(params, nullptr);
        if (!sctx) {
            throw std::runtime_error("geniex::Sampler: failed to initialise sampler context");
        }
        if (grammar) {
            attach_grammar_to_context();
        }
    }

    void attach_grammar_to_context() {
        if (!sctx || !grammar) return;
        // Clone the grammar sampler so context owns an independent copy
        geniex_sampler_context_set_grammar(
            sctx, geniex_sampler_clone(grammar->impl_->grammar_sampler));
    }
};

// ============================================================
// Sampler public API
// ============================================================

Sampler::Sampler(const geniex_sampler_params& params)
    : impl_(std::make_unique<Impl>(params))
{}

Sampler::~Sampler() = default;
Sampler::Sampler(Sampler&&) noexcept = default;
Sampler& Sampler::operator=(Sampler&&) noexcept = default;

geniex_token Sampler::sample(const std::vector<float>& logits, bool grammar_first) {
    if (!impl_->sctx) {
        throw std::runtime_error("geniex::Sampler: context not initialised");
    }
    return geniex_sampler_context_sample(
        impl_->sctx,
        logits.data(),
        static_cast<int32_t>(logits.size()),
        grammar_first);
}

geniex_token Sampler::sample_greedy(const std::vector<float>& logits) {
    return static_cast<geniex_token>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void Sampler::accept(geniex_token token) {
    if (impl_->sctx) {
        geniex_sampler_context_accept(impl_->sctx, token);
    }
}

bool Sampler::is_eog(geniex_token token) const {
    const auto& eog = impl_->eog_tokens;
    return std::find(eog.begin(), eog.end(), token) != eog.end();
}

void Sampler::reset() {
    if (impl_->sctx) {
        geniex_sampler_context_reset(impl_->sctx);
    }
}

void Sampler::init(const std::vector<int32_t>& input_ids) {
    if (!impl_->sctx) return;
    for (auto id : input_ids) {
        geniex_sampler_context_accept(impl_->sctx, id);
    }
}

void Sampler::set_grammar(std::unique_ptr<Grammar> grammar) {
    impl_->grammar = std::move(grammar);
    if (impl_->sctx) {
        if (impl_->grammar) {
            impl_->attach_grammar_to_context();
        } else {
            geniex_sampler_context_set_grammar(impl_->sctx, nullptr);
        }
    }
}

void Sampler::print_chain() const {
    if (impl_->sctx) {
        geniex_perf_sampler_context_print(impl_->sctx);
    }
}

void Sampler::print_perf() const {
    if (impl_->sctx) {
        geniex_perf_sampler_context_print(impl_->sctx);
    }
}

} // namespace geniex
