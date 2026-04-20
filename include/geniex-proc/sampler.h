#pragma once

// geniex-proc/sampler.h
//
// Public Sampler + Grammar classes.
//
// Sampler is a standalone logits→token machine with zero tokenizer dependency
// by default. Grammar is a separate composable object that introduces the
// tokenizer dependency explicitly.
//
// Usage:
//   // Basic (no tokenizer)
//   auto sampler = geniex::Sampler(params);
//   auto token   = sampler.sample(logits);
//
//   // With DRY + EOS (tokenizer used at init only)
//   auto sampler = geniex::Sampler(params, *tokenizer);
//
//   // With grammar (ongoing tokenizer dependency via Grammar)
//   auto grammar = std::make_unique<geniex::Grammar>(grammar_str, *tokenizer);
//   sampler.set_grammar(std::move(grammar));

#include <memory>
#include <vector>

#include "geniex-proc/export.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace geniex {

// ============================================================
// Grammar
// ============================================================

/**
 * @brief Grammar constraint for structured generation (e.g. JSON, EBNF).
 *
 * Requires a Tokenizer for ongoing vocab lookups (token→string mapping is
 * needed to evaluate grammar rules at each sampling step).
 *
 * Create a Grammar and attach it to a Sampler via Sampler::set_grammar().
 */
class GENIEXPROC_API Grammar {
public:
    /**
     * @param grammar_str GBNF grammar string.
     * @param tokenizer   Tokenizer whose vocab is used for constraint evaluation.
     *                    Must outlive this Grammar object.
     * @param grammar_root Root rule name (default: "root").
     */
    Grammar(const std::string& grammar_str,
            Tokenizer& tokenizer,
            const std::string& grammar_root = "root");

    ~Grammar();

    // Non-copyable, movable
    Grammar(const Grammar&)            = delete;
    Grammar& operator=(const Grammar&) = delete;
    Grammar(Grammar&&) noexcept;
    Grammar& operator=(Grammar&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class Sampler;
};

// ============================================================
// Sampler
// ============================================================

/**
 * @brief Token sampler — pure logits→token with optional tokenizer dependency.
 *
 * Two construction modes:
 *
 *   1. No tokenizer (basic sampling only — no DRY, no EOS detection):
 *        Sampler sampler(params);
 *
 *   2. With tokenizer (DRY sequence breakers resolved at init, EOS detection enabled):
 *        Sampler sampler(params, *tokenizer);
 */
class GENIEXPROC_API Sampler {
public:
    /**
     * @brief Construct sampler from params.
     *
     * EOS detection uses params.eog_tokens — populate from tokenizer->token_eos() etc.
     * DRY sequence breakers use params.dry_sequence_breaker_tokens — pre-tokenize strings
     * via tokenizer->encode() before constructing. Sampler holds no tokenizer reference.
     */
    explicit Sampler(const geniex_sampler_params& params = geniex_sampler_params());

    ~Sampler();

    // Non-copyable, movable
    Sampler(const Sampler&)            = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) noexcept;
    Sampler& operator=(Sampler&&) noexcept;

    /**
     * @brief Sample the next token from logits.
     *
     * Applies the full sampler chain (penalties, DRY, temperature, top-k/p,
     * grammar if set) and returns the selected token ID.
     *
     * @param logits Raw logits from the model (size = vocab_size).
     * @param grammar_first Apply grammar mask before other samplers (default false).
     */
    geniex_token sample(const std::vector<float>& logits,
                        bool grammar_first = false);

    /**
     * @brief Greedy argmax — no sampler chain applied.
     */
    static geniex_token sample_greedy(const std::vector<float>& logits);

    /**
     * @brief Notify the sampler that a token was accepted.
     *
     * Must be called after sample() so penalty/DRY state is updated.
     */
    void accept(geniex_token token);

    /**
     * @brief Check if a token is an end-of-generation token.
     *
     * Returns false if no tokenizer was provided at construction.
     */
    bool is_eog(geniex_token token) const;

    /**
     * @brief Reset sampler state (clears token history, grammar state).
     */
    void reset();

    /**
     * @brief Seed the sampler's token history with existing context.
     *
     * Call before the generation loop when continuing from a prefix.
     */
    void init(const std::vector<int32_t>& input_ids);

    /**
     * @brief Attach a grammar constraint.
     *
     * Pass nullptr to detach grammar.
     * The Sampler takes ownership of the Grammar object.
     */
    void set_grammar(std::unique_ptr<Grammar> grammar);

    /** @brief Print the sampler chain description (for debugging). */
    void print_chain() const;

    /** @brief Print sampling performance counters. */
    void print_perf() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geniex
