// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp)
// Original work Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License (https://opensource.org/licenses/MIT)

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "geniex-proc/types.h"
#include "sampling-vocab.h"

// Forward declaration for grammar support
struct geniex_grammar;

// Core types
typedef struct geniex_token_data {
    geniex_token id;  // token id
    float logit;    // log-odds of the token
    float p;        // probability of the token
} geniex_token_data;

typedef struct geniex_token_data_array {
    geniex_token_data* data;
    size_t size;
    int64_t selected;  // index in the data array (not the token id)
    bool sorted;
} geniex_token_data_array;

typedef struct geniex_logit_bias {
    geniex_token token;
    float bias;
} geniex_logit_bias;

// Forward declarations
struct geniex_sampler;
struct geniex_sampler_params;
typedef void* geniex_sampler_context_t;

// Sampler interface
struct geniex_sampler_i {
    const char* (*name)(const struct geniex_sampler* smpl);
    void (*accept)(struct geniex_sampler* smpl, geniex_token token);
    void (*apply)(struct geniex_sampler* smpl, geniex_token_data_array* cur_p);
    void (*reset)(struct geniex_sampler* smpl);
    struct geniex_sampler* (*clone)(const struct geniex_sampler* smpl);
    void (*free)(struct geniex_sampler* smpl);
};

// Sampler (similar to llama_sampler in llama.cpp)
struct geniex_sampler {
    const struct geniex_sampler_i* iface;
    geniex_sampler_context_t ctx;
};

// Chain parameters
typedef struct geniex_sampler_chain_params {
    bool no_perf;  // whether to measure performance timings
} geniex_sampler_chain_params;

// Performance data
struct geniex_perf_sampler_data {
    double t_sample_ms;
    int32_t n_sample;
};

// Core sampler API
struct geniex_sampler* geniex_sampler_init(const struct geniex_sampler_i* iface, geniex_sampler_context_t ctx);
const char* geniex_sampler_name(const struct geniex_sampler* smpl);
void geniex_sampler_accept(struct geniex_sampler* smpl, geniex_token token);
void geniex_sampler_apply(struct geniex_sampler* smpl, geniex_token_data_array* cur_p);
void geniex_sampler_reset(struct geniex_sampler* smpl);
struct geniex_sampler* geniex_sampler_clone(const struct geniex_sampler* smpl);
void geniex_sampler_free(struct geniex_sampler* smpl);

// Sampler chain
struct geniex_sampler* geniex_sampler_chain_init(struct geniex_sampler_chain_params params);
void geniex_sampler_chain_add(struct geniex_sampler* chain, struct geniex_sampler* smpl);
struct geniex_sampler* geniex_sampler_chain_get(const struct geniex_sampler* chain, int32_t i);
int geniex_sampler_chain_n(const struct geniex_sampler* chain);
struct geniex_sampler* geniex_sampler_chain_remove(struct geniex_sampler* chain, int32_t i);

// Individual samplers
struct geniex_sampler* geniex_sampler_init_greedy();
struct geniex_sampler* geniex_sampler_init_dist(uint32_t seed);
struct geniex_sampler* geniex_sampler_init_softmax();
struct geniex_sampler* geniex_sampler_init_top_k(int32_t k);
struct geniex_sampler* geniex_sampler_init_top_p(float p, size_t min_keep);
struct geniex_sampler* geniex_sampler_init_min_p(float p, size_t min_keep);
struct geniex_sampler* geniex_sampler_init_typical(float p, size_t min_keep);
struct geniex_sampler* geniex_sampler_init_temp(float temp);
struct geniex_sampler* geniex_sampler_init_temp_ext(float temp, float delta, float exponent);
struct geniex_sampler* geniex_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed);
struct geniex_sampler* geniex_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m);
struct geniex_sampler* geniex_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta);
struct geniex_sampler* geniex_sampler_init_penalties(int32_t penalty_last_n, float penalty_repeat, float penalty_freq,
                                                 float penalty_present);
struct geniex_sampler* geniex_sampler_init_top_n_sigma(float n);
struct geniex_sampler* geniex_sampler_init_dry(geniex_vocab_interface* vocab, int32_t context_size, float dry_multiplier,
                                           float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n,
                                           const char** seq_breakers, size_t num_breakers);

// Variant accepting pre-tokenized breaker sequences — no vocab required.
struct geniex_sampler* geniex_sampler_init_dry_tokenized(
        int32_t context_size, float dry_multiplier, float dry_base,
        int32_t dry_allowed_length, int32_t dry_penalty_last_n,
        const geniex_token* const* breaker_seqs, const size_t* breaker_lens, size_t num_breakers);

struct geniex_sampler* geniex_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias,
                                                  const geniex_logit_bias* logit_bias);

// Grammar sampler (requires vocab interface)
struct geniex_sampler* geniex_sampler_init_grammar(geniex_vocab_interface* vocab, const char* grammar_str,
                                               const char* grammar_root);

// Infill sampler (requires vocab interface)
struct geniex_sampler* geniex_sampler_init_infill(geniex_vocab_interface* vocab);

// Utility functions
uint32_t geniex_sampler_get_seed(const struct geniex_sampler* smpl);

// Performance functions
struct geniex_perf_sampler_data geniex_perf_sampler(const struct geniex_sampler* chain);
void geniex_perf_sampler_print(const struct geniex_sampler* chain);
void geniex_perf_sampler_reset(struct geniex_sampler* chain);

// Default parameters
struct geniex_sampler_chain_params geniex_sampler_chain_default_params();

// ============================================================
// High-level sampler context (internal — used by geniex::Sampler)
// ============================================================

// Forward declaration — full definition in geniex-sampling.cpp
struct geniex_sampler_context;

struct geniex_sampler_context* geniex_sampler_init_context(const geniex_sampler_params& params,
                                                           geniex_vocab_interface* vocab);
struct geniex_sampler_context* geniex_sampler_context_clone(const struct geniex_sampler_context* sctx);
std::string geniex_sampler_context_print(const struct geniex_sampler_context* sctx);
geniex_token geniex_sampler_context_sample(struct geniex_sampler_context* sctx, const float* logits,
                                           int32_t n_vocab, bool grammar_first = false);
geniex_token geniex_sampler_context_sample_no_accept(struct geniex_sampler_context* sctx, const float* logits,
                                                     int32_t n_vocab, bool grammar_first = false);
void geniex_sampler_context_accept(struct geniex_sampler_context* sctx, geniex_token token);
void geniex_sampler_context_reset(struct geniex_sampler_context* sctx);
void geniex_sampler_context_free(struct geniex_sampler_context* sctx);
void geniex_sampler_context_set_grammar(struct geniex_sampler_context* sctx, struct geniex_sampler* grammar_sampler);
void geniex_perf_sampler_context_print(const struct geniex_sampler_context* sctx);
