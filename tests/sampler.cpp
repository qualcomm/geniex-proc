// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for geniex::Sampler.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "geniex-proc/sampler.h"
#include "geniex-proc/types.h"

namespace {

std::vector<float> peak_logits(size_t vocab_size, int peak_idx, float peak = 10.f, float baseline = 0.f) {
    std::vector<float> logits(vocab_size, baseline);
    logits[peak_idx] = peak;
    return logits;
}

}  // namespace

// ─── sample_greedy ───────────────────────────────────────────────────────────

TEST(SamplerGreedy, PicksArgmax) {
    auto logits = peak_logits(100, /*peak_idx=*/42);
    EXPECT_EQ(geniex::Sampler::sample_greedy(logits), 42);
}

TEST(SamplerGreedy, PicksFirstOnTie) {
    // All zero — argmax by definition picks the first index.
    std::vector<float> logits(10, 0.f);
    EXPECT_EQ(geniex::Sampler::sample_greedy(logits), 0);
}

TEST(SamplerGreedy, HandlesNegativeLogits) {
    std::vector<float> logits = {-5.f, -3.f, -9.f, -1.f, -7.f};
    EXPECT_EQ(geniex::Sampler::sample_greedy(logits), 3);
}

// ─── Sampler::sample — deterministic when temperature == 0 ───────────────────

TEST(Sampler, TemperatureZeroIsGreedy) {
    geniex_sampler_params params;
    params.temp   = 0.0f;         // <= 0 → greedy
    params.seed   = 1234;
    geniex::Sampler sampler(params);

    auto logits = peak_logits(100, /*peak_idx=*/7);
    EXPECT_EQ(sampler.sample(logits), 7);
}

TEST(Sampler, SeededSamplingIsReproducible) {
    auto make_sampler = []() {
        geniex_sampler_params params;
        params.temp = 1.0f;
        params.seed = 42;
        params.top_k = 40;
        return geniex::Sampler(params);
    };

    // Flat logits so multiple tokens are plausible — the RNG is what decides.
    std::vector<float> logits(50, 1.0f);
    logits[10] = 5.0f;  // slight bias so sampling has some signal

    auto s1 = make_sampler();
    auto s2 = make_sampler();

    // 16 successive samples from the same seed must match exactly.
    for (int i = 0; i < 16; ++i) {
        auto t1 = s1.sample(logits);
        auto t2 = s2.sample(logits);
        EXPECT_EQ(t1, t2) << "divergence at step " << i;
        s1.accept(t1);
        s2.accept(t2);
    }
}

// ─── logit_bias forces a specific token ──────────────────────────────────────

TEST(Sampler, LogitBiasForcesToken) {
    geniex_sampler_params params;
    params.temp = 0.0f;  // greedy so bias dominates deterministically
    // Bias token 99 to +1000 so it outranks any baseline peak.
    params.logit_bias = {{99, 1000.0f}};
    geniex::Sampler sampler(params);

    auto logits = peak_logits(128, /*peak_idx=*/5, /*peak=*/50.f);
    EXPECT_EQ(sampler.sample(logits), 99);
}

// ─── is_eog ─────────────────────────────────────────────────────────────────

TEST(Sampler, EogUnsetWithoutEogTokens) {
    geniex_sampler_params params;
    geniex::Sampler sampler(params);

    EXPECT_FALSE(sampler.is_eog(0));
    EXPECT_FALSE(sampler.is_eog(151643));
}

TEST(Sampler, EogDetectsConfiguredTokens) {
    geniex_sampler_params params;
    params.eog_tokens = {2, 151643, 151645};
    geniex::Sampler sampler(params);

    EXPECT_TRUE (sampler.is_eog(2));
    EXPECT_TRUE (sampler.is_eog(151643));
    EXPECT_TRUE (sampler.is_eog(151645));
    EXPECT_FALSE(sampler.is_eog(0));
    EXPECT_FALSE(sampler.is_eog(42));
}

// ─── reset() clears token history ────────────────────────────────────────────

TEST(Sampler, ResetDoesNotThrow) {
    geniex_sampler_params params;
    params.temp = 0.0f;
    geniex::Sampler sampler(params);

    auto logits = peak_logits(50, 3);
    sampler.accept(sampler.sample(logits));
    sampler.accept(sampler.sample(logits));

    EXPECT_NO_THROW(sampler.reset());

    // Post-reset, greedy still works.
    EXPECT_EQ(sampler.sample(logits), 3);
}

// ─── init() seeds history without throwing ───────────────────────────────────

TEST(Sampler, InitAcceptsPriorTokens) {
    geniex_sampler_params params;
    params.temp = 0.0f;
    geniex::Sampler sampler(params);

    std::vector<int32_t> prefix = {1, 2, 3, 4, 5};
    EXPECT_NO_THROW(sampler.init(prefix));

    auto logits = peak_logits(50, 10);
    EXPECT_EQ(sampler.sample(logits), 10);
}
