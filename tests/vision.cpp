// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for src/vision/vision.h helpers.

#include <stdexcept>
#include <tuple>

#include <gtest/gtest.h>

#include "vision/vision.h"

namespace v = geniex::vision;

// ─── round_by_factor / ceil_by_factor / floor_by_factor ──────────────────────

TEST(VisionFactor, RoundByFactorRoundsHalfwayAwayFromZero) {
    // 10 / 28 = 0.357 → round to 0 → 0*28 = 0
    EXPECT_EQ(v::round_by_factor(10, 28),  0);
    // 14 / 28 = 0.5 → std::round → 1 → 1*28 = 28 (on platforms where round
    // does round-half-away-from-zero, which is standard in C++11+).
    EXPECT_EQ(v::round_by_factor(14, 28), 28);
    // 15 / 28 = 0.536 → round to 1 → 28
    EXPECT_EQ(v::round_by_factor(15, 28), 28);
    EXPECT_EQ(v::round_by_factor(56, 28), 56);  // exact multiple
}

TEST(VisionFactor, CeilByFactor) {
    EXPECT_EQ(v::ceil_by_factor(1,  28), 28);
    EXPECT_EQ(v::ceil_by_factor(28, 28), 28);  // exact
    EXPECT_EQ(v::ceil_by_factor(29, 28), 56);
}

TEST(VisionFactor, FloorByFactor) {
    EXPECT_EQ(v::floor_by_factor(27, 28),  0);
    EXPECT_EQ(v::floor_by_factor(28, 28), 28);  // exact
    EXPECT_EQ(v::floor_by_factor(55, 28), 28);
    EXPECT_EQ(v::floor_by_factor(56, 28), 56);  // exact
}

// ─── smart_resize — pure output invariants ───────────────────────────────────

TEST(VisionSmartResize, OutputDimsAreDivisibleByFactor) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;     // 3136
    constexpr int max_px = 1280 * 28 * 28;  // 1003520

    auto [h, w] = v::smart_resize(768, 1024, factor, min_px, max_px);
    EXPECT_EQ(h % factor, 0);
    EXPECT_EQ(w % factor, 0);
    EXPECT_GE(static_cast<int64_t>(h) * w, min_px);
    EXPECT_LE(static_cast<int64_t>(h) * w, max_px);
}

TEST(VisionSmartResize, HugeImageIsDownscaledToFitMaxPixels) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    auto [h, w] = v::smart_resize(4096, 4096, factor, min_px, max_px);
    EXPECT_LE(static_cast<int64_t>(h) * w, max_px);
    EXPECT_EQ(h % factor, 0);
    EXPECT_EQ(w % factor, 0);
}

TEST(VisionSmartResize, TinyImageIsUpscaledToMeetMinPixels) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    auto [h, w] = v::smart_resize(10, 10, factor, min_px, max_px);
    EXPECT_GE(static_cast<int64_t>(h) * w, min_px);
    EXPECT_EQ(h % factor, 0);
    EXPECT_EQ(w % factor, 0);
}

TEST(VisionSmartResize, AspectRatioIsRoughlyPreserved) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    // 2:1 input → expect output roughly 2:1 (tolerance for divisibility rounding).
    auto [h, w] = v::smart_resize(512, 1024, factor, min_px, max_px);
    const double in_ratio  = 1024.0 / 512.0;
    const double out_ratio = static_cast<double>(w) / h;
    EXPECT_NEAR(out_ratio, in_ratio, 0.25);
}

TEST(VisionSmartResize, ThrowsOnExtremeAspectRatio) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    // max/min = 201/1 = 201 > 200 → should throw.
    EXPECT_THROW(v::smart_resize(201, 1, factor, min_px, max_px),
                 std::runtime_error);
}

TEST(VisionSmartResize, ClampsToAtLeastOneFactorInEachDim) {
    // Very small input: output must still be >= factor in both dimensions
    // (the h_bar = std::max(factor, ...) clamp in the implementation).
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    auto [h, w] = v::smart_resize(1, 1, factor, min_px, max_px);
    EXPECT_GE(h, factor);
    EXPECT_GE(w, factor);
}
