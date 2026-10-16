// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for src/vision/vision.h helpers.

#include "vision/vision.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace v = geniex::vision;

TEST(VisionResize, IdentityRgb) {
    const xt::xtensor<uint8_t, 3> image = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    const auto resized = v::resize_image(image, 2, 2);
    EXPECT_EQ(resized.shape(), image.shape());
    EXPECT_EQ(resized, image);
}

TEST(VisionResize, ConstantImageRemainsConstant) {
    const xt::xtensor<uint8_t, 3> image = xt::ones<uint8_t>({2, 3, 3}) * 77;
    const auto resized = v::resize_image(image, 4, 5);
    EXPECT_EQ(resized.shape(0), 4U);
    EXPECT_EQ(resized.shape(1), 5U);
    EXPECT_EQ(resized.shape(2), 3U);
    EXPECT_TRUE(std::all_of(resized.cbegin(), resized.cend(), [](uint8_t value) { return value == 77; }));
}

TEST(VisionResize, RejectsInvalidInputs) {
    const xt::xtensor<uint8_t, 3> image = {{{1, 2, 3}}};
    const auto empty = xt::xtensor<uint8_t, 3>::from_shape({0, 1, 3});
    const auto grayscale = xt::xtensor<uint8_t, 3>::from_shape({1, 1, 1});
    EXPECT_THROW(v::resize_image(empty, 1, 1), std::invalid_argument);
    EXPECT_THROW(v::resize_image(grayscale, 1, 1), std::invalid_argument);
    EXPECT_THROW(v::resize_image(image, 0, 1), std::invalid_argument);
    EXPECT_THROW(v::resize_image(image, 1, -1), std::invalid_argument);
    EXPECT_THROW(v::resize_image(image, INT_MAX, INT_MAX), std::invalid_argument);
}

TEST(VisionResize, DownsampleMatchesCpuAntialiasOracle) {
    auto image = xt::xtensor<uint8_t, 3>::from_shape({47, 63, 3});
    for (std::size_t index = 0; index < image.size(); ++index) {
        image.data()[index] = static_cast<uint8_t>((index * 37 + index / 11 + 91) % 256);
    }
    const auto resized = v::resize_image(image, 23, 19);
    ASSERT_EQ(resized.shape(0), 23U);
    ASSERT_EQ(resized.shape(1), 19U);
    ASSERT_EQ(resized.shape(2), 3U);

    // CPU Catmull-Rom oracle; allow one uint8 level for floating-point rounding.
    const std::size_t offsets[] = {0, 54, 654, 1254, 1308};
    const uint8_t expected[][3] = {{116, 151, 121}, {128, 125, 130}, {110, 140, 148}, {144, 94, 135}, {122, 123, 151}};
    for (std::size_t sample = 0; sample < 5; ++sample) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            EXPECT_NEAR(resized.data()[offsets[sample] + channel], expected[sample][channel], 1)
                << "sample " << sample << ", channel " << channel;
        }
    }
}

TEST(VisionResize, NonconstantCatmullRomVector) {
    const xt::xtensor<uint8_t, 3> image = {
        {{0, 0, 0}, {255, 0, 0}},
        {{0, 255, 0}, {0, 0, 255}},
    };
    const std::vector<uint8_t> expected = {
        0, 0, 1, 135, 0, 0, 255, 1, 0, 0, 135, 0, 64, 64, 64, 135, 0, 135, 1, 255, 0, 0, 135, 135, 0, 0, 255,
    };
    const auto resized = v::resize_image(image, 3, 3);
    EXPECT_EQ(std::vector<uint8_t>(resized.cbegin(), resized.cend()), expected);
}

// ─── round_by_factor / ceil_by_factor / floor_by_factor ──────────────────────

TEST(VisionFactor, RoundByFactorRoundsHalfwayAwayFromZero) {
    // 10 / 28 = 0.357 → round to 0 → 0*28 = 0
    EXPECT_EQ(v::round_by_factor(10, 28), 0);
    // 14 / 28 = 0.5 → std::round → 1 → 1*28 = 28 (on platforms where round
    // does round-half-away-from-zero, which is standard in C++11+).
    EXPECT_EQ(v::round_by_factor(14, 28), 28);
    // 15 / 28 = 0.536 → round to 1 → 28
    EXPECT_EQ(v::round_by_factor(15, 28), 28);
    EXPECT_EQ(v::round_by_factor(56, 28), 56);  // exact multiple
}

TEST(VisionFactor, CeilByFactor) {
    EXPECT_EQ(v::ceil_by_factor(1, 28), 28);
    EXPECT_EQ(v::ceil_by_factor(28, 28), 28);  // exact
    EXPECT_EQ(v::ceil_by_factor(29, 28), 56);
}

TEST(VisionFactor, FloorByFactor) {
    EXPECT_EQ(v::floor_by_factor(27, 28), 0);
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
    const double in_ratio = 1024.0 / 512.0;
    const double out_ratio = static_cast<double>(w) / h;
    EXPECT_NEAR(out_ratio, in_ratio, 0.25);
}

TEST(VisionSmartResize, ThrowsOnExtremeAspectRatio) {
    constexpr int factor = 28;
    constexpr int min_px = 4 * 28 * 28;
    constexpr int max_px = 1280 * 28 * 28;

    // max/min = 201/1 = 201 > 200 → should throw.
    EXPECT_THROW(v::smart_resize(201, 1, factor, min_px, max_px), std::runtime_error);
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
