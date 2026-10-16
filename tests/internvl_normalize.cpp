// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "processors/internvl_normalize.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

void compare_normalize(size_t pixels, float scale, const float mean[3], const float stddev[3]) {
    std::vector<uint8_t> input(pixels * 3 + 2, 0x5a);
    for (size_t i = 0; i < pixels; ++i)
        for (size_t c = 0; c < 3; ++c) input[1 + 3 * i + c] = static_cast<uint8_t>((i + 87 * c) % 256);
    std::vector<float> output(pixels * 3 + 2, 12345.0f);
    geniex::internvl::normalize_chw(input.data() + 1, output.data() + 1, pixels, scale, mean, stddev);
    ASSERT_EQ(output.front(), 12345.0f);
    ASSERT_EQ(output.back(), 12345.0f);
    for (size_t c = 0; c < 3; ++c) {
        const float inv_std = 1.0f / stddev[c];
        for (size_t i = 0; i < pixels; ++i) {
            const float value = static_cast<float>(input[1 + 3 * i + c]) * scale;
            const float expected = (value - mean[c]) * inv_std;
            ASSERT_NEAR(output[1 + c * pixels + i], expected, 2e-6f * std::max(1.0f, std::abs(expected)))
                << "pixels=" << pixels << " channel=" << c << " pixel=" << i;
        }
    }
    EXPECT_EQ(input.front(), 0x5a);
    EXPECT_EQ(input.back(), 0x5a);
}

}  // namespace

TEST(InternVLNormalize, DeinterleavesChannelsExactly) {
    const float mean[3] = {0, 0, 0}, stddev[3] = {1, 1, 1};
    const uint8_t input[] = {0, 127, 255, 23, 42, 99};
    float output[6] = {};
    geniex::internvl::normalize_chw(input, output, 2, 1, mean, stddev);
    const float expected[] = {0, 23, 127, 42, 255, 99};
    for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(output[i], expected[i]);
}

TEST(InternVLNormalize, MatchesOriginalFormulaAcrossByteValuesAndTails) {
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stddev[3] = {0.229f, 0.224f, 0.225f};
    for (const size_t pixels : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 256u, 437u}) {
        ASSERT_NO_FATAL_FAILURE(compare_normalize(pixels, 1.0f / 255.0f, mean, stddev));
    }
}

TEST(InternVLNormalize, SupportsCustomParameters) {
    const float mean[3] = {0.1f, -0.2f, 0.3f}, stddev[3] = {0.5f, 2.0f, 0.125f};
    compare_normalize(37, 0.0125f, mean, stddev);
}

TEST(InternVLNormalize, RejectsInvalidArguments) {
    const uint8_t input[3] = {};
    float output[3] = {};
    const float mean[3] = {0, 0, 0}, stddev[3] = {1, 1, 1}, invalid_std[3] = {1, 0, 1};
    namespace n = geniex::internvl;
    EXPECT_THROW(n::normalize_chw(nullptr, output, 1, 1, mean, stddev), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, nullptr, 1, 1, mean, stddev), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, output, 0, 1, mean, stddev), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, output, static_cast<size_t>(INT_MAX), 1, mean, stddev), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, output, 1, 1, nullptr, stddev), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, output, 1, 1, mean, invalid_std), std::invalid_argument);
    EXPECT_THROW(n::normalize_chw(input, output, 1, std::numeric_limits<float>::quiet_NaN(), mean, stddev),
                 std::invalid_argument);
}
