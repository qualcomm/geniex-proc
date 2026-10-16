// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace geniex::internvl {

inline void validate_normalize(const uint8_t* input, float* output, size_t pixels, float scale, const float* mean,
                               const float* stddev) {
    if (!input || !output || !mean || !stddev || pixels == 0 || pixels > static_cast<size_t>(INT_MAX) / 12 ||
        !std::isfinite(scale)) {
        throw std::invalid_argument("geniex::internvl: invalid Normalize buffers or dimensions");
    }
    for (size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(mean[c]) || !std::isfinite(stddev[c]) || stddev[c] == 0 ||
            !std::isfinite(1.0f / stddev[c])) {
            throw std::invalid_argument("geniex::internvl: invalid Normalize parameters");
        }
    }
}

void normalize_chw(const uint8_t* input, float* output, size_t pixels, float scale, const float* mean,
                   const float* stddev);

}  // namespace geniex::internvl
