// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "internvl_normalize.h"

namespace geniex::internvl {

void normalize_chw(const uint8_t* input, float* output, size_t pixels, float scale, const float* mean,
                   const float* stddev) {
    validate_normalize(input, output, pixels, scale, mean, stddev);
    for (size_t c = 0; c < 3; ++c) {
        const float inv_std = 1.0f / stddev[c];
        for (size_t i = 0; i < pixels; ++i) {
            const float value = static_cast<float>(input[3 * i + c]) * scale;
            output[c * pixels + i] = (value - mean[c]) * inv_std;
        }
    }
}

}  // namespace geniex::internvl
