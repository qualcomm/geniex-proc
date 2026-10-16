// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <AEEStdErr.h>

#include <array>
#include <cstdio>
#include <string>

#include "geniex_proc_htp.h"
#include "htp/htp_runtime.h"
#include "internvl_normalize.h"

namespace geniex::internvl {

void normalize_chw(const uint8_t* input, float* output, size_t pixels, float scale, const float* mean,
                   const float* stddev) {
    validate_normalize(input, output, pixels, scale, mean, stddev);
    const std::array<float, 7> parameters = {scale, mean[0], mean[1], mean[2], stddev[0], stddev[1], stddev[2]};
    auto access = geniex::htp::acquire(geniex::htp::default_session());
    const int elements = static_cast<int>(pixels * 3);
    const auto result =
        geniex_proc_htp_normalize(access.handle, static_cast<int32_t>(pixels), parameters.data(),
                                  static_cast<int>(parameters.size()), input, elements, output, elements);
    if (result != AEE_SUCCESS) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%x", static_cast<unsigned>(result));
        throw std::runtime_error(std::string("geniex::internvl: HTP Normalize failed with AEEResult ") + code);
    }
}

}  // namespace geniex::internvl
