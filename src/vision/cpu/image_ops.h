// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <xtensor/containers/xtensor.hpp>

namespace geniex::vision::cpu {

// Resize HWC uint8 RGB with Catmull-Rom interpolation.
xt::xtensor<uint8_t, 3> resize_image(const xt::xtensor<uint8_t, 3>& image, int resized_height, int resized_width);

}  // namespace geniex::vision::cpu
