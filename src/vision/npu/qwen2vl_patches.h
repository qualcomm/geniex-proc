// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace geniex::vision::npu {

class NpuRuntime;

xt::xarray<float> preprocess_qwen2vl_patches(NpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height,
                                             int width, float scale, const float* mean, const float* stddev,
                                             int patch_size, int merge_size, int temporal_patch_size);

}  // namespace geniex::vision::npu
