// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "gpu_runtime.h"

namespace geniex::vision::gpu {

// Fused Qwen2VL patch math using buffers and cached kernels from the shared context.
xt::xarray<float> preprocess_qwen2vl_patches(GpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height,
                                             int width, float scale, const float* mean, const float* stddev,
                                             int patch_size, int merge_size, int temporal_patch_size);

}  // namespace geniex::vision::gpu
