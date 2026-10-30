// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <xtensor/containers/xtensor.hpp>

#include "gpu_runtime.h"

namespace geniex::vision::gpu {

// Reusable host-output RGB resize. Kernel and scratch buffers live in runtime.
xt::xtensor<uint8_t, 3> resize_rgb(GpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height, int width);

// Enqueue resize on an already locked runtime queue for fused device pipelines.
cl_mem enqueue_resize_rgb(GpuRuntime::Access& access, const xt::xtensor<uint8_t, 3>& image, int height, int width);

}  // namespace geniex::vision::gpu
