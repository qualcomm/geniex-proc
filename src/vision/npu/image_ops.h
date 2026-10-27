// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <xtensor/containers/xtensor.hpp>

#include "npu_runtime.h"

namespace geniex::vision::npu {

// Reusable HTP base image operator. Qwen2VL uses the fused patch RPC instead.
xt::xtensor<uint8_t, 3> resize_rgb(NpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height, int width);

}  // namespace geniex::vision::npu
