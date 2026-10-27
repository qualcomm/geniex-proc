// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>

#include "geniex-proc/types.h"

namespace geniex::vision {
namespace gpu {
class GpuRuntime;
}
namespace npu {
class NpuRuntime;
}

// Resolves a requested backend once and owns its reusable hardware resources.
class PreprocessDevice {
   public:
    using Type = ::geniex::PreprocessDevice;

    explicit PreprocessDevice(Type requested);
    ~PreprocessDevice();
    PreprocessDevice(const PreprocessDevice&) = delete;
    PreprocessDevice& operator=(const PreprocessDevice&) = delete;

    Type type() const noexcept { return type_; }
    gpu::GpuRuntime& gpu_runtime() const;
    npu::NpuRuntime& npu_runtime() const;

   private:
    struct Impl;
    Type type_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex::vision
