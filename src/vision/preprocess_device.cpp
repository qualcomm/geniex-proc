// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "preprocess_device.h"

#include <stdexcept>
#include <string>

#ifdef GENIEXPROC_ENABLE_OPENCL
#include "gpu/gpu_runtime.h"
#endif
#ifdef GENIEXPROC_ENABLE_HTP
#include "npu/npu_runtime.h"
#endif
#include "vision.h"

namespace geniex::vision {

struct PreprocessDevice::Impl {
#ifdef GENIEXPROC_ENABLE_OPENCL
    std::unique_ptr<gpu::GpuRuntime> gpu;
#endif
#ifdef GENIEXPROC_ENABLE_HTP
    std::unique_ptr<npu::NpuRuntime> npu;
#endif
};

PreprocessDevice::PreprocessDevice(Type requested)
    : type_(resolve_preprocess_device(requested)), impl_(std::make_unique<Impl>()) {
    switch (type_) {
        case Type::CPU:
            break;
        case Type::GPU:
#ifdef GENIEXPROC_ENABLE_OPENCL
            impl_->gpu = std::make_unique<gpu::GpuRuntime>();
            break;
#else
            throw std::runtime_error("Image preprocessing GPU requires GENIEXPROC_ENABLE_OPENCL");
#endif
        case Type::NPU:
#ifdef GENIEXPROC_ENABLE_HTP
            impl_->npu = std::make_unique<npu::NpuRuntime>();
            break;
#else
            throw std::runtime_error("Image preprocessing NPU requires GENIEXPROC_ENABLE_HTP");
#endif
        default:
            throw std::invalid_argument("Unknown image preprocessing device");
    }
}

PreprocessDevice::~PreprocessDevice() = default;

gpu::GpuRuntime& PreprocessDevice::gpu_runtime() const {
#ifdef GENIEXPROC_ENABLE_OPENCL
    if (impl_->gpu) return *impl_->gpu;
#endif
    throw std::logic_error("Image preprocessing GPU context is unavailable");
}

npu::NpuRuntime& PreprocessDevice::npu_runtime() const {
#ifdef GENIEXPROC_ENABLE_HTP
    if (impl_->npu) return *impl_->npu;
#endif
    throw std::logic_error("Image preprocessing NPU context is unavailable");
}

}  // namespace geniex::vision
