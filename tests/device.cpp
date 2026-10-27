// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <xtensor/generators/xbuilder.hpp>

#include "vision/cpu/image_ops.h"
#include "vision/preprocess_device.h"
#include "vision/vision.h"
#ifdef GENIEXPROC_TEST_ENABLE_OPENCL
#include "vision/gpu/image_ops.h"
#endif
#ifdef GENIEXPROC_TEST_ENABLE_HTP
#include "vision/npu/image_ops.h"
#endif

namespace v = geniex::vision;

namespace {
class ScopedPreprocessDeviceEnv {
   public:
    explicit ScopedPreprocessDeviceEnv(const char* value) {
        const char* previous = std::getenv("GENIEX_QAIRT_PREPROCESS_DEVICE");
        had_previous_ = previous != nullptr;
        if (previous) previous_ = previous;
        set(value);
    }

    ~ScopedPreprocessDeviceEnv() { set(had_previous_ ? previous_.c_str() : nullptr); }

   private:
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("GENIEX_QAIRT_PREPROCESS_DEVICE", value ? value : "");
#else
        if (value)
            setenv("GENIEX_QAIRT_PREPROCESS_DEVICE", value, 1);
        else
            unsetenv("GENIEX_QAIRT_PREPROCESS_DEVICE");
#endif
    }

    bool had_previous_ = false;
    std::string previous_;
};

xt::xtensor<uint8_t, 3> constant_rgb(size_t height, size_t width, uint8_t value) {
    const std::array<size_t, 3> shape{height, width, 3};
    xt::xtensor<uint8_t, 3> image = xt::empty<uint8_t>(shape);
    image.fill(value);
    return image;
}
}  // namespace

TEST(DeviceSelection, EnvironmentAndExplicitChoice) {
    using Type = geniex::PreprocessDevice;
    {
        ScopedPreprocessDeviceEnv env(nullptr);
        EXPECT_EQ(v::resolve_preprocess_device(Type::Auto), Type::CPU);
    }
    for (const char* value : {"", "auto", "cpu"}) {
        ScopedPreprocessDeviceEnv env(value);
        EXPECT_EQ(v::resolve_preprocess_device(Type::Auto), Type::CPU);
    }
    {
        ScopedPreprocessDeviceEnv env("GPU");
        EXPECT_EQ(v::resolve_preprocess_device(Type::Auto), Type::GPU);
        v::PreprocessDevice explicit_cpu(Type::CPU);
        EXPECT_EQ(explicit_cpu.type(), Type::CPU);
    }
    {
        ScopedPreprocessDeviceEnv env("nPu");
        EXPECT_EQ(v::resolve_preprocess_device(Type::Auto), Type::NPU);
    }
    {
        ScopedPreprocessDeviceEnv env("unknown");
        EXPECT_THROW(v::PreprocessDevice{Type::Auto}, std::invalid_argument);
        EXPECT_EQ(v::PreprocessDevice(Type::CPU).type(), Type::CPU);
    }
}

TEST(DeviceSelection, RejectsUnbuiltBackends) {
    using Type = geniex::PreprocessDevice;
#ifndef GENIEXPROC_TEST_ENABLE_OPENCL
    EXPECT_THROW(v::PreprocessDevice{Type::GPU}, std::runtime_error);
#endif
#ifndef GENIEXPROC_TEST_ENABLE_HTP
    EXPECT_THROW(v::PreprocessDevice{Type::NPU}, std::runtime_error);
#endif
    EXPECT_THROW(v::PreprocessDevice(static_cast<Type>(99)), std::invalid_argument);
}

TEST(DeviceImageOperators, ExplicitCpuReferenceAndCheckedSizes) {
    using Type = geniex::PreprocessDevice;
    ScopedPreprocessDeviceEnv env("npu");
    v::PreprocessDevice cpu(Type::CPU);
    EXPECT_EQ(cpu.type(), Type::CPU);
    const auto image = constant_rgb(2, 2, 91);
    const auto resized = v::cpu::resize_image(image, 3, 3);
    EXPECT_EQ(resized.shape(0), 3u);
    EXPECT_EQ(resized.shape(1), 3u);
    for (const auto pixel : resized) EXPECT_EQ(pixel, 91);
    EXPECT_EQ(v::rgb_image_bytes(2, 3, sizeof(uint8_t)), 18u);
    EXPECT_EQ(v::rgb_image_bytes(2, 3, sizeof(float)), 72u);
    EXPECT_THROW(v::rgb_image_bytes(static_cast<size_t>(-1), 3, sizeof(uint8_t)), std::invalid_argument);
    EXPECT_THROW(v::rgb_image_bytes(static_cast<size_t>(std::numeric_limits<int>::max()),
                                    static_cast<size_t>(std::numeric_limits<int>::max()), sizeof(float)),
                 std::invalid_argument);
    EXPECT_THROW(v::checked_product(std::numeric_limits<size_t>::max(), 2, "product overflow"),
                 std::invalid_argument);
}

#ifdef GENIEXPROC_TEST_ENABLE_OPENCL
TEST(DeviceImageOperators, GpuRgbResizeExecutesOnSelectedDevice) {
    v::PreprocessDevice device(geniex::PreprocessDevice::GPU);
    const auto image = constant_rgb(2, 2, 91);
    const auto resized = v::gpu::resize_rgb(device.gpu_runtime(), image, 3, 3);
    EXPECT_EQ(resized.shape(0), 3u);
    EXPECT_EQ(resized.shape(1), 3u);
    for (const auto pixel : resized) EXPECT_EQ(pixel, 91);
}
#endif

#ifdef GENIEXPROC_TEST_ENABLE_HTP
TEST(DeviceImageOperators, NpuRgbResizeExecutesOnSelectedDevice) {
    v::PreprocessDevice device(geniex::PreprocessDevice::NPU);
    const auto image = constant_rgb(2, 2, 91);
    const auto resized = v::npu::resize_rgb(device.npu_runtime(), image, 3, 3);
    EXPECT_EQ(resized.shape(0), 3u);
    EXPECT_EQ(resized.shape(1), 3u);
    for (const auto pixel : resized) EXPECT_EQ(pixel, 91);
}
#endif
