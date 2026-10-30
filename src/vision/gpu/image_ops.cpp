// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "image_ops.h"

#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_utils.h"
#include "../vision.h"

#include <array>
#include <stdexcept>
#include <string>
#include <xtensor/generators/xbuilder.hpp>

namespace geniex::vision::gpu {
namespace {

constexpr char kResizeKernel[] = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
float catmull_rom(float x) {
    x = fabs(x);
    if (x < 1.0f) return ((1.5f * x - 2.5f) * x * x + 1.0f);
    if (x < 2.0f) return (((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f);
    return 0.0f;
}
uchar saturate_u8(float value) {
    value = floor(value + 0.5f);
    if (value <= 0.0f) return (uchar)0;
    if (value >= 255.0f) return (uchar)255;
    return (uchar)value;
}
__kernel void resize_catmullrom_antialias(
    __global const uchar* input, __global uchar* output,
    const int source_width, const int source_height,
    const int destination_width, const int destination_height) {
    const size_t pixel = get_global_id(0);
    const int x = (int)(pixel % (size_t)destination_width);
    const int y = (int)(pixel / (size_t)destination_width);
    const float scale_x = fmax((float)source_width / (float)destination_width, 1.0f);
    const float scale_y = fmax((float)source_height / (float)destination_height, 1.0f);
    const float center_x = ((float)x + 0.5f) * (float)source_width / (float)destination_width - 0.5f;
    const float center_y = ((float)y + 0.5f) * (float)source_height / (float)destination_height - 0.5f;
    const int first_x = (int)ceil(center_x - 2.0f * scale_x);
    const int last_x = (int)floor(center_x + 2.0f * scale_x);
    const int first_y = (int)ceil(center_y - 2.0f * scale_y);
    const int last_y = (int)floor(center_y + 2.0f * scale_y);
    float value[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum = 0.0f;
    for (int raw_y = first_y; raw_y <= last_y; ++raw_y) {
        const int source_y = clamp(raw_y, 0, source_height - 1);
        const float weight_y = catmull_rom(((float)raw_y - center_y) / scale_y);
        for (int raw_x = first_x; raw_x <= last_x; ++raw_x) {
            const int source_x = clamp(raw_x, 0, source_width - 1);
            const float weight = weight_y * catmull_rom(((float)raw_x - center_x) / scale_x);
            const size_t offset = ((size_t)source_y * (size_t)source_width + (size_t)source_x) * 3;
            value[0] += (float)input[offset] * weight;
            value[1] += (float)input[offset + 1] * weight;
            value[2] += (float)input[offset + 2] * weight;
            weight_sum += weight;
        }
    }
    const size_t offset = pixel * 3;
    output[offset] = saturate_u8(value[0] / weight_sum);
    output[offset + 1] = saturate_u8(value[1] / weight_sum);
    output[offset + 2] = saturate_u8(value[2] / weight_sum);
}
)CLC";

}  // namespace

cl_mem enqueue_resize_rgb(GpuRuntime::Access& access, const xt::xtensor<uint8_t, 3>& image, int height, int width) {
    if (image.shape(2) != 3 || image.size() != rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t))) {
        throw std::invalid_argument("geniex::vision::gpu::GpuRgbResizer: expected HWC RGB image");
    }
    const size_t source_size = rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t));
    const size_t resized_size = rgb_image_bytes(static_cast<size_t>(height), static_cast<size_t>(width), sizeof(uint8_t));
    const cl_mem source = access.buffer("vision.rgb.source", source_size, CL_MEM_READ_ONLY);
    const cl_mem resized = access.buffer("vision.rgb.resized", resized_size, CL_MEM_READ_WRITE);
    opencl::check(clEnqueueWriteBuffer(access.queue(), source, CL_TRUE, 0, source_size, image.data(), 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(RGB image)");

    const int source_width = static_cast<int>(image.shape(1));
    const int source_height = static_cast<int>(image.shape(0));
    const cl_kernel kernel = access.kernel("vision.rgb.resize", kResizeKernel, "resize_catmullrom_antialias",
                                           "build RGB resize program / create RGB resize kernel");
    opencl::argument(kernel, 0, sizeof(cl_mem), &source);
    opencl::argument(kernel, 1, sizeof(cl_mem), &resized);
    opencl::argument(kernel, 2, sizeof(int), &source_width);
    opencl::argument(kernel, 3, sizeof(int), &source_height);
    opencl::argument(kernel, 4, sizeof(int), &width);
    opencl::argument(kernel, 5, sizeof(int), &height);
    const size_t pixels = resized_size / 3;
    opencl::check(clEnqueueNDRangeKernel(access.queue(), kernel, 1, nullptr, &pixels, nullptr, 0, nullptr, nullptr),
                  "clEnqueueNDRangeKernel(RGB resize)");
    return resized;
}

xt::xtensor<uint8_t, 3> resize_rgb(GpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height, int width) {
    auto access = runtime.acquire();
    const cl_mem resized = enqueue_resize_rgb(access, image, height, width);
    const size_t bytes = rgb_image_bytes(static_cast<size_t>(height), static_cast<size_t>(width), sizeof(uint8_t));
    const std::array<size_t, 3> shape{static_cast<size_t>(height), static_cast<size_t>(width), 3};
    xt::xtensor<uint8_t, 3> result = xt::empty<uint8_t>(shape);
    opencl::check(clEnqueueReadBuffer(access.queue(), resized, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr),
                  "clEnqueueReadBuffer(RGB resize)");
    return result;
}

}  // namespace geniex::vision::gpu
