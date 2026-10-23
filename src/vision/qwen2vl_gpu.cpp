// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "qwen2vl_device.h"

#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_utils.h"
#include "vision.h"

#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/generators/xbuilder.hpp>

namespace geniex::vision {
namespace {

constexpr char kKernels[] = R"CLC(
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
__kernel void normalize_extract_qwen2vl(
    __global const uchar* input, __global float* output,
    const ulong output_elements, const ulong width,
    const ulong grid_height, const ulong grid_width,
    const ulong patch_size, const ulong merge_size, const ulong temporal_patch_size,
    const float scale,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r, const float std_g, const float std_b) {
    const ulong index = (ulong)get_global_id(0);
    if (index >= output_elements) return;

    // Flat output follows CPU transpose order [grid_h/ms, grid_w/ms, ms, ms,
    // C, T, P, P]. The input image is repeated for each temporal position.
    const ulong features_per_patch = 3 * temporal_patch_size * patch_size * patch_size;
    const ulong feature = index % features_per_patch;
    ulong patch = index / features_per_patch;
    const ulong pixel_x = feature % patch_size;
    const ulong pixel_y = (feature / patch_size) % patch_size;
    const ulong channel = feature / (temporal_patch_size * patch_size * patch_size);

    const ulong merge_x = patch % merge_size;
    patch /= merge_size;
    const ulong merge_y = patch % merge_size;
    patch /= merge_size;
    const ulong grid_x = patch % (grid_width / merge_size);
    const ulong grid_y = patch / (grid_width / merge_size);
    if (grid_y >= grid_height / merge_size) return;
    const ulong x = (grid_x * merge_size + merge_x) * patch_size + pixel_x;
    const ulong y = (grid_y * merge_size + merge_y) * patch_size + pixel_y;
    const ulong pixel = y * width + x;
    const uchar value = input[pixel * 3 + channel];
    const float mean = channel == 0 ? mean_r : (channel == 1 ? mean_g : mean_b);
    const float stddev = channel == 0 ? std_r : (channel == 1 ? std_g : std_b);
    output[index] = ((float)value * scale - mean) / stddev;
}
)CLC";

}  // namespace

struct Qwen2VLGpu::Impl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel resize = nullptr;
    cl_kernel normalize_extract = nullptr;
    cl_mem source = nullptr;
    cl_mem resized = nullptr;
    cl_mem output = nullptr;
    size_t source_capacity = 0;
    size_t resized_capacity = 0;
    size_t output_capacity = 0;
    std::mutex mutex;

    Impl() {
        try {
            cl_uint platform_count = 0;
            opencl::check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
            if (platform_count == 0) throw std::runtime_error("Qwen2VL: no OpenCL platforms");
            std::vector<cl_platform_id> platforms(platform_count);
            opencl::check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
            cl_device_id device = nullptr;
            for (auto platform : platforms) {
                cl_uint count = 0;
                const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
                if (status == CL_DEVICE_NOT_FOUND) continue;
                opencl::check(status, "clGetDeviceIDs");
                if (count) {
                    opencl::check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");
                    break;
                }
            }
            if (!device) throw std::runtime_error("Qwen2VL: no OpenCL GPU device (no CPU fallback)");
            cl_int status = CL_SUCCESS;
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
            opencl::check(status, "clCreateContext");
            queue = clCreateCommandQueue(context, device, 0, &status);
            opencl::check(status, "clCreateCommandQueue");
            const char* source_code = kKernels;
            program = clCreateProgramWithSource(context, 1, &source_code, nullptr, &status);
            opencl::check(status, "clCreateProgramWithSource");
            status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
            if (status != CL_SUCCESS) {
                size_t length = 0;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
                std::string log(length, '\0');
                if (length) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
                throw std::runtime_error("Qwen2VL OpenCL kernel build failed: " + log);
            }
            resize = clCreateKernel(program, "resize_catmullrom_antialias", &status);
            opencl::check(status, "clCreateKernel(resize)");
            normalize_extract = clCreateKernel(program, "normalize_extract_qwen2vl", &status);
            opencl::check(status, "clCreateKernel(normalize_extract)");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() {
        if (output) clReleaseMemObject(output);
        if (resized) clReleaseMemObject(resized);
        if (source) clReleaseMemObject(source);
        if (normalize_extract) clReleaseKernel(normalize_extract);
        if (resize) clReleaseKernel(resize);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void ensure(cl_mem& buffer, size_t& capacity, size_t length, cl_mem_flags flags) {
        if (capacity >= length && buffer) return;
        cl_int status = CL_SUCCESS;
        auto replacement = clCreateBuffer(context, flags, length, nullptr, &status);
        if (status != CL_SUCCESS) {
            if (replacement) clReleaseMemObject(replacement);
            opencl::check(status, "clCreateBuffer");
        }
        if (buffer) clReleaseMemObject(buffer);
        buffer = replacement;
        capacity = length;
    }
};

Qwen2VLGpu::Qwen2VLGpu() : impl_(std::make_unique<Impl>()) {}
Qwen2VLGpu::~Qwen2VLGpu() = default;

xt::xarray<float> Qwen2VLGpu::preprocess_patches(const xt::xtensor<uint8_t, 3>& image, int height, int width,
                                                 float scale, const float* mean, const float* stddev, int patch_size,
                                                 int merge_size, int temporal_patch_size) {
    if (image.shape(2) != 3 || height <= 0 || width <= 0 || !std::isfinite(scale) || !mean || !stddev ||
        patch_size <= 0 || merge_size <= 0 || temporal_patch_size <= 0 ||
        patch_size > std::numeric_limits<int>::max() / merge_size) {
        throw std::invalid_argument("Qwen2VL GPU invalid image or preprocessing parameters");
    }
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(mean[channel]) || !std::isfinite(stddev[channel]) || stddev[channel] == 0) {
            throw std::invalid_argument("Qwen2VL GPU invalid mean/std");
        }
    }
    const size_t source_size = rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t));
    const size_t resized_size = rgb_image_bytes(static_cast<size_t>(height), static_cast<size_t>(width), sizeof(uint8_t));
    if (image.size() != source_size) throw std::invalid_argument("Qwen2VL GPU expected HWC RGB image");

    const int resize_factor = patch_size * merge_size;
    if (height % resize_factor != 0 || width % resize_factor != 0) {
        throw std::invalid_argument("Qwen2VL GPU resized dimensions must be divisible by patch_size * merge_size");
    }
    const size_t grid_height = static_cast<size_t>(height / patch_size);
    const size_t grid_width = static_cast<size_t>(width / patch_size);
    const size_t merge = static_cast<size_t>(merge_size);
    if (grid_height % merge != 0 || grid_width % merge != 0) {
        throw std::invalid_argument("Qwen2VL GPU patch grid must be divisible by merge_size");
    }
    const size_t temporal = static_cast<size_t>(temporal_patch_size);
    const size_t patch = static_cast<size_t>(patch_size);
    const size_t patch_count = checked_product(grid_height, grid_width, "Qwen2VL GPU patch count overflow");
    size_t features_per_patch = checked_product(3, temporal, "Qwen2VL GPU patch feature size overflow");
    features_per_patch = checked_product(features_per_patch, patch, "Qwen2VL GPU patch feature size overflow");
    features_per_patch = checked_product(features_per_patch, patch, "Qwen2VL GPU patch feature size overflow");
    const size_t output_elements =
        checked_product(patch_count, features_per_patch, "Qwen2VL GPU output element count overflow");
    const size_t output_size = checked_product(output_elements, sizeof(float), "Qwen2VL GPU output byte size overflow");
    const size_t pixels = resized_size / 3;
    const cl_ulong output_elements_arg = opencl::as_cl_ulong(output_elements, "Qwen2VL GPU output exceeds OpenCL index range");
    const cl_ulong width_arg = opencl::as_cl_ulong(static_cast<size_t>(width), "Qwen2VL GPU width exceeds OpenCL index range");
    const cl_ulong grid_height_arg = opencl::as_cl_ulong(grid_height, "Qwen2VL GPU grid height exceeds OpenCL index range");
    const cl_ulong grid_width_arg = opencl::as_cl_ulong(grid_width, "Qwen2VL GPU grid width exceeds OpenCL index range");
    const cl_ulong patch_arg = opencl::as_cl_ulong(patch, "Qwen2VL GPU patch size exceeds OpenCL index range");
    const cl_ulong merge_arg = opencl::as_cl_ulong(merge, "Qwen2VL GPU merge size exceeds OpenCL index range");
    const cl_ulong temporal_arg = opencl::as_cl_ulong(temporal, "Qwen2VL GPU temporal size exceeds OpenCL index range");

    auto& state = *impl_;
    std::lock_guard lock(state.mutex);
    state.ensure(state.source, state.source_capacity, source_size, CL_MEM_READ_ONLY);
    state.ensure(state.resized, state.resized_capacity, resized_size, CL_MEM_READ_WRITE);
    state.ensure(state.output, state.output_capacity, output_size, CL_MEM_WRITE_ONLY);
    opencl::check(clEnqueueWriteBuffer(state.queue, state.source, CL_TRUE, 0, source_size, image.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    const int src_width = static_cast<int>(image.shape(1));
    const int src_height = static_cast<int>(image.shape(0));
    opencl::argument(state.resize, 0, sizeof(cl_mem), &state.source);
    opencl::argument(state.resize, 1, sizeof(cl_mem), &state.resized);
    opencl::argument(state.resize, 2, sizeof(int), &src_width);
    opencl::argument(state.resize, 3, sizeof(int), &src_height);
    opencl::argument(state.resize, 4, sizeof(int), &width);
    opencl::argument(state.resize, 5, sizeof(int), &height);
    opencl::check(clEnqueueNDRangeKernel(state.queue, state.resize, 1, nullptr, &pixels, nullptr, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(resize)");

    opencl::argument(state.normalize_extract, 0, sizeof(cl_mem), &state.resized);
    opencl::argument(state.normalize_extract, 1, sizeof(cl_mem), &state.output);
    opencl::argument(state.normalize_extract, 2, sizeof(output_elements_arg), &output_elements_arg);
    opencl::argument(state.normalize_extract, 3, sizeof(width_arg), &width_arg);
    opencl::argument(state.normalize_extract, 4, sizeof(grid_height_arg), &grid_height_arg);
    opencl::argument(state.normalize_extract, 5, sizeof(grid_width_arg), &grid_width_arg);
    opencl::argument(state.normalize_extract, 6, sizeof(patch_arg), &patch_arg);
    opencl::argument(state.normalize_extract, 7, sizeof(merge_arg), &merge_arg);
    opencl::argument(state.normalize_extract, 8, sizeof(temporal_arg), &temporal_arg);
    opencl::argument(state.normalize_extract, 9, sizeof(scale), &scale);
    for (cl_uint channel = 0; channel < 3; ++channel) {
        opencl::argument(state.normalize_extract, 10 + channel, sizeof(float), mean + channel);
        opencl::argument(state.normalize_extract, 13 + channel, sizeof(float), stddev + channel);
    }
    opencl::check(clEnqueueNDRangeKernel(state.queue, state.normalize_extract, 1, nullptr, &output_elements, nullptr, 0,
                                 nullptr, nullptr),
          "clEnqueueNDRangeKernel(normalize_extract)");

    xt::xarray<float> result = xt::empty<float>(std::array<size_t, 2>{patch_count, features_per_patch});
    opencl::check(clEnqueueReadBuffer(state.queue, state.output, CL_TRUE, 0, output_size, result.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");
    return result;
}

}  // namespace geniex::vision
