// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "qwen2vl_patches.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/generators/xbuilder.hpp>

#include "opencl_utils.h"
#include "image_ops.h"
#include "../vision.h"

namespace geniex::vision::gpu {
namespace {

constexpr char kPatchKernel[] = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
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

    // Qwen2-VL flat order: [grid_h/ms, grid_w/ms, ms, ms, C, T, P, P].
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

xt::xarray<float> preprocess_qwen2vl_patches(GpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height,
                                             int width, float scale, const float* mean, const float* stddev,
                                             int patch_size, int merge_size, int temporal_patch_size) {
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
    const size_t resized_size = rgb_image_bytes(static_cast<size_t>(height), static_cast<size_t>(width), sizeof(uint8_t));
    if (image.size() != rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t))) {
        throw std::invalid_argument("Qwen2VL GPU expected HWC RGB image");
    }
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

    auto access = runtime.acquire();
    const cl_mem resized = enqueue_resize_rgb(access, image, height, width);
    const cl_mem output = access.buffer("vision.qwen2vl.patch_output", output_size, CL_MEM_WRITE_ONLY);
    const cl_kernel extract_kernel = access.kernel("vision.qwen2vl.patch", kPatchKernel, "normalize_extract_qwen2vl",
                                                   "build Qwen2-VL patch program / create Qwen2-VL patch kernel");
    opencl::argument(extract_kernel, 0, sizeof(cl_mem), &resized);
    opencl::argument(extract_kernel, 1, sizeof(cl_mem), &output);
    opencl::argument(extract_kernel, 2, sizeof(output_elements_arg), &output_elements_arg);
    opencl::argument(extract_kernel, 3, sizeof(width_arg), &width_arg);
    opencl::argument(extract_kernel, 4, sizeof(grid_height_arg), &grid_height_arg);
    opencl::argument(extract_kernel, 5, sizeof(grid_width_arg), &grid_width_arg);
    opencl::argument(extract_kernel, 6, sizeof(patch_arg), &patch_arg);
    opencl::argument(extract_kernel, 7, sizeof(merge_arg), &merge_arg);
    opencl::argument(extract_kernel, 8, sizeof(temporal_arg), &temporal_arg);
    opencl::argument(extract_kernel, 9, sizeof(scale), &scale);
    for (cl_uint channel = 0; channel < 3; ++channel) {
        opencl::argument(extract_kernel, 10 + channel, sizeof(float), mean + channel);
        opencl::argument(extract_kernel, 13 + channel, sizeof(float), stddev + channel);
    }
    opencl::check(clEnqueueNDRangeKernel(access.queue(), extract_kernel, 1, nullptr, &output_elements, nullptr, 0,
                                          nullptr, nullptr),
                  "clEnqueueNDRangeKernel(Qwen2-VL patch extraction)");

    xt::xarray<float> result = xt::empty<float>(std::array<size_t, 2>{patch_count, features_per_patch});
    opencl::check(clEnqueueReadBuffer(access.queue(), output, CL_TRUE, 0, output_size, result.data(), 0, nullptr, nullptr),
                  "clEnqueueReadBuffer(Qwen2-VL patches)");
    return result;
}

}  // namespace geniex::vision::gpu
