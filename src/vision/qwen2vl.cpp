// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vision.h"

#include <limits>
#include <stdexcept>
#include <vector>

#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/views/xview.hpp>

#include "geniex-proc/qwen2vl.h"
#include "cpu/image_ops.h"
#include "preprocess_device.h"
#ifdef GENIEXPROC_ENABLE_OPENCL
#include "gpu/qwen2vl_patches.h"
#endif
#ifdef GENIEXPROC_ENABLE_HTP
#include "npu/qwen2vl_patches.h"
#endif

namespace geniex::vision {

std::tuple<xt::xarray<float>, xt::xarray<size_t>> preprocess_qwen2vl(
    const PreprocessDevice& device, const xt::xtensor<uint8_t, 3>& image,
    const ::geniex::qwen2vl::Qwen2VLConfig& config) {
    if (config.image_mean.size() != 3 || config.image_std.size() != 3) {
        throw std::invalid_argument("Qwen2VL image mean/std must have three channels");
    }
    if (device.type() != ::geniex::PreprocessDevice::CPU) {
        const auto max_int = static_cast<size_t>(std::numeric_limits<int>::max());
        if (image.shape(0) == 0 || image.shape(1) == 0 || image.shape(0) > max_int || image.shape(1) > max_int ||
            config.patch_size <= 0 || config.merge_size <= 0 || config.temporal_patch_size <= 0 ||
            config.patch_size > std::numeric_limits<int>::max() / config.merge_size) {
            throw std::invalid_argument("Qwen2VL accelerator invalid image dimensions or patch configuration");
        }
        if (!(config.fixed_height > 0 && config.fixed_width > 0) &&
            (config.min_pixels <= 0 || config.max_pixels < config.min_pixels ||
             config.max_pixels > std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Qwen2VL accelerator invalid smart_resize pixel limits");
        }
    }

    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    int resized_height = height;
    int resized_width = width;
    if (config.fixed_height > 0 && config.fixed_width > 0) {
        resized_height = config.fixed_height;
        resized_width = config.fixed_width;
    } else {
        std::tie(resized_height, resized_width) = smart_resize(
            height, width, config.patch_size * config.merge_size, static_cast<int>(config.min_pixels),
            static_cast<int>(config.max_pixels));
    }

    xt::xarray<float> flat;
    size_t grid_t = 1;
    const size_t grid_h = static_cast<size_t>(resized_height) / config.patch_size;
    const size_t grid_w = static_cast<size_t>(resized_width) / config.patch_size;
    if (device.type() == ::geniex::PreprocessDevice::GPU) {
#ifdef GENIEXPROC_ENABLE_OPENCL
        flat = gpu::preprocess_qwen2vl_patches(device.gpu_runtime(), image, resized_height, resized_width,
                                               config.rescale_factor, config.image_mean.data(),
                                               config.image_std.data(), config.patch_size, config.merge_size,
                                               config.temporal_patch_size);
#else
        throw std::runtime_error("Qwen2VL GPU preprocessing requires GENIEXPROC_ENABLE_OPENCL");
#endif
    } else if (device.type() == ::geniex::PreprocessDevice::NPU) {
#ifdef GENIEXPROC_ENABLE_HTP
        flat = npu::preprocess_qwen2vl_patches(device.npu_runtime(), image, resized_height, resized_width,
                                              config.rescale_factor, config.image_mean.data(), config.image_std.data(),
                                              config.patch_size, config.merge_size, config.temporal_patch_size);
#else
        throw std::runtime_error("Qwen2VL NPU preprocessing requires GENIEXPROC_ENABLE_HTP");
#endif
    } else {
        auto processed = cpu::resize_image(image, resized_height, resized_width);
        xt::xtensor<float, 3> img_float = xt::cast<float>(processed);
        img_float = img_float * config.rescale_factor;
        for (int channel = 0; channel < 3; ++channel) {
            xt::view(img_float, xt::all(), xt::all(), channel) =
                (xt::view(img_float, xt::all(), xt::all(), channel) - config.image_mean[channel]) /
                config.image_std[channel];
        }
        xt::xtensor<float, 3> img_chw = xt::transpose(img_float, {2, 0, 1});
        xt::xarray<float> patches = xt::view(img_chw, xt::newaxis(), xt::all(), xt::all(), xt::all());
        patches = xt::eval(xt::repeat(patches, static_cast<size_t>(config.temporal_patch_size), 0));

        const size_t channels = patches.shape(1);
        grid_t = patches.shape(0) / config.temporal_patch_size;
        patches = patches.reshape({
            grid_t,
            static_cast<size_t>(config.temporal_patch_size),
            channels,
            grid_h / config.merge_size,
            static_cast<size_t>(config.merge_size),
            static_cast<size_t>(config.patch_size),
            grid_w / config.merge_size,
            static_cast<size_t>(config.merge_size),
            static_cast<size_t>(config.patch_size),
        });
        patches = xt::transpose(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
        flat = patches.reshape({
            grid_t * grid_h * grid_w,
            channels * config.temporal_patch_size * config.patch_size * config.patch_size,
        });
    }

    xt::xarray<size_t> grid_thw = xt::adapt(std::vector<size_t>{grid_t, grid_h, grid_w}, {1UL, 3UL});
    return {flat, grid_thw};
}

}  // namespace geniex::vision
