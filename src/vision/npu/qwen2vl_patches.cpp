// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "geniex_proc_htp.h"
#include "npu_runtime.h"
#include "qwen2vl_patches.h"
#include "../vision.h"

namespace geniex::vision::npu {

xt::xarray<float> preprocess_qwen2vl_patches(NpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height,
                                             int width, float scale, const float* mean, const float* stddev,
                                             int patch_size, int merge_size, int temporal_patch_size) {
    if (image.shape(2) != 3 || height <= 0 || width <= 0 || !std::isfinite(scale) || !mean || !stddev ||
        patch_size <= 0 || merge_size <= 0 || temporal_patch_size <= 0 || height % patch_size != 0 ||
        width % patch_size != 0 || (height / patch_size) % merge_size != 0 || (width / patch_size) % merge_size != 0) {
        throw std::invalid_argument("Qwen2VL HTP invalid image, patch dimensions, or parameters");
    }
    for (size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(mean[c]) || !std::isfinite(stddev[c]) || stddev[c] == 0 ||
            !std::isfinite(1.0f / stddev[c])) {
            throw std::invalid_argument("Qwen2VL HTP invalid mean/std");
        }
    }

    const size_t scratch_bytes = geniex::vision::rgb_image_bytes(image.shape(0), image.shape(1), sizeof(float));
    if (scratch_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Qwen2VL HTP image dimensions exceed the FastRPC limit");
    }
    const int input_bytes = static_cast<int>(geniex::vision::rgb_image_bytes(
        image.shape(0), image.shape(1), sizeof(uint8_t)));
    if (image.size() != static_cast<size_t>(input_bytes))
        throw std::invalid_argument("Qwen2VL HTP expected HWC RGB image");

    const size_t pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
    if (pixels > std::numeric_limits<size_t>::max() / 3 ||
        pixels * 3 > std::numeric_limits<size_t>::max() / static_cast<size_t>(temporal_patch_size)) {
        throw std::invalid_argument("Qwen2VL HTP patch output size overflows");
    }
    const size_t output_elements = pixels * 3 * static_cast<size_t>(temporal_patch_size);
    if (output_elements > static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float)) {
        throw std::invalid_argument("Qwen2VL HTP patch output exceeds the FastRPC length limit");
    }

    const size_t patch = static_cast<size_t>(patch_size);
    size_t feature_width = static_cast<size_t>(temporal_patch_size);
    if (feature_width > std::numeric_limits<size_t>::max() / 3) {
        throw std::invalid_argument("Qwen2VL HTP patch feature size overflows");
    }
    feature_width *= 3;
    if (feature_width > std::numeric_limits<size_t>::max() / patch ||
        feature_width * patch > std::numeric_limits<size_t>::max() / patch) {
        throw std::invalid_argument("Qwen2VL HTP patch feature size overflows");
    }
    feature_width *= patch * patch;
    const size_t grid_h = static_cast<size_t>(height / patch_size);
    const size_t grid_w = static_cast<size_t>(width / patch_size);
    const std::array<size_t, 2> shape{grid_h * grid_w, feature_width};

    auto access = runtime.acquire();
    const std::array<float, 7> parameters{scale, mean[0], mean[1], mean[2], stddev[0], stddev[1], stddev[2]};
    xt::xarray<float> result = xt::zeros<float>(shape);
    check(geniex_proc_htp_qwen2vl_resize_patches(
              access.handle, static_cast<int32_t>(image.shape(0)), static_cast<int32_t>(image.shape(1)), height, width,
              patch_size, merge_size, temporal_patch_size, parameters.data(), static_cast<int>(parameters.size()),
              image.data(), input_bytes, result.data(), static_cast<int>(result.size())),
          "resize+normalize+patch extraction");
    return result;
}

}  // namespace geniex::vision::npu
