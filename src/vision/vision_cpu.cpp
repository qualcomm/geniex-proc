// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/vision/vision_cpu.cpp — CPU image resize implementation.

#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vision.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb/stb_image_resize2.h"

namespace geniex::vision {
namespace {

size_t checked_rgb_bytes(size_t height, size_t width) {
    if (height == 0 || width == 0 || height > static_cast<size_t>(INT_MAX) || width > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("geniex::vision::resize_image: dimensions must be positive 32-bit integers");
    }
    constexpr size_t channels = 3;
    if (height > std::numeric_limits<size_t>::max() / width ||
        height * width > std::numeric_limits<size_t>::max() / channels) {
        throw std::invalid_argument("geniex::vision::resize_image: image is too large");
    }
    const size_t bytes = height * width * channels;
    if (bytes > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("geniex::vision::resize_image: image byte count exceeds FastRPC limit");
    }
    return bytes;
}

}  // namespace

xt::xtensor<uint8_t, 3> resize_image(const xt::xtensor<uint8_t, 3>& image, int resized_height, int resized_width) {
    const size_t source_height = image.shape(0);
    const size_t source_width = image.shape(1);
    const size_t source_bytes = checked_rgb_bytes(source_height, source_width);
    if (resized_height <= 0 || resized_width <= 0) {
        throw std::invalid_argument("geniex::vision::resize_image: dimensions must be positive");
    }
    const size_t destination_bytes =
        checked_rgb_bytes(static_cast<size_t>(resized_height), static_cast<size_t>(resized_width));
    if (image.size() != source_bytes) {
        throw std::invalid_argument("geniex::vision::resize_image: input tensor must contain exactly RGB bytes");
    }

    std::vector<uint8_t> resized_data(destination_bytes);
    stbir_resize(image.data(), static_cast<int>(source_width), static_cast<int>(source_height), 0, resized_data.data(),
                 resized_width, resized_height, 0, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP,
                 STBIR_FILTER_CATMULLROM);

    return xt::adapt(std::move(resized_data),
                     std::vector<size_t>{static_cast<size_t>(resized_height), static_cast<size_t>(resized_width), 3UL});
}

}  // namespace geniex::vision
