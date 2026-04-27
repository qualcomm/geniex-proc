// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/vision/vision.cpp — Internal shared vision utilities implementation.
//
// Compiled only when GENIEXPROC_ENABLE_VISION is ON.

#include "vision.h"

#include <cstring>
#include <filesystem>
#include <vector>

#include "stb/stb_static.h"

namespace geniex::vision {

xt::xtensor<uint8_t, 3> load_image(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("geniex::vision::load_image: file does not exist: " + path);
    }

    int width, height, channels_native;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels_native, 3);
    if (!data) {
        throw std::runtime_error("geniex::vision::load_image: failed to load image: " + path);
    }

    std::vector<uint8_t> image_data(static_cast<size_t>(height) * width * 3);
    std::memcpy(image_data.data(), data, image_data.size());
    stbi_image_free(data);

    return xt::adapt(image_data, std::vector<size_t>{static_cast<size_t>(height), static_cast<size_t>(width), 3UL});
}

xt::xtensor<uint8_t, 3> resize_image(const xt::xtensor<uint8_t, 3>& image, int resized_height, int resized_width) {
    int ori_height = static_cast<int>(image.shape(0));
    int ori_width  = static_cast<int>(image.shape(1));

    size_t resized_size = static_cast<size_t>(resized_height) * resized_width * 3;
    std::vector<uint8_t> resized_data(resized_size);

    stbir_resize(image.data(), ori_width, ori_height, 0,
                 resized_data.data(), resized_width, resized_height, 0,
                 STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);

    return xt::adapt(resized_data, std::vector<size_t>{static_cast<size_t>(resized_height), static_cast<size_t>(resized_width), 3UL});
}

std::tuple<int, int> smart_resize(int height, int width, int factor, int min_pixels, int max_pixels) {
    constexpr int MAX_RATIO = 200;
    if (std::max(height, width) / std::min(height, width) > MAX_RATIO) {
        throw std::runtime_error(
            "geniex::vision::smart_resize: absolute aspect ratio must be smaller than " +
            std::to_string(MAX_RATIO));
    }

    int h_bar = std::max(factor, round_by_factor(height, factor));
    int w_bar = std::max(factor, round_by_factor(width, factor));

    int total_pixels = h_bar * w_bar;
    if (total_pixels > max_pixels) {
        float beta = std::sqrt(static_cast<float>(height * width) / static_cast<float>(max_pixels));
        h_bar = floor_by_factor(static_cast<int>(h_bar / beta), factor);
        w_bar = floor_by_factor(static_cast<int>(w_bar / beta), factor);
    } else if (total_pixels < min_pixels) {
        float beta = std::sqrt(static_cast<float>(min_pixels) / static_cast<float>(height * width));
        h_bar = ceil_by_factor(static_cast<int>(h_bar * beta), factor);
        w_bar = ceil_by_factor(static_cast<int>(w_bar * beta), factor);
    }

    return {h_bar, w_bar};
}

}  // namespace geniex::vision
