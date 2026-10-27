// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/vision/vision.cpp — Internal shared vision utilities implementation.
//
// Compiled only when GENIEXPROC_ENABLE_VISION is ON.

#include "vision.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb/stb_static.h"

namespace geniex::vision {

::geniex::PreprocessDevice resolve_preprocess_device(::geniex::PreprocessDevice requested) {
    using Type = ::geniex::PreprocessDevice;
    if (requested != Type::Auto) return requested;
    const char* value = std::getenv("GENIEX_QAIRT_PREPROCESS_DEVICE");
    if (!value || !*value) return Type::CPU;
    std::string device = value;
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (device == "auto" || device == "cpu") return Type::CPU;
    if (device == "gpu") return Type::GPU;
    if (device == "npu") return Type::NPU;
    throw std::invalid_argument("Invalid GENIEX_QAIRT_PREPROCESS_DEVICE: " + device);
}

size_t checked_product(size_t left, size_t right, const char* error) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw std::invalid_argument(error);
    }
    return left * right;
}

size_t rgb_image_bytes(size_t height, size_t width, size_t element_size) {
    if (height == 0 || width == 0 || height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        width > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("geniex::vision::rgb_image_bytes: dimensions must be nonzero and fit in int range");
    }
    const size_t pixels = checked_product(height, width, "geniex::vision::rgb_image_bytes: pixel count overflow");
    const size_t rgb_bytes = checked_product(pixels, 3, "geniex::vision::rgb_image_bytes: RGB byte count overflow");
    return checked_product(rgb_bytes, element_size, "geniex::vision::rgb_image_bytes: image byte size overflow");
}

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
