// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <AEEStdErr.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geniex_proc_htp.h"
#include "htp/htp_runtime.h"
#include "vision.h"

namespace geniex::vision {
namespace {

size_t checked_rgb_bytes(size_t height, size_t width) {
    if (height == 0 || width == 0 || height > static_cast<size_t>(INT_MAX) || width > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("geniex::vision::resize_image: dimensions must be positive 32-bit integers");
    }
    if (height > std::numeric_limits<size_t>::max() / width ||
        height * width > std::numeric_limits<size_t>::max() / 3) {
        throw std::invalid_argument("geniex::vision::resize_image: image is too large");
    }
    const size_t bytes = height * width * 3;
    if (bytes > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("geniex::vision::resize_image: image byte count exceeds FastRPC limit");
    }
    return bytes;
}

}  // namespace

xt::xtensor<uint8_t, 3> resize_image(const xt::xtensor<uint8_t, 3>& image, int resized_height, int resized_width) {
    if (image.shape(2) != 3) {
        throw std::invalid_argument("geniex::vision::resize_image: input tensor must be HWC RGB");
    }
    const size_t source_height = image.shape(0);
    const size_t source_width = image.shape(1);
    const size_t source_bytes = checked_rgb_bytes(source_height, source_width);
    if (image.size() != source_bytes || !image.data()) {
        throw std::invalid_argument("geniex::vision::resize_image: input must contain exactly RGB bytes");
    }
    if (resized_height <= 0 || resized_width <= 0) {
        throw std::invalid_argument("geniex::vision::resize_image: dimensions must be positive");
    }
    const size_t destination_bytes =
        checked_rgb_bytes(static_cast<size_t>(resized_height), static_cast<size_t>(resized_width));
    std::vector<uint8_t> resized_data(destination_bytes);
    auto access = geniex::htp::acquire(geniex::htp::default_session());
    const auto result =
        geniex_proc_htp_resize(access.handle, static_cast<int32_t>(source_height), static_cast<int32_t>(source_width),
                               resized_height, resized_width, image.data(), static_cast<int>(source_bytes),
                               resized_data.data(), static_cast<int>(destination_bytes));
    if (result != AEE_SUCCESS) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%x", static_cast<unsigned>(result));
        throw std::runtime_error(std::string("geniex::vision::resize_image: HTP resize failed with AEEResult ") + code);
    }
    return xt::adapt(std::move(resized_data),
                     std::vector<size_t>{static_cast<size_t>(resized_height), static_cast<size_t>(resized_width), 3UL});
}

}  // namespace geniex::vision
