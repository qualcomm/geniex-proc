// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "image_ops.h"

#include <stdexcept>
#include <vector>

#include <xtensor/containers/xadapt.hpp>

#include "../vision.h"
#include "stb/stb_static.h"

namespace geniex::vision::cpu {

xt::xtensor<uint8_t, 3> resize_image(const xt::xtensor<uint8_t, 3>& image, int resized_height, int resized_width) {
    if (image.shape(2) != 3 || image.size() != rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t))) {
        throw std::invalid_argument("geniex::vision::cpu::resize_image: expected HWC RGB image");
    }
    const size_t resized_size = rgb_image_bytes(static_cast<size_t>(resized_height),
                                                static_cast<size_t>(resized_width), sizeof(uint8_t));
    std::vector<uint8_t> resized_data(resized_size);
    const int source_height = static_cast<int>(image.shape(0));
    const int source_width = static_cast<int>(image.shape(1));
    stbir_resize(image.data(), source_width, source_height, 0,
                 resized_data.data(), resized_width, resized_height, 0,
                 STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
    return xt::adapt(resized_data, std::vector<size_t>{static_cast<size_t>(resized_height),
                                                       static_cast<size_t>(resized_width), 3UL});
}

}  // namespace geniex::vision::cpu
