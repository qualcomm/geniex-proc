// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "image_ops.h"

#include <array>
#include <limits>
#include <stdexcept>

#include <xtensor/generators/xbuilder.hpp>

#include "geniex_proc_htp.h"
#include "../vision.h"

namespace geniex::vision::npu {

xt::xtensor<uint8_t, 3> resize_rgb(NpuRuntime& runtime, const xt::xtensor<uint8_t, 3>& image, int height, int width) {
    if (image.shape(2) != 3 || image.size() != rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t))) {
        throw std::invalid_argument("geniex::vision::npu::resize_rgb: expected HWC RGB image");
    }
    const size_t source_bytes = rgb_image_bytes(image.shape(0), image.shape(1), sizeof(uint8_t));
    const size_t output_bytes =
        rgb_image_bytes(static_cast<size_t>(height), static_cast<size_t>(width), sizeof(uint8_t));
    if (source_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        output_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("geniex::vision::npu::resize_rgb: image exceeds the FastRPC length limit");
    }
    const std::array<size_t, 3> shape{static_cast<size_t>(height), static_cast<size_t>(width), 3};
    xt::xtensor<uint8_t, 3> result = xt::empty<uint8_t>(shape);
    auto access = runtime.acquire();
    check(geniex_proc_htp_resize_rgb(access.handle, static_cast<int32_t>(image.shape(0)),
                                     static_cast<int32_t>(image.shape(1)), height, width, image.data(),
                                     static_cast<int>(source_bytes), result.data(), static_cast<int>(output_bytes)),
          "RGB resize");
    return result;
}

}  // namespace geniex::vision::npu
