// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// src/vision/vision.h — Internal shared vision utilities.
//
// NOT a public header. Used only by processor implementations inside src/processors/.
// Requires GENIEXPROC_ENABLE_VISION to be compiled.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "geniex-proc/types.h"

namespace geniex::qwen2vl {
struct Qwen2VLConfig;
}

namespace geniex::vision {

class PreprocessDevice;

::geniex::PreprocessDevice resolve_preprocess_device(::geniex::PreprocessDevice requested);
size_t checked_product(size_t left, size_t right, const char* error);
size_t rgb_image_bytes(size_t height, size_t width, size_t element_size);

// ============================================================
// Image loading
// ============================================================

/**
 * @brief Load an image from disk as a uint8 HWC tensor (H x W x 3).
 * @param path Path to the image file (JPEG, PNG, BMP, etc.).
 * @throws std::runtime_error if the file cannot be loaded.
 */
xt::xtensor<uint8_t, 3> load_image(const std::string& path);

// ============================================================
// Smart resize
// ============================================================

// Helper rounding functions
inline int round_by_factor(int number, int factor) {
    return static_cast<int>(std::round(static_cast<float>(number) / factor)) * factor;
}
inline int ceil_by_factor(int number, int factor) {
    return static_cast<int>(std::ceil(static_cast<float>(number) / factor)) * factor;
}
inline int floor_by_factor(int number, int factor) {
    return static_cast<int>(std::floor(static_cast<float>(number) / factor)) * factor;
}

/**
 * @brief Resize dimensions so that:
 *   1. Both are divisible by `factor`.
 *   2. Total pixels stay within [min_pixels, max_pixels].
 *   3. Aspect ratio is preserved as closely as possible.
 *
 * Mirrors the Python `smart_resize` from the Qwen-VL processing utilities.
 *
 * @throws std::runtime_error if the aspect ratio exceeds MAX_RATIO (200).
 */
std::tuple<int, int> smart_resize(int height, int width, int factor, int min_pixels, int max_pixels);

std::tuple<xt::xarray<float>, xt::xarray<size_t>> preprocess_qwen2vl(
    const PreprocessDevice& device, const xt::xtensor<uint8_t, 3>& image,
    const ::geniex::qwen2vl::Qwen2VLConfig& config);

}  // namespace geniex::vision
