// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <memory>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace geniex::vision {

class Qwen2VLGpu {
   public:
    Qwen2VLGpu();
    ~Qwen2VLGpu();
    Qwen2VLGpu(const Qwen2VLGpu&) = delete;
    Qwen2VLGpu& operator=(const Qwen2VLGpu&) = delete;

    // Resize, normalize, and extract the Qwen2-VL patch layout on the GPU.
    // Only the final flat patch tensor required by BatchFeatures is copied back.
    xt::xarray<float> preprocess_patches(const xt::xtensor<uint8_t, 3>& image, int height, int width, float scale,
                                         const float* mean, const float* stddev, int patch_size, int merge_size,
                                         int temporal_patch_size);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The patch path fuses resize, normalization, and extraction in one HTP RPC.
xt::xarray<float> preprocess_qwen2vl_htp_patches(const xt::xtensor<uint8_t, 3>& image, int height, int width,
                                                 float scale, const float* mean, const float* stddev, int patch_size,
                                                 int merge_size, int temporal_patch_size);

}  // namespace geniex::vision
