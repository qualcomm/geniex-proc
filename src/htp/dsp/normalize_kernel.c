// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_kernel.h"

#include <limits.h>
#include <math.h>

/* Normalize directly into the CPU Qwen2VL patch order.  No CHW tensor or
   second full-size DSP intermediate is needed; temporal copies are emitted
   as part of each patch's feature row. */
int geniex_proc_htp_qwen2vl_patches_kernel(const uint8_t* input, float* output, int height, int width, int patch_size,
                                           int merge_size, int temporal_patch_size, const float parameters[7]) {
    size_t pixels;
    size_t feature_count;
    int grid_h;
    int grid_w;
    int blocks_h;
    int blocks_w;
    float scale;
    float mean[3];
    float stddev[3];

    if (!input || !output || !parameters || height <= 0 || width <= 0 || patch_size <= 0 || merge_size <= 0 ||
        temporal_patch_size <= 0 || height % patch_size != 0 || width % patch_size != 0 ||
        (height / patch_size) % merge_size != 0 || (width / patch_size) % merge_size != 0) {
        return -1;
    }
    if ((size_t)height > SIZE_MAX / (size_t)width) return -1;
    pixels = (size_t)height * (size_t)width;
    if (pixels > (size_t)INT_MAX / 3u) return -1;
    feature_count = (size_t)3;
    if (feature_count > SIZE_MAX / (size_t)temporal_patch_size) return -1;
    feature_count *= (size_t)temporal_patch_size;
    if (feature_count > SIZE_MAX / (size_t)patch_size) return -1;
    feature_count *= (size_t)patch_size;
    if (feature_count > SIZE_MAX / (size_t)patch_size) return -1;
    feature_count *= (size_t)patch_size;
    if (pixels > SIZE_MAX / 3u || pixels * 3u > SIZE_MAX / (size_t)temporal_patch_size ||
        pixels * 3u * (size_t)temporal_patch_size > (size_t)INT_MAX / sizeof(float)) {
        return -1;
    }

    scale = parameters[0];
    if (!isfinite(scale)) return -1;
    for (size_t c = 0; c < 3; ++c) {
        if (!isfinite(parameters[1 + c]) || !isfinite(parameters[4 + c]) || parameters[4 + c] == 0.0f ||
            !isfinite(1.0f / parameters[4 + c])) {
            return -1;
        }
        mean[c] = parameters[1 + c];
        stddev[c] = parameters[4 + c];
    }

    grid_h = height / patch_size;
    grid_w = width / patch_size;
    blocks_h = grid_h / merge_size;
    blocks_w = grid_w / merge_size;

    for (int block_h = 0; block_h < blocks_h; ++block_h) {
        for (int block_w = 0; block_w < blocks_w; ++block_w) {
            for (int inner_h = 0; inner_h < merge_size; ++inner_h) {
                const int patch_y = block_h * merge_size + inner_h;
                for (int inner_w = 0; inner_w < merge_size; ++inner_w) {
                    const int patch_x = block_w * merge_size + inner_w;
                    const size_t patch_index =
                        (((size_t)block_h * (size_t)blocks_w + (size_t)block_w) * (size_t)merge_size +
                         (size_t)inner_h) *
                            (size_t)merge_size +
                        (size_t)inner_w;
                    float* patch_output = output + patch_index * feature_count;
                    for (size_t c = 0; c < 3; ++c) {
                        for (int temporal = 0; temporal < temporal_patch_size; ++temporal) {
                            for (int py = 0; py < patch_size; ++py) {
                                const size_t image_y = (size_t)patch_y * (size_t)patch_size + (size_t)py;
                                for (int px = 0; px < patch_size; ++px) {
                                    const size_t image_x = (size_t)patch_x * (size_t)patch_size + (size_t)px;
                                    const size_t pixel = image_y * (size_t)width + image_x;
                                    const size_t feature =
                                        (((c * (size_t)temporal_patch_size + (size_t)temporal) * (size_t)patch_size +
                                          (size_t)py) *
                                             (size_t)patch_size +
                                         (size_t)px);
                                    const float value = (float)input[pixel * 3u + c];
                                    /* Preserve the CPU's multiply, subtract, divide sequence. */
                                    volatile float scaled = value * scale;
                                    volatile float centered = scaled - mean[c];
                                    patch_output[feature] = centered / stddev[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
