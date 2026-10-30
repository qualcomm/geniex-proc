// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stddef.h>
#include <stdint.h>

int geniex_proc_htp_qwen2vl_patches_kernel(const uint8_t* input, float* output, int height, int width, int patch_size,
                                           int merge_size, int temporal_patch_size, const float parameters[7]);
