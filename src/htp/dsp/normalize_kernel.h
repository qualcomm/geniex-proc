// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stddef.h>
#include <stdint.h>

int geniex_proc_htp_normalize_kernel(const uint8_t* input, float* output, size_t pixels, const float parameters[7]);
