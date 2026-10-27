// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENIEX_PROC_HTP_RESIZE_KERNEL_H
#define GENIEX_PROC_HTP_RESIZE_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, -1 for invalid input, -2 for allocation, or -3 for HVX context errors.
int geniex_proc_htp_resize_kernel(const uint8_t* input, int sw, int sh, uint8_t* output, int dw, int dh);

#ifdef __cplusplus
}
#endif

#endif
