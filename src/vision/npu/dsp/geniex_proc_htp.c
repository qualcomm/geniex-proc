// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "AEEStdDef.h"
#include "AEEStdErr.h"
#include "normalize_kernel.h"
#include "remote.h"
#include "resize_kernel.h"

static const remote_handle64 kHtpHandle = 1;

static AEEResult checked_rgb_bytes(int height, int width, int* bytes) {
    uint64_t pixel_count;
    uint64_t byte_count;

    if (!bytes || height <= 0 || width <= 0) return AEE_EBADPARM;
    pixel_count = (uint64_t)(unsigned int)height * (uint64_t)(unsigned int)width;
    if (pixel_count > UINT64_MAX / 3U) return AEE_EBADPARM;
    byte_count = pixel_count * 3U;
    if (byte_count > (uint64_t)INT_MAX) return AEE_EBADPARM;
    *bytes = (int)byte_count;
    return AEE_SUCCESS;
}

AEEResult geniex_proc_htp_open(const char* uri, remote_handle64* handle) {
    (void)uri;
    if (!handle) return AEE_EBADPARM;
    *handle = kHtpHandle;
    return AEE_SUCCESS;
}

AEEResult geniex_proc_htp_close(remote_handle64 handle) {
    if (handle != kHtpHandle) return AEE_ENOSUCHHANDLE;
    return AEE_SUCCESS;
}

AEEResult geniex_proc_htp_resize_rgb(remote_handle64 handle, int sourceHeight, int sourceWidth,
                                     int destinationHeight, int destinationWidth, const uint8* input, int inputLen,
                                     uint8* output, int outputLen) {
    int sourceBytes;
    int destinationBytes;
    if (handle != kHtpHandle) return AEE_ENOSUCHHANDLE;
    if (checked_rgb_bytes(sourceHeight, sourceWidth, &sourceBytes) != AEE_SUCCESS ||
        checked_rgb_bytes(destinationHeight, destinationWidth, &destinationBytes) != AEE_SUCCESS || !input || !output ||
        inputLen != sourceBytes || outputLen < destinationBytes) {
        return AEE_EBADPARM;
    }
    const int status = geniex_proc_htp_resize_kernel(input, sourceWidth, sourceHeight, output, destinationWidth,
                                                      destinationHeight);
    if (status == -2) return AEE_ENOMEMORY;
    if (status == -3) return AEE_EBADSTATE;
    return status == 0 ? AEE_SUCCESS : AEE_EBADPARM;
}

AEEResult geniex_proc_htp_qwen2vl_resize_patches(remote_handle64 handle, int sourceHeight, int sourceWidth,
                                                 int destinationHeight, int destinationWidth, int patchSize,
                                                 int mergeSize, int temporalPatchSize, const float* parameters,
                                                 int parametersLen, const uint8* input, int inputLen, float* output,
                                                 int outputLen) {
    int sourceBytes;
    int destinationBytes;
    uint64_t pixels;
    uint64_t outputFloats;

    if (handle != kHtpHandle) return AEE_ENOSUCHHANDLE;
    if (checked_rgb_bytes(sourceHeight, sourceWidth, &sourceBytes) != AEE_SUCCESS ||
        checked_rgb_bytes(destinationHeight, destinationWidth, &destinationBytes) != AEE_SUCCESS ||
        inputLen != sourceBytes || parametersLen != 7 || !parameters || !input || !output || outputLen < 0 ||
        patchSize <= 0 || mergeSize <= 0 || temporalPatchSize <= 0 || destinationHeight % patchSize != 0 ||
        destinationWidth % patchSize != 0 || (destinationHeight / patchSize) % mergeSize != 0 ||
        (destinationWidth / patchSize) % mergeSize != 0) {
        return AEE_EBADPARM;
    }

    pixels = (uint64_t)(unsigned int)destinationHeight * (uint64_t)(unsigned int)destinationWidth;
    if (pixels > UINT64_MAX / 3u || pixels * 3u > UINT64_MAX / (uint64_t)(unsigned int)temporalPatchSize) {
        return AEE_EBADPARM;
    }
    outputFloats = pixels * 3u * (uint64_t)(unsigned int)temporalPatchSize;
    if (outputFloats > (uint64_t)INT_MAX / sizeof(float)) return AEE_EBADPARM;
    if ((uint64_t)(unsigned int)outputLen < outputFloats) return AEE_EBUFFERTOOSMALL;

    uint8* resized = (uint8*)malloc((size_t)destinationBytes);
    if (!resized) return AEE_ENOMEMORY;
    int status =
        geniex_proc_htp_resize_kernel(input, sourceWidth, sourceHeight, resized, destinationWidth, destinationHeight);
    if (status == 0) {
        status = geniex_proc_htp_qwen2vl_patches_kernel(resized, output, destinationHeight, destinationWidth, patchSize,
                                                        mergeSize, temporalPatchSize, parameters);
    }
    free(resized);
    if (status == -2) return AEE_ENOMEMORY;
    if (status == -3) return AEE_EBADSTATE;
    return status == 0 ? AEE_SUCCESS : AEE_EBADPARM;
}
