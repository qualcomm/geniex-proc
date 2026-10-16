// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "AEEStdDef.h"
#include "AEEStdErr.h"
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

AEEResult geniex_proc_htp_echo(remote_handle64 handle, const uint8* input, int inputLen, uint8* output, int outputLen) {
    if (handle != kHtpHandle) return AEE_ENOSUCHHANDLE;
    if (inputLen < 0 || outputLen < 0) return AEE_EBADPARM;
    if ((inputLen > 0 && !input) || (outputLen > 0 && !output)) return AEE_EBADPARM;
    if (outputLen < inputLen) return AEE_EBUFFERTOOSMALL;
    if (inputLen > 0) memcpy(output, input, (size_t)inputLen);
    return AEE_SUCCESS;
}

AEEResult geniex_proc_htp_resize(remote_handle64 handle, int sourceHeight, int sourceWidth, int destinationHeight,
                                 int destinationWidth, const uint8* input, int inputLen, uint8* output, int outputLen) {
    int sourceBytes;
    int destinationBytes;

    if (handle != kHtpHandle) return AEE_ENOSUCHHANDLE;
    if (checked_rgb_bytes(sourceHeight, sourceWidth, &sourceBytes) != AEE_SUCCESS ||
        checked_rgb_bytes(destinationHeight, destinationWidth, &destinationBytes) != AEE_SUCCESS) {
        return AEE_EBADPARM;
    }
    if (inputLen < 0 || outputLen < 0) return AEE_EBADPARM;
    if (inputLen != sourceBytes) return AEE_EBADPARM;
    if (outputLen < destinationBytes) return AEE_EBUFFERTOOSMALL;
    if (!input || !output) return AEE_EBADPARM;

    const int result =
        geniex_proc_htp_resize_kernel(input, sourceWidth, sourceHeight, output, destinationWidth, destinationHeight);
    if (result == -2) return AEE_ENOMEMORY;
    if (result == -3) return AEE_EBADSTATE;
    return result == 0 ? AEE_SUCCESS : AEE_EBADPARM;
}
