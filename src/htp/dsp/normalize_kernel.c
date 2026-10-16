// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_kernel.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <limits.h>
#include <math.h>
#include <qurt_hvx.h>
#include <string.h>

static HVX_Vector splat_float(float value) {
    int32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return Q6_V_vsplat_R(bits);
}

int geniex_proc_htp_normalize_kernel(const uint8_t* input, float* output, size_t pixels, const float parameters[7]) {
    if (!input || !output || !parameters || pixels == 0 || pixels > (size_t)INT_MAX / 12 || !isfinite(parameters[0]))
        return -1;
    float inv_std[3];
    for (size_t c = 0; c < 3; ++c) {
        if (!isfinite(parameters[1 + c]) || !isfinite(parameters[4 + c]) || parameters[4 + c] == 0) return -1;
        inv_std[c] = 1.0f / parameters[4 + c];
        if (!isfinite(inv_std[c])) return -1;
    }
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return -3;
    int32_t indices[3][32] __attribute__((aligned(128)));
    for (size_t c = 0; c < 3; ++c)
        for (size_t lane = 0; lane < 32; ++lane) indices[c][lane] = (int32_t)(3 * lane + c);
    uint8_t rgb[128] __attribute__((aligned(128)));
    memset(rgb, 0, sizeof(rgb));
    const HVX_Vector scale = splat_float(parameters[0]);
    const HVX_Vector mask = Q6_V_vsplat_R(255);
    HVX_Vector means[3], reciprocals[3];
    for (size_t c = 0; c < 3; ++c) {
        means[c] = splat_float(parameters[1 + c]);
        reciprocals[c] = splat_float(inv_std[c]);
    }
    for (size_t i = 0; i < pixels; i += 32) {
        const size_t count = pixels - i < 32 ? pixels - i : 32;
        if (count == 32)
            memcpy(rgb, input + 3 * i, 96);
        else
            memcpy(rgb, input + 3 * i, count * 3);
        // Selectors 0/1 use even table bytes; selector 2 uses odd bytes.
        const HVX_VectorPair table = Q6_Wuh_vunpack_Vub(*(const HVX_Vector*)rgb);
        const HVX_Vector upper = Q6_Vh_vasl_VhR(Q6_V_hi_W(table), 8);
        for (size_t c = 0; c < 3; ++c) {
            const HVX_Vector index = *(const HVX_Vector*)indices[c];
            HVX_Vector bytes = Q6_Vb_vlut32_VbVbI(index, Q6_V_lo_W(table), 0);
            bytes = Q6_Vb_vlut32or_VbVbVbI(bytes, index, Q6_V_lo_W(table), 1);
            bytes = Q6_Vb_vlut32or_VbVbVbI(bytes, index, upper, 2);
            const HVX_Vector values = Q6_Vsf_equals_Vw(Q6_V_vand_VV(bytes, mask));
            const HVX_Vector scaled = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(values, scale));
            const HVX_Vector centered = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(scaled, means[c]));
            const HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(centered, reciprocals[c]));
            if (count == 32)
                *(HVX_UVector*)(output + c * pixels + i) = result;
            else {
                float tail[32] __attribute__((aligned(128)));
                *(HVX_Vector*)tail = result;
                memcpy(output + c * pixels + i, tail, count * sizeof(float));
            }
        }
    }
    qurt_hvx_unlock();
    return 0;
}
