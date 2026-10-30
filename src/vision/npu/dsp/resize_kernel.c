// Copyright (c) 2026 Qualcomm Technologies, Inc.
// SPDX-License-Identifier: BSD-3-Clause

#include "resize_kernel.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __hexagon__
#include <hexagon_types.h>

#include "hvx_hexagon_protos.h"
#include "qurt_hvx.h"
#endif

typedef struct {
    size_t offset;
    size_t count;
} AxisEntry;

typedef struct {
    AxisEntry* entries;
    int* indices;
    float* weights;
    size_t count;
} Axis;

static int checked_mul_size(size_t a, size_t b, size_t* result) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

static int checked_add_size(size_t a, size_t b, size_t* result) {
    if (b > SIZE_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int checked_rgb_bytes(int width, int height, size_t* bytes) {
    size_t pixels;
    if (!bytes || width <= 0 || height <= 0) return 0;
    if (!checked_mul_size((size_t)width, (size_t)height, &pixels) || !checked_mul_size(pixels, 3u, bytes)) {
        return 0;
    }
    /* The RPC interface carries image sizes in signed ints.  Applying the same
       bound here also prevents an unchecked direct-kernel caller from asking
       the DSP allocator for an unrepresentable RPC-sized image. */
    return *bytes <= (size_t)INT_MAX;
}

static float cubic_weight(float x) {
    x = fabsf(x);
    if (x < 1.0f) return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    if (x < 2.0f) return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    return 0.0f;
}

static void axis_destroy(Axis* axis) {
    if (!axis) return;
    free(axis->entries);
    free(axis->indices);
    free(axis->weights);
    memset(axis, 0, sizeof(*axis));
}

/* First pass for one output coordinate.  Keeping the raw bounds signed and
   clamping only after enumerating them is important for widened downsampling:
   several raw samples can map to the same edge pixel. */
static int axis_entry_bounds(int output, int input, int coordinate, int64_t* first, int64_t* last) {
    const float scale = (float)input / (float)output;
    const float filter_scale = scale > 1.0f ? scale : 1.0f;
    const float center = ((float)coordinate + 0.5f) * scale - 0.5f;
    const float support = 2.0f * filter_scale;
    const double lower = ceil((double)center - (double)support);
    const double upper = floor((double)center + (double)support);

    if (!isfinite(lower) || !isfinite(upper) || lower < (double)INT64_MIN || upper > (double)INT64_MAX ||
        lower > upper) {
        return 0;
    }
    *first = (int64_t)lower;
    *last = (int64_t)upper;
    return 1;
}

static int checked_axis_span(int64_t first, int64_t last, uint64_t* span) {
    uint64_t distance;
    if (!span || first > last) return 0;
    if (first < 0 && last >= 0) {
        const uint64_t negative_magnitude = (uint64_t)(-(first + 1)) + 1u;
        if (negative_magnitude > UINT64_MAX - (uint64_t)last - 1u) return 0;
        *span = negative_magnitude + (uint64_t)last + 1u;
        return 1;
    }
    if (first >= 0) {
        distance = (uint64_t)last - (uint64_t)first;
    } else {
        distance = (uint64_t)(-(first + 1)) - (uint64_t)(-(last + 1));
    }
    if (distance == UINT64_MAX) return 0;
    *span = distance + 1u;
    return 1;
}

static int axis_build(Axis* axis, int output, int input) {
    size_t entries_bytes;
    size_t total = 0;
    int coordinate;

    memset(axis, 0, sizeof(*axis));
    if (!checked_mul_size((size_t)output, sizeof(*axis->entries), &entries_bytes)) return -2;
    axis->entries = (AxisEntry*)calloc(1, entries_bytes);
    if (!axis->entries) return -2;

    for (coordinate = 0; coordinate < output; ++coordinate) {
        int64_t first;
        int64_t last;
        uint64_t span;
        size_t count;
        size_t next_total;

        if (!axis_entry_bounds(output, input, coordinate, &first, &last) || !checked_axis_span(first, last, &span)) {
            axis_destroy(axis);
            return -1;
        }
        if (span > (uint64_t)SIZE_MAX) {
            axis_destroy(axis);
            return -2;
        }
        count = (size_t)span;
        if (!checked_add_size(total, count, &next_total)) {
            axis_destroy(axis);
            return -2;
        }
        axis->entries[coordinate].offset = total;
        axis->entries[coordinate].count = count;
        total = next_total;
    }

    if (!checked_mul_size(total, sizeof(*axis->indices), &entries_bytes)) {
        axis_destroy(axis);
        return -2;
    }
    axis->indices = (int*)calloc(1, entries_bytes);
    if (!checked_mul_size(total, sizeof(*axis->weights), &entries_bytes)) {
        axis_destroy(axis);
        return -2;
    }
    axis->weights = (float*)calloc(1, entries_bytes);
    if ((total != 0 && (!axis->indices || !axis->weights))) {
        axis_destroy(axis);
        return -2;
    }
    axis->count = total;

    for (coordinate = 0; coordinate < output; ++coordinate) {
        const float scale = (float)input / (float)output;
        const float filter_scale = scale > 1.0f ? scale : 1.0f;
        const float center = ((float)coordinate + 0.5f) * scale - 0.5f;
        const AxisEntry entry = axis->entries[coordinate];
        int64_t first;
        int64_t last;
        uint64_t span;
        float sum = 0.0f;
        size_t k;

        if (!axis_entry_bounds(output, input, coordinate, &first, &last) || !checked_axis_span(first, last, &span) ||
            entry.count != (size_t)span || entry.count > (size_t)INT64_MAX ||
            (entry.count != 0 && first > INT64_MAX - (int64_t)(entry.count - 1u))) {
            axis_destroy(axis);
            return -1;
        }
        for (k = 0; k < entry.count; ++k) {
            const int64_t raw = first + (int64_t)k;
            int clamped = raw < 0 ? 0 : raw >= (int64_t)input ? input - 1 : (int)raw;
            const float weight = cubic_weight((float)((double)raw - (double)center) / filter_scale);
            axis->indices[entry.offset + k] = clamped;
            axis->weights[entry.offset + k] = weight;
            sum += weight;
        }
        const float total = sum;
        if (!isfinite(total) || total == 0.0f) {
            axis_destroy(axis);
            return -1;
        }
        for (k = 0; k < entry.count; ++k) axis->weights[entry.offset + k] /= total;
    }
    return 0;
}

#ifdef __hexagon__
static float* allocate_float_rows(size_t bytes) { return (float*)memalign(128u, bytes); }
#else
static float* allocate_float_rows(size_t bytes) { return (float*)calloc(1, bytes); }
#endif

static uint8_t saturate_u8(float value) {
    value = floorf(value + 0.5f);
    if (value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return (uint8_t)value;
}

int geniex_proc_htp_resize_kernel(const uint8_t* input, int sw, int sh, uint8_t* output, int dw, int dh) {
    Axis horizontal;
    Axis vertical;
    float* horizontal_rows = NULL;
    uint8_t* staging = NULL;
    size_t source_bytes;
    size_t destination_bytes;
    size_t feature_count;
    size_t row_stride;
    size_t row_elements;
    size_t horizontal_bytes;
    size_t staging_bytes;
    size_t index;
    int result = -1;

    memset(&horizontal, 0, sizeof(horizontal));
    memset(&vertical, 0, sizeof(vertical));
    if (!input || !output || !checked_rgb_bytes(sw, sh, &source_bytes) ||
        !checked_rgb_bytes(dw, dh, &destination_bytes)) {
        return -1;
    }
    (void)source_bytes;

    if (!checked_mul_size((size_t)dw, 3u, &feature_count) || !checked_add_size(feature_count, 31u, &row_elements)) {
        return -1;
    }
    row_stride = row_elements & ~(size_t)31u;
    if (!checked_mul_size(row_stride, (size_t)sh, &row_elements) ||
        !checked_mul_size(row_elements, sizeof(float), &horizontal_bytes) ||
        !checked_mul_size((size_t)dw, (size_t)dh, &staging_bytes) ||
        !checked_mul_size(staging_bytes, 3u, &staging_bytes)) {
        return -1;
    }

    result = axis_build(&horizontal, dw, sw);
    if (result != 0) goto cleanup;
    result = axis_build(&vertical, dh, sh);
    if (result != 0) goto cleanup;

    horizontal_rows = allocate_float_rows(horizontal_bytes);
    staging = (uint8_t*)calloc(1, staging_bytes);
    if (!horizontal_rows || !staging) {
        result = -2;
        goto cleanup;
    }
    /* Padding is never read as an image value, but deterministic zero padding
       makes every aligned HVX load within a row well-defined. */
    memset(horizontal_rows, 0, horizontal_bytes);

    for (int y = 0; y < sh; ++y) {
        const size_t source_row = (size_t)y * (size_t)sw;
        for (int x = 0; x < dw; ++x) {
            const AxisEntry entry = horizontal.entries[x];
            const int* indices = horizontal.indices + entry.offset;
            const float* weights = horizontal.weights + entry.offset;
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (index = 0; index < entry.count; ++index) {
                    const size_t source_pixel = source_row + (size_t)indices[index];
                    value += (float)input[source_pixel * 3u + (size_t)channel] * weights[index];
                }
                horizontal_rows[(size_t)y * row_stride + (size_t)x * 3u + (size_t)channel] = value;
            }
        }
    }

#ifdef __hexagon__
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
        result = -3;
        goto cleanup;
    }
    for (int y = 0; y < dh; ++y) {
        const AxisEntry entry = vertical.entries[y];
        const int* indices = vertical.indices + entry.offset;
        const float* weights = vertical.weights + entry.offset;
        size_t block;
        for (block = 0; block < feature_count; block += 32u) {
            const size_t valid = feature_count - block < 32u ? feature_count - block : 32u;
            HVX_Vector accumulator;
            size_t tap;
            float lanes[32] __attribute__((aligned(128)));
            uint32_t weight_bits;

            memcpy(&weight_bits, &weights[0], sizeof(weight_bits));
            accumulator =
                Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector*)(horizontal_rows + (size_t)indices[0] * row_stride + block),
                                     Q6_V_vsplat_R((int32_t)weight_bits));
            for (tap = 1; tap < entry.count; ++tap) {
                memcpy(&weight_bits, &weights[tap], sizeof(weight_bits));
                accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
                    accumulator, Q6_Vqf32_vmpy_VsfVsf(
                                     *(const HVX_UVector*)(horizontal_rows + (size_t)indices[tap] * row_stride + block),
                                     Q6_V_vsplat_R((int32_t)weight_bits)));
            }
            *(HVX_UVector*)lanes = Q6_Vsf_equals_Vqf32(accumulator);
            for (index = 0; index < valid; ++index) {
                staging[(size_t)y * feature_count + block + index] = saturate_u8(lanes[index]);
            }
        }
    }
    if (qurt_hvx_unlock() != 0) {
        result = -3;
        goto cleanup;
    }
#else
    for (int y = 0; y < dh; ++y) {
        const AxisEntry entry = vertical.entries[y];
        const int* indices = vertical.indices + entry.offset;
        const float* weights = vertical.weights + entry.offset;
        for (index = 0; index < feature_count; ++index) {
            float value = 0.0f;
            for (size_t tap = 0; tap < entry.count; ++tap) {
                value += horizontal_rows[(size_t)indices[tap] * row_stride + index] * weights[tap];
            }
            staging[(size_t)y * feature_count + index] = saturate_u8(value);
        }
    }
#endif

    free(horizontal_rows);
    horizontal_rows = NULL;
    axis_destroy(&horizontal);
    axis_destroy(&vertical);
    memcpy(output, staging, destination_bytes);
    free(staging);
    return 0;

cleanup:
#ifdef __hexagon__
    /* The only error path after lock is the checked unlock above; no other
       operation can fail while the context is held. */
#endif
    free(horizontal_rows);
    free(staging);
    axis_destroy(&horizontal);
    axis_destroy(&vertical);
    return result;
}
