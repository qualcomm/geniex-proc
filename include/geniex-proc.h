// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// geniex-proc.h — Core umbrella header for the geniex-proc library.
//
// Include this single header to get the full public API

#include "geniex-proc/export.h"
#include "geniex-proc/types.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/sampler.h"

#ifdef GENIEXPROC_ENABLE_VISION
#include "geniex-vision.h"
#endif

#ifdef GENIEXPROC_ENABLE_AUDIO
#include "geniex-audio.h"
#endif
