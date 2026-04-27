// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef GENIEXPROC_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef GENIEXPROC_BUILD
#define GENIEXPROC_API __declspec(dllexport)
#else
#define GENIEXPROC_API __declspec(dllimport)
#endif
#else
#define GENIEXPROC_API __attribute__((visibility("default")))
#endif
#else
#define GENIEXPROC_API
#endif
