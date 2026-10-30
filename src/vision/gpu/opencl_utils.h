// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/cl.h>

#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace geniex::vision::opencl {

inline void check(cl_int code, const char* operation) {
    if (code != CL_SUCCESS) {
        std::ostringstream message;
        message << "geniex::vision::opencl " << operation << " failed: 0x" << std::hex
                << static_cast<unsigned int>(code);
        throw std::runtime_error(message.str());
    }
}

inline cl_ulong as_cl_ulong(size_t value, const char* message) {
    if (value > static_cast<size_t>(std::numeric_limits<cl_ulong>::max())) throw std::invalid_argument(message);
    return static_cast<cl_ulong>(value);
}

inline void argument(cl_kernel kernel, cl_uint index, size_t length, const void* value) {
    check(clSetKernelArg(kernel, index, length, value), "clSetKernelArg");
}

}  // namespace geniex::vision::opencl
