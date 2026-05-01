// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENIEX_PROC_SRC_INTERNAL_UTILS_H_
#define GENIEX_PROC_SRC_INTERNAL_UTILS_H_

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

inline std::string read_file_to_string(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open file: " + path);
    }

    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

#endif  // GENIEX_PROC_SRC_INTERNAL_UTILS_H_