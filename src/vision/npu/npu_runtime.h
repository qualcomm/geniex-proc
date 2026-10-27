// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <mutex>

namespace geniex::vision::npu {

struct NpuAccess {
    std::unique_lock<std::mutex> lock;
    std::uint64_t handle;
};

// Owns one FastRPC session; the lock serializes use of its shared DSP handle.
class NpuRuntime {
   public:
    NpuRuntime();
    ~NpuRuntime() noexcept;
    NpuRuntime(const NpuRuntime&) = delete;
    NpuRuntime& operator=(const NpuRuntime&) = delete;

    NpuAccess acquire();

   private:
    std::uint64_t handle_ = 0;
    std::mutex mutex_;
};

void check(int32_t status, const char* operation);

}  // namespace geniex::vision::npu
