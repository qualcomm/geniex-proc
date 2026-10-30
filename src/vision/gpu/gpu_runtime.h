// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace geniex::vision::gpu {

class GpuRuntime {
   public:
    class Access {
       public:
        cl_context context() const;
        cl_command_queue queue() const;
        cl_program create_program(const char* source, const char* operation);
        cl_kernel create_kernel(cl_program program, const char* name, const char* operation);
        cl_kernel kernel(std::string_view key, const char* source, const char* name, const char* operation);
        cl_mem buffer(std::string_view name, size_t bytes, cl_mem_flags flags);

       private:
        friend class GpuRuntime;
        explicit Access(GpuRuntime& runtime);
        GpuRuntime& runtime_;
        std::unique_lock<std::mutex> lock_;
    };

    GpuRuntime();
    ~GpuRuntime() noexcept;
    GpuRuntime(const GpuRuntime&) = delete;
    GpuRuntime& operator=(const GpuRuntime&) = delete;

    Access acquire();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex::vision::gpu
