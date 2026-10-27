// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu_runtime.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencl_utils.h"

namespace geniex::vision::gpu {
namespace {

std::string cl_error(cl_int status) {
    std::ostringstream message;
    message << "0x" << std::hex << static_cast<unsigned int>(status);
    return message.str();
}

}  // namespace

struct GpuRuntime::Impl {
    struct Buffer {
        cl_mem memory = nullptr;
        size_t capacity = 0;
        cl_mem_flags flags = 0;
    };

    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
    std::vector<cl_program> programs;
    std::vector<cl_kernel> kernels;
    std::map<std::string, cl_kernel> named_kernels;
    std::map<std::string, Buffer> buffers;
    std::mutex mutex;

    Impl() {
        cl_uint platform_count = 0;
        opencl::check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
        if (platform_count == 0) throw std::runtime_error("geniex::vision::GpuRuntime: no OpenCL platforms");
        std::vector<cl_platform_id> platforms(platform_count);
        opencl::check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

        for (cl_platform_id platform : platforms) {
            cl_uint count = 0;
            const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
            if (status == CL_DEVICE_NOT_FOUND) continue;
            opencl::check(status, "clGetDeviceIDs");
            if (count != 0) {
                opencl::check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");
                break;
            }
        }
        if (!device) throw std::runtime_error("geniex::vision::GpuRuntime: no OpenCL GPU device (no CPU fallback)");

        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        opencl::check(status, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &status);
        if (status != CL_SUCCESS) {
            if (queue) clReleaseCommandQueue(queue);
            queue = nullptr;
            clReleaseContext(context);
            context = nullptr;
            opencl::check(status, "clCreateCommandQueue");
        }
    }

    ~Impl() {
        if (queue) clFinish(queue);
        for (auto& [name, buffer] : buffers) {
            (void)name;
            if (buffer.memory) clReleaseMemObject(buffer.memory);
        }
        for (cl_kernel kernel : kernels) clReleaseKernel(kernel);
        for (cl_program program : programs) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

GpuRuntime::Access::Access(GpuRuntime& runtime) : runtime_(runtime), lock_(runtime.impl_->mutex) {}
cl_context GpuRuntime::Access::context() const { return runtime_.impl_->context; }
cl_command_queue GpuRuntime::Access::queue() const { return runtime_.impl_->queue; }

cl_program GpuRuntime::Access::create_program(const char* source, const char* operation) {
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context(), 1, &source, nullptr, &status);
    if (status != CL_SUCCESS) {
        if (program) clReleaseProgram(program);
        opencl::check(status, operation);
    }
    status = clBuildProgram(program, 1, &runtime_.impl_->device, "-cl-std=CL1.2", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t length = 0;
        clGetProgramBuildInfo(program, runtime_.impl_->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
        std::string log(length, '\0');
        if (length) clGetProgramBuildInfo(program, runtime_.impl_->device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error(std::string("geniex::vision::GpuRuntime: ") + operation + " failed (" +
                                 cl_error(status) + "): " + log);
    }
    runtime_.impl_->programs.push_back(program);
    return program;
}

cl_kernel GpuRuntime::Access::create_kernel(cl_program program, const char* name, const char* operation) {
    cl_int status = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, name, &status);
    if (status != CL_SUCCESS) {
        if (kernel) clReleaseKernel(kernel);
        opencl::check(status, operation);
    }
    runtime_.impl_->kernels.push_back(kernel);
    return kernel;
}

cl_kernel GpuRuntime::Access::kernel(std::string_view key, const char* source, const char* name,
                                     const char* operation) {
    const std::string cache_key(key);
    const auto found = runtime_.impl_->named_kernels.find(cache_key);
    if (found != runtime_.impl_->named_kernels.end()) return found->second;
    const cl_program program = create_program(source, operation);
    const cl_kernel result = create_kernel(program, name, operation);
    runtime_.impl_->named_kernels.emplace(cache_key, result);
    return result;
}

cl_mem GpuRuntime::Access::buffer(std::string_view name, size_t bytes, cl_mem_flags flags) {
    if (bytes == 0) throw std::invalid_argument("geniex::vision::GpuRuntime::buffer: size must be nonzero");
    auto& slot = runtime_.impl_->buffers[std::string(name)];
    if (slot.memory && slot.capacity >= bytes && slot.flags == flags) return slot.memory;

    cl_int status = CL_SUCCESS;
    cl_mem replacement = clCreateBuffer(context(), flags, bytes, nullptr, &status);
    if (status != CL_SUCCESS) {
        if (replacement) clReleaseMemObject(replacement);
        opencl::check(status, "clCreateBuffer");
    }
    if (slot.memory) clReleaseMemObject(slot.memory);
    slot.memory = replacement;
    slot.capacity = bytes;
    slot.flags = flags;
    return slot.memory;
}

GpuRuntime::GpuRuntime() : impl_(std::make_unique<Impl>()) {}
GpuRuntime::~GpuRuntime() noexcept = default;
GpuRuntime::Access GpuRuntime::acquire() { return Access(*this); }

}  // namespace geniex::vision::gpu
