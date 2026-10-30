// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "npu_runtime.h"

#include <AEEStdErr.h>
#include <remote.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "geniex_proc_htp.h"

namespace geniex::vision::npu {

void check(int32_t status, const char* operation) {
    if (status == AEE_SUCCESS) return;
    char code[16];
    std::snprintf(code, sizeof(code), "0x%x", static_cast<unsigned>(status));
    throw std::runtime_error(std::string("geniex::vision::npu: ") + operation + " failed with AEEResult " + code);
}

namespace {

constexpr int kCdspDomain = 3;
constexpr unsigned int kBuiltArchitectures[] = {GENIEXPROC_HTP_ARCHS};

std::string skel_uri() {
    fastrpc_capability info = {};
    info.domain = kCdspDomain;
    info.attribute_ID = ARCH_VER;
    check(remote_handle_control(DSPRPC_GET_DSP_INFO, &info, sizeof(info)), "query DSP architecture");
    const unsigned int value = info.capability & 0xffu;
    if (value == 0 || (value & 0x0fu) > 9 || (value >> 4) > 9) {
        throw std::runtime_error("geniex::vision::npu: invalid DSP architecture capability");
    }
    const unsigned int architecture = (value >> 4) * 10u + (value & 0x0fu);
    bool supported = false;
    for (const auto built : kBuiltArchitectures) supported |= architecture == built;
    if (!supported) {
        throw std::runtime_error("geniex::vision::npu: no kernel built for v" + std::to_string(architecture));
    }
    return "file:///libgeniex-proc-htp-v" + std::to_string(architecture) +
           ".so?geniex_proc_htp_skel_handle_invoke&_modver=1.0&_dom=cdsp";
}

}  // namespace

NpuRuntime::NpuRuntime() {
    const auto uri = skel_uri();
    remote_rpc_control_unsigned_module control = {};
    control.domain = kCdspDomain;
    control.enable = 1;
    check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control)),
          "CDSP unsigned-module control");
    remote_handle64 handle = 0;
    check(geniex_proc_htp_open(uri.c_str(), &handle), "open");
    handle_ = handle;
}

NpuRuntime::~NpuRuntime() noexcept {
    if (handle_ == 0) return;
    try {
        check(geniex_proc_htp_close(handle_), "close");
    } catch (...) {
        // Destruction is best-effort cleanup; runtime errors are reported by open and calls.
    }
}

NpuAccess NpuRuntime::acquire() {
    std::unique_lock lock(mutex_);
    if (handle_ == 0) throw std::logic_error("geniex::vision::npu: session is not open");
    return {std::move(lock), handle_};
}

}  // namespace geniex::vision::npu
