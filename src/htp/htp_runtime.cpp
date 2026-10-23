// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "htp_runtime.h"

#include <AEEStdErr.h>
#include <remote.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "geniex_proc_htp.h"

namespace geniex::htp {

void check(int32_t status, const char* operation) {
    if (status == AEE_SUCCESS) return;
    char code[16];
    std::snprintf(code, sizeof(code), "0x%x", static_cast<unsigned>(status));
    throw std::runtime_error(std::string("geniex::htp: ") + operation + " failed with AEEResult " + code);
}

namespace {

constexpr int kCdspDomain = 3;
constexpr unsigned int kBuiltArchitectures[] = {GENIEXPROC_HTP_ARCHS};

std::string skel_uri() {
    fastrpc_capability info = {};
    info.domain = kCdspDomain;
    info.attribute_ID = ARCH_VER;
    const auto result = remote_handle_control(DSPRPC_GET_DSP_INFO, &info, sizeof(info));
    check(result, "query DSP architecture");
    const unsigned int value = info.capability & 0xffu;
    if (value == 0 || (value & 0x0fu) > 9 || (value >> 4) > 9) {
        throw std::runtime_error("geniex::htp: invalid DSP architecture capability");
    }
    const unsigned int architecture = (value >> 4) * 10u + (value & 0x0fu);
    bool supported = false;
    for (const auto built : kBuiltArchitectures) supported |= architecture == built;
    if (!supported) {
        throw std::runtime_error("geniex::htp: no kernel built for v" + std::to_string(architecture));
    }
    return "file:///libgeniex-proc-htp-v" + std::to_string(architecture) +
           ".so?geniex_proc_htp_skel_handle_invoke&_modver=1.0&_dom=cdsp";
}

}  // namespace

void open(Session& session) {
    std::lock_guard lock(session.mutex_);
    if (session.handle_ != 0) return;
    const auto uri = skel_uri();

    remote_rpc_control_unsigned_module control = {};
    control.domain = kCdspDomain;
    control.enable = 1;
    const int control_result = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
    check(control_result, "CDSP unsigned-module control");

    remote_handle64 handle = 0;
    const auto result = geniex_proc_htp_open(uri.c_str(), &handle);
    check(result, "open");
    session.handle_ = handle;
}

void close(Session& session) {
    std::lock_guard lock(session.mutex_);
    if (session.handle_ == 0) return;

    const auto result = geniex_proc_htp_close(session.handle_);
    session.handle_ = 0;
    check(result, "close");
}

SessionGuard acquire(Session& session) {
    std::unique_lock lock(session.mutex_);
    if (session.handle_ == 0) throw std::logic_error("geniex::htp: session is not open");
    return {std::move(lock), session.handle_};
}

Session::~Session() noexcept {
    try {
        close(*this);
    } catch (...) {
        // Explicit close reports errors; destruction is best-effort cleanup.
    }
}

Session& default_session() {
    struct Owner {
        Session session;
        Owner() { open(session); }
    };
    static Owner owner;
    return owner.session;
}

}  // namespace geniex::htp
