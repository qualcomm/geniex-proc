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
namespace {

constexpr int kCdspDomain = 3;
#define GENIEXPROC_HTP_STRINGIFY_IMPL(value) #value
#define GENIEXPROC_HTP_STRINGIFY(value) GENIEXPROC_HTP_STRINGIFY_IMPL(value)
constexpr char kSkelUri[] = "file:///libgeniex-proc-htp-" GENIEXPROC_HTP_STRINGIFY(
    GENIEXPROC_HTP_DSP_VERSION) ".so?geniex_proc_htp_skel_handle_invoke&_modver=1.0&_dom=cdsp";

[[noreturn]] void rpc_error(const char* operation, AEEResult result) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%x", static_cast<unsigned>(result));
    throw std::runtime_error(std::string("geniex::htp: ") + operation + " failed with AEEResult " + code);
}

}  // namespace

void open(Session& session) {
    std::lock_guard lock(session.mutex_);
    if (session.handle_ != 0) return;

    remote_rpc_control_unsigned_module control = {};
    control.domain = kCdspDomain;
    control.enable = 1;
    const int control_result = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
    if (control_result != 0) rpc_error("CDSP unsigned-module control", control_result);

    remote_handle64 handle = 0;
    const auto result = geniex_proc_htp_open(kSkelUri, &handle);
    if (result != AEE_SUCCESS) rpc_error("open", result);
    session.handle_ = handle;
}

void close(Session& session) {
    std::lock_guard lock(session.mutex_);
    if (session.handle_ == 0) return;

    const auto result = geniex_proc_htp_close(session.handle_);
    session.handle_ = 0;
    if (result != AEE_SUCCESS) rpc_error("close", result);
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
