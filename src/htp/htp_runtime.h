// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <mutex>

namespace geniex::htp {

struct SessionGuard {
    std::unique_lock<std::mutex> lock;
    std::uint64_t handle;
};

class Session {
   public:
    Session() = default;
    ~Session() noexcept;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

   private:
    friend void open(Session& session);
    friend void close(Session& session);
    friend SessionGuard acquire(Session& session);

    std::uint64_t handle_ = 0;
    std::mutex mutex_;
};

void open(Session& session);
void close(Session& session);
SessionGuard acquire(Session& session);
Session& default_session();

}  // namespace geniex::htp
