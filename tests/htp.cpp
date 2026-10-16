// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "htp/htp_runtime.h"

namespace htp = geniex::htp;

TEST(HtpSession, ClosedSessionCannotBeAcquired) {
    htp::Session session;
    EXPECT_THROW(htp::acquire(session), std::logic_error);
    EXPECT_NO_THROW(htp::close(session));
}

TEST(HtpSession, OpenCloseAndReopen) {
    htp::Session session;
    ASSERT_NO_THROW(htp::open(session));
    EXPECT_NO_THROW(htp::open(session));
    {
        auto access = htp::acquire(session);
        EXPECT_NE(access.handle, 0U);
    }
    EXPECT_NO_THROW(htp::close(session));
    EXPECT_NO_THROW(htp::close(session));
    EXPECT_THROW(htp::acquire(session), std::logic_error);

    ASSERT_NO_THROW(htp::open(session));
    {
        auto access = htp::acquire(session);
        EXPECT_NE(access.handle, 0U);
    }
    EXPECT_NO_THROW(htp::close(session));
}

TEST(HtpSession, DefaultSessionIsReused) {
    auto& session = htp::default_session();
    EXPECT_EQ(&session, &htp::default_session());
    std::uint64_t handle;
    {
        auto access = htp::acquire(session);
        handle = access.handle;
        EXPECT_NE(handle, 0U);
    }
    {
        auto access = htp::acquire(session);
        EXPECT_EQ(access.handle, handle);
    }
}
