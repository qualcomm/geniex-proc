// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Re-exports the vendored single-header nlohmann::json (`include/json.hpp`)
// at its canonical `<nlohmann/json.hpp>` path so upstream libraries (e.g.
// minja) resolve against the same TU-local copy instead of pulling a
// duplicate via FetchContent.
#pragma once

#include "../json.hpp"
