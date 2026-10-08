// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <nlohmann/json.hpp>
#include <utility>

#include "geniex-proc/tokenizer.h"

namespace geniex::internal {

template <typename Json>
Json parse_tools(const ApplyChatTemplateOptions& opts) {
    if (!opts.tools_json.empty()) return Json::parse(opts.tools_json);

    Json tools = Json::array();
    for (const auto& tool : opts.tools) {
        Json fn;
        fn["name"] = tool.name;
        fn["description"] = tool.description;
        fn["parameters"] = tool.parameters_json.empty() ? Json::object() : Json::parse(tool.parameters_json);
        tools.push_back({{"type", "function"}, {"function", std::move(fn)}});
    }
    return tools;
}

}  // namespace geniex::internal
