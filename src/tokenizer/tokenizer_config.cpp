// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "tokenizer_config.h"

#include <stdexcept>

#include <nlohmann/json.hpp>

#include "src/internal/utils.h"

namespace geniex::internal {

namespace {

// HF tokenizer_config.json sometimes stores special tokens as plain strings
// and sometimes as AddedToken objects with a "content" field. Accept both.
std::string parse_token_field(const nlohmann::json& v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_object() && v.contains("content") && v.at("content").is_string()) {
        return v.at("content").get<std::string>();
    }
    return {};
}

}  // namespace

TokenizerConfig load_tokenizer_config(const std::string& path) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(read_file_to_string(path));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "failed to parse tokenizer_config.json at " + path + ": " + e.what());
    }
    if (!j.is_object()) {
        throw std::runtime_error(
            "tokenizer_config.json at " + path + " is not a JSON object");
    }

    TokenizerConfig cfg;
    if (j.contains("chat_template") && j.at("chat_template").is_string()) {
        cfg.chat_template = j.at("chat_template").get<std::string>();
    }
    if (j.contains("bos_token")) {
        cfg.bos_token = parse_token_field(j.at("bos_token"));
    }
    if (j.contains("eos_token")) {
        cfg.eos_token = parse_token_field(j.at("eos_token"));
    }
    return cfg;
}

}  // namespace geniex::internal
