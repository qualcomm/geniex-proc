// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENIEX_PROC_SRC_TOKENIZER_TOKENIZER_CONFIG_H_
#define GENIEX_PROC_SRC_TOKENIZER_TOKENIZER_CONFIG_H_

#include <string>

namespace geniex::internal {

// Subset of HuggingFace tokenizer_config.json that minja's chat-template
// engine consumes. All fields default to empty when absent in the source.
struct TokenizerConfig {
    std::string chat_template;
    std::string bos_token;
    std::string eos_token;
};

// Parses tokenizer_config.json. `bos_token`/`eos_token` may be either a
// plain string or an `{"content": "..."}` AddedToken object in HF's schema;
// both are accepted.
//
// Throws std::runtime_error if the file cannot be read or parsed.
TokenizerConfig load_tokenizer_config(const std::string& path);

}  // namespace geniex::internal

#endif  // GENIEX_PROC_SRC_TOKENIZER_TOKENIZER_CONFIG_H_
