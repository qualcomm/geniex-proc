// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_
#define GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_

#include <cstdint>
#include <string>

namespace tokenizers { class Tokenizer; }

namespace geniex::internal {

// Decode a single token id to its raw byte sequence (potentially a partial
// UTF-8 multi-byte sequence). Bypasses tokenizers-cpp::Decode so partial
// codepoints are *not* substituted with U+FFFD.
//
// Detection rules, applied in order:
//   A) SentencePiece byte fallback — piece exactly matches the literal
//      6-character ASCII pattern "<0xNN>" → returns 1 byte.
//   B) GPT-2 byte-level BPE — every codepoint of the piece is a member of
//      the 256-entry GPT-2 byte-level alphabet → returns the mapped raw bytes.
//   C) Otherwise — returns the piece verbatim. Covers special tokens
//      ("<|im_start|>", "<|endoftext|>"), SentencePiece word pieces ("▁hello"),
//      and plain ASCII pieces.
//
// Rule disambiguation is structural, not configuration-driven: rule (A)'s
// literal `<0xNN>` form contains `<` / `>` which are *not* in rule (B)'s
// alphabet, and rule (B) requires *every* codepoint to map (so SentencePiece
// pieces containing `▁` U+2581, or any CJK literal, fall through to (C)).
std::string token_id_to_raw_bytes(tokenizers::Tokenizer* tokenizer,
                                  int32_t token_id);

}  // namespace geniex::internal

#endif  // GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_
