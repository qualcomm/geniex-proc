// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_
#define GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_

#include <cstdint>
#include <string>

namespace tokenizers {
class Tokenizer;
}

namespace geniex::internal {

// Decode a single token id to its raw byte sequence (potentially a partial
// UTF-8 multi-byte sequence). Bypasses tokenizers-cpp::Decode so partial
// codepoints are *not* substituted with U+FFFD.
//
// A piece is resolved in three steps:
//   1. A SentencePiece byte-fallback piece, the literal 6-char pattern
//      "<0xNN>", becomes the single byte it names.
//   2. When `byte_level_encoding` is true, a GPT-2 byte-level piece (every
//      codepoint is in the 256-entry byte-level alphabet) is mapped back to
//      its raw bytes.
//   3. Otherwise the piece is returned verbatim, with the SentencePiece
//      metaspace U+2581 rewritten to a space. This covers special tokens,
//      SentencePiece word pieces, and plain ASCII / already-UTF-8 pieces.
//
// `byte_level_encoding` MUST reflect whether the tokenizer actually uses GPT-2
// byte-level encoding. The byte-level alphabet overlaps Latin-1 (bytes
// 0xA1..0xFF map to themselves as codepoints U+00A1..U+00FF), so a
// SentencePiece-BPE tokenizer that stores a Latin-1 letter as a direct 2-byte
// UTF-8 piece (e.g. Gemma's "ü" = C3 BC) would be misread as byte-level and
// collapsed to the single Latin-1 byte 0xFC, corrupting every non-ASCII
// character. Setting the flag correctly keeps such pieces verbatim, where the
// UTF-8 bytes are already right, while byte-fallback still resolves normally.
std::string token_id_to_raw_bytes(tokenizers::Tokenizer* tokenizer, int32_t token_id, bool byte_level_encoding);

// Detects whether a tokenizer.json blob describes GPT-2 byte-level encoding
// (a ByteLevel decoder, standalone or inside a Sequence) as opposed to
// SentencePiece byte-fallback. This is the value to pass as
// `byte_level_encoding` above. Returns false on parse failure.
bool tokenizer_json_is_byte_level(const std::string& tokenizer_json);

}  // namespace geniex::internal

#endif  // GENIEX_PROC_SRC_INTERNAL_TOKEN_BYTES_H_
