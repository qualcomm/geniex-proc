// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "src/internal/token_bytes.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <string>

#include <tokenizers_cpp.h>

namespace geniex::internal {

namespace {

// ───────────────────────────────────────────────────────────────────────────
// GPT-2 byte-level alphabet
// ───────────────────────────────────────────────────────────────────────────
//
// The canonical GPT-2 `bytes_to_unicode()` mapping (see openai/gpt-2 and
// huggingface/transformers) defines a bijection between the 256 byte values
// 0..255 and 256 printable, non-whitespace Unicode codepoints.
//
// "Already-printable" bytes map to themselves (e.g. byte 0x41 'A' → U+0041),
// while bytes that would be control / whitespace / unprintable get pushed into
// the U+0100..U+0143 range (68 of them).
//
// We only need the *reverse* direction: codepoint → byte. The largest
// codepoint produced is U+0143 (the reverse of byte 0xAD), so a flat
// 0x144-entry table (covering codepoints 0..0x143) is enough for O(1) lookup.
// -1 marks codepoints that are not part of the alphabet at all.

constexpr int32_t kMaxByteLevelCodepoint = 0x143;

using ByteLevelTable = std::array<int16_t, kMaxByteLevelCodepoint + 1>;

ByteLevelTable build_byte_level_reverse_table() {
    ByteLevelTable t{};
    t.fill(-1);

    // First, "already-printable" bytes that map to themselves.
    auto add_self = [&](int lo, int hi) {
        for (int b = lo; b < hi; ++b) {
            t[static_cast<size_t>(b)] = static_cast<int16_t>(b);
        }
    };
    add_self(33, 127);   // '!'..'~'
    add_self(161, 173);  // ¡..¬   (skips the soft hyphen at 0xAD)
    add_self(174, 256);  // ®..ÿ

    // Then, the unprintable bytes get pushed into U+0100..U+0143 in order.
    int32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        bool already_in_bs = (b >= 33 && b < 127) ||
                             (b >= 161 && b < 173) ||
                             (b >= 174 && b < 256);
        if (!already_in_bs) {
            int32_t cp = 256 + n;
            ++n;
            // cp is in [256..256+68) = [0x100..0x143]. Bounds-check just in
            // case the alphabet definition is ever amended upstream.
            if (cp >= 0 && cp <= kMaxByteLevelCodepoint) {
                t[static_cast<size_t>(cp)] = static_cast<int16_t>(b);
            }
        }
    }
    return t;
}

const ByteLevelTable& byte_level_reverse_table() {
    static const ByteLevelTable kTable = build_byte_level_reverse_table();
    return kTable;
}

// ───────────────────────────────────────────────────────────────────────────
// UTF-8 codepoint walker
// ───────────────────────────────────────────────────────────────────────────
//
// Decodes one codepoint from `s` starting at `i`. On success advances `i` by
// the number of bytes consumed and returns true. On a malformed lead byte or
// truncated continuation, returns false (caller treats the piece as not-byte-
// level, i.e. falls through to verbatim).

bool decode_utf8_codepoint(const std::string& s, size_t& i, uint32_t& cp) {
    if (i >= s.size()) return false;
    uint8_t b0 = static_cast<uint8_t>(s[i]);

    if (b0 < 0x80) {
        cp = b0;
        i += 1;
        return true;
    }

    int extra = 0;
    if      ((b0 & 0xE0) == 0xC0) { extra = 1; cp = b0 & 0x1F; }
    else if ((b0 & 0xF0) == 0xE0) { extra = 2; cp = b0 & 0x0F; }
    else if ((b0 & 0xF8) == 0xF0) { extra = 3; cp = b0 & 0x07; }
    else                          { return false; }

    if (i + 1 + extra > s.size()) return false;
    for (int k = 0; k < extra; ++k) {
        uint8_t bk = static_cast<uint8_t>(s[i + 1 + k]);
        if ((bk & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (bk & 0x3F);
    }
    i += 1 + extra;
    return true;
}

bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

uint8_t hex_value(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
    return static_cast<uint8_t>(c - 'A' + 10);
}

// ───────────────────────────────────────────────────────────────────────────
// Rule A: SentencePiece byte fallback — piece is exactly "<0xNN>" (6 chars).
// ───────────────────────────────────────────────────────────────────────────
bool try_sp_byte_fallback(const std::string& piece, std::string& out) {
    if (piece.size() != 6) return false;
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' ||
        piece[5] != '>') {
        return false;
    }
    if (!is_hex_digit(piece[3]) || !is_hex_digit(piece[4])) return false;
    uint8_t b = static_cast<uint8_t>((hex_value(piece[3]) << 4) |
                                     hex_value(piece[4]));
    out.assign(1, static_cast<char>(b));
    return true;
}

// ───────────────────────────────────────────────────────────────────────────
// Rule B: GPT-2 byte-level — every codepoint must be in the alphabet.
// ───────────────────────────────────────────────────────────────────────────
bool try_byte_level(const std::string& piece, std::string& out) {
    const auto& table = byte_level_reverse_table();
    std::string buf;
    buf.reserve(piece.size());

    size_t i = 0;
    while (i < piece.size()) {
        uint32_t cp = 0;
        if (!decode_utf8_codepoint(piece, i, cp)) return false;
        if (cp > static_cast<uint32_t>(kMaxByteLevelCodepoint)) return false;
        int16_t mapped = table[cp];
        if (mapped < 0) return false;
        buf.push_back(static_cast<char>(static_cast<uint8_t>(mapped)));
    }
    out = std::move(buf);
    return true;
}

}  // namespace

std::string token_id_to_raw_bytes(tokenizers::Tokenizer* tokenizer,
                                  int32_t token_id) {
    if (!tokenizer) return {};

    std::string piece = tokenizer->IdToToken(token_id);
    if (piece.empty()) return {};

    std::string out;
    if (try_sp_byte_fallback(piece, out)) return out;
    if (try_byte_level(piece, out))       return out;

    // Rule C — verbatim fallback.
    return piece;
}

}  // namespace geniex::internal
