// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for geniex::internal::token_id_to_raw_bytes and
// tokenizer_json_is_byte_level. These exercise the raw-byte detokenization
// rules directly against synthetic vocab pieces, with no downloaded fixture,
// so they run everywhere CI does.

#include "src/internal/token_bytes.h"

#include <gtest/gtest.h>
#include <tokenizers_cpp.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using geniex::internal::token_id_to_raw_bytes;
using geniex::internal::tokenizer_json_is_byte_level;

namespace {

// Minimal in-memory tokenizer: only IdToToken is meaningful; the rest satisfy
// the abstract interface. Lets us drive token_id_to_raw_bytes with exactly the
// vocab pieces we want to test.
class FakeTokenizer : public tokenizers::Tokenizer {
   public:
    explicit FakeTokenizer(std::unordered_map<int32_t, std::string> vocab) : vocab_(std::move(vocab)) {}

    std::vector<int32_t> Encode(const std::string&) override { return {}; }
    std::string Decode(const std::vector<int32_t>&) override { return {}; }
    size_t GetVocabSize() override { return vocab_.size(); }
    int32_t TokenToId(const std::string&) override { return -1; }

    std::string IdToToken(int32_t id) override {
        auto it = vocab_.find(id);
        return it == vocab_.end() ? std::string{} : it->second;
    }

   private:
    std::unordered_map<int32_t, std::string> vocab_;
};

std::string hex(const std::string& s) {
    static const char* d = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        out.push_back(d[c >> 4]);
        out.push_back(d[c & 0xF]);
        out.push_back(' ');
    }
    if (!out.empty()) out.pop_back();
    return out;
}

}  // namespace

// ─── SentencePiece family: byte_level_encoding == false ──────────────────────
//
// Gemma/Llama-SP store common Latin-1 letters as *direct UTF-8 pieces*. With
// Rule B off, they must come back as their exact UTF-8 bytes, not be collapsed
// to a single Latin-1 byte (the bug: "ü" C3 BC -> FC).

TEST(TokenBytes, SentencePieceLatin1PiecePreservedVerbatim) {
    FakeTokenizer tok({
        {1, "\xC3\xBC"},  // ü  U+00FC
        {2, "\xC3\x9F"},  // ß  U+00DF
        {3, "\xC3\xA4"},  // ä  U+00E4
        {4, "\xC3\xA9"},  // é  U+00E9
        {5, "\xC2\xB0"},  // °  U+00B0
        {6, "\xC2\xB2"},  // ²  U+00B2
    });

    struct Case {
        int32_t id;
        std::string expected;
    };
    const Case cases[] = {
        {1, "\xC3\xBC"}, {2, "\xC3\x9F"}, {3, "\xC3\xA4"}, {4, "\xC3\xA9"}, {5, "\xC2\xB0"}, {6, "\xC2\xB2"},
    };
    for (const auto& c : cases) {
        const std::string got = token_id_to_raw_bytes(&tok, c.id, /*byte_level_encoding=*/false);
        EXPECT_EQ(got, c.expected) << "id=" << c.id << " got=[" << hex(got) << "] want=[" << hex(c.expected) << "]";
    }
}

// A word piece like "süße" (whole word in the vocab) must survive intact.
TEST(TokenBytes, SentencePieceWordWithLatin1PreservedVerbatim) {
    FakeTokenizer tok({{10,
                        "s\xC3\xBC\xC3\x9F"
                        "e"}});  // "süße"
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 10, /*byte_level_encoding=*/false),
              "s\xC3\xBC\xC3\x9F"
              "e");
}

// Metaspace ▁ (U+2581) still becomes a space on the SentencePiece path.
TEST(TokenBytes, SentencePieceMetaspaceBecomesSpace) {
    FakeTokenizer tok({{11, "\xE2\x96\x81hello"}});  // "▁hello"
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 11, /*byte_level_encoding=*/false), " hello");
}

// Byte-fallback pieces "<0xNN>" are decoded on either path (Rule A).
TEST(TokenBytes, ByteFallbackPieceDecodesToSingleByte) {
    FakeTokenizer tok({{12, "<0xC3>"}, {13, "<0xBC>"}});
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 12, /*byte_level_encoding=*/false), std::string(1, '\xC3'));
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 13, /*byte_level_encoding=*/false), std::string(1, '\xBC'));
}

// ─── GPT-2 byte-level family: byte_level_encoding == true ────────────────────
//
// Byte-level tokenizers remap each raw byte to a printable codepoint. Rule B
// must reverse that mapping. "Ĥ" (U+0124) is the remap of byte 0x84; "Ġ"
// (U+0120) is the remap of a leading space.

TEST(TokenBytes, ByteLevelPieceMapsBackToRawBytes) {
    FakeTokenizer tok({
        {20, "\xC4\xA0"},  // Ġ U+0120 -> byte 0x20 (space)
        {21, "A"},         // self-mapped
    });
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 20, /*byte_level_encoding=*/true), " ");
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 21, /*byte_level_encoding=*/true), "A");
}

// The regression guard, stated as intent: the SAME Latin-1 UTF-8 piece is
// interpreted differently by the two modes. Byte-level collapses "ü" (C3 BC)
// to the two raw bytes C3, BC (each codepoint self-maps); SentencePiece keeps
// it verbatim as C3 BC. Both happen to be C3 BC here BUT a single-codepoint
// Latin-1 piece diverges — that's the bug surface.
TEST(TokenBytes, ModesDivergeOnSingleLatin1Codepoint) {
    FakeTokenizer tok({{30, "\xC3\xBC"}});  // ü, one codepoint U+00FC
    // SentencePiece: verbatim 2-byte UTF-8.
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 30, /*byte_level_encoding=*/false), "\xC3\xBC");
    // Byte-level: U+00FC self-maps to the single byte 0xFC.
    EXPECT_EQ(token_id_to_raw_bytes(&tok, 30, /*byte_level_encoding=*/true), std::string(1, '\xFC'));
}

// ─── Decoder-type detection ──────────────────────────────────────────────────

TEST(TokenBytes, DetectsByteLevelSequenceDecoder) {
    const std::string json = R"({
        "decoder": {"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"_"},"content":" "},
            {"type":"ByteLevel"}
        ]}
    })";
    EXPECT_TRUE(tokenizer_json_is_byte_level(json));
}

TEST(TokenBytes, DetectsBareByteLevelDecoder) {
    EXPECT_TRUE(tokenizer_json_is_byte_level(R"({"decoder":{"type":"ByteLevel"}})"));
}

TEST(TokenBytes, SentencePieceByteFallbackIsNotByteLevel) {
    // Gemma-shaped decoder: Replace(▁->space) -> ByteFallback -> Fuse.
    const std::string json = R"({
        "decoder": {"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"\u2581"},"content":" "},
            {"type":"ByteFallback"},
            {"type":"Fuse"}
        ]}
    })";
    EXPECT_FALSE(tokenizer_json_is_byte_level(json));
}

TEST(TokenBytes, MalformedOrMissingDecoderIsNotByteLevel) {
    EXPECT_FALSE(tokenizer_json_is_byte_level("not json"));
    EXPECT_FALSE(tokenizer_json_is_byte_level("{}"));
}
