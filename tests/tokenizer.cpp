// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for geniex::Tokenizer.

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "geniex-proc/tokenizer.h"

namespace fs = std::filesystem;

namespace {

// Shared tokenizer — loaded once for all tests in this binary.
class TokenizerTest : public ::testing::Test {
protected:
    static std::unique_ptr<geniex::Tokenizer> tok_;
    static bool tried_load_;

    static void SetUpTestSuite() {
        const fs::path path = GENIEXPROC_TEST_TOKENIZER_PATH;
        tried_load_ = true;
        if (!fs::exists(path)) {
            GTEST_SKIP() << "Tokenizer fixture not present at " << path
                         << ". CMake download may have failed; pass "
                            "-DGENIEXPROC_TEST_TOKENIZER=<path>.";
        }
        tok_ = geniex::Tokenizer::from_file(path.string());
        ASSERT_NE(tok_, nullptr);
    }

    static void TearDownTestSuite() {
        tok_.reset();
    }
};

std::unique_ptr<geniex::Tokenizer> TokenizerTest::tok_{};
bool TokenizerTest::tried_load_ = false;

}  // namespace

// ─── Loading ────────────────────────────────────────────────────────────────

TEST_F(TokenizerTest, VocabSizeIsPositive) {
    ASSERT_NE(tok_, nullptr);
    EXPECT_GT(tok_->vocab_size(), 0);
}

// Note: tokenizers-cpp's FromJSON aborts the process on a missing file
// rather than returning nullptr or throwing, so we can't test the
// "missing file" path from here without custom process-death assertions.

// ─── Encode / decode roundtrip ───────────────────────────────────────────────

TEST_F(TokenizerTest, EncodeDecodeRoundtripAscii) {
    ASSERT_NE(tok_, nullptr);

    const std::string text = "Hello, world!";
    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    EXPECT_FALSE(ids.empty());

    auto decoded = tok_->decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, EncodeDecodeRoundtripUtf8) {
    ASSERT_NE(tok_, nullptr);

    // Mix of Latin, punctuation, and CJK to catch UTF-8 edge cases. The file
    // is saved as UTF-8 so a plain narrow literal with escape sequences gives
    // us portable bytes without relying on C++20 char8_t conversions.
    //   \xe4\xb8\x96\xe7\x95\x8c = 世界   (U+4E16 U+754C)
    //   \xe4\xbd\xa0\xe5\xa5\xbd = 你好   (U+4F60 U+597D)
    const std::string text =
        "Hello, \xe4\xb8\x96\xe7\x95\x8c! (Qwen says \xe4\xbd\xa0\xe5\xa5\xbd)";

    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    EXPECT_FALSE(ids.empty());

    auto decoded = tok_->decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, EmptyInputProducesNoTokens) {
    ASSERT_NE(tok_, nullptr);
    auto ids = tok_->encode("", /*add_special_tokens=*/false);
    EXPECT_TRUE(ids.empty());

    auto text = tok_->decode({}, /*skip_special_tokens=*/true);
    EXPECT_EQ(text, "");
}

// ─── Streaming decode: decode({single_id}) ──────────────────────────────────
// Mirrors the hot loop in llm_pipeline.cpp / vlm_pipeline.cpp, which calls
// `tokenizer->decode({tok})` once per generated token. Concatenating those
// pieces must reconstruct the original text for any encoded input.

TEST_F(TokenizerTest, StreamingDecodeConcatenatesToFullText) {
    ASSERT_NE(tok_, nullptr);

    const std::string text = "Hello, world! 12345";
    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    std::string rebuilt;
    for (int32_t id : ids) {
        rebuilt += tok_->decode({id}, /*skip_special_tokens=*/true);
    }
    EXPECT_EQ(rebuilt, text);
}

// ─── id_to_piece / piece_to_id inverse ───────────────────────────────────────

TEST_F(TokenizerTest, PieceIdRoundtripsKnownToken) {
    ASSERT_NE(tok_, nullptr);

    // Encode a known short word to get a guaranteed-valid token id.
    auto ids = tok_->encode("hello", /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    const int32_t id    = ids[0];
    const std::string p = tok_->id_to_piece(id);
    EXPECT_FALSE(p.empty());

    const int32_t back = tok_->piece_to_id(p);
    EXPECT_EQ(back, id);
}

TEST_F(TokenizerTest, PieceToIdReturnsMinusOneForUnknownPiece) {
    ASSERT_NE(tok_, nullptr);
    // This string is deliberately unlikely to be a real vocab piece.
    EXPECT_EQ(tok_->piece_to_id("\x01\x02\x03_not_a_real_piece"), -1);
}

// ─── Special tokens ─────────────────────────────────────────────────────────

TEST_F(TokenizerTest, EosTokenIsRecognized) {
    ASSERT_NE(tok_, nullptr);
    const int32_t eos = tok_->token_eos();
    if (eos == -1) GTEST_SKIP() << "Tokenizer has no EOS token";
    EXPECT_TRUE(tok_->is_eog(eos));
}

TEST_F(TokenizerTest, NonEogTokensAreNotEog) {
    ASSERT_NE(tok_, nullptr);
    // Token 0 is usually <s>/BOS or <unk>, not EOS.
    // It is possible (but unusual) for a tokenizer to mark id 0 as EOG; in
    // that case just don't assert here.
    if (tok_->is_eog(0)) GTEST_SKIP() << "Token 0 is EOG in this vocab";

    auto ids = tok_->encode("hello", /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());
    EXPECT_FALSE(tok_->is_eog(ids[0]));
}

// ─── Streaming-safe per-token decode (decode_token + stream_safe_utf8 arg)

namespace {

// True iff `s` contains at least one U+FFFD (encoded as EF BF BD).
bool contains_replacement_char(const std::string& s) {
    for (size_t i = 0; i + 2 < s.size(); ++i) {
        if (static_cast<uint8_t>(s[i])     == 0xEF &&
            static_cast<uint8_t>(s[i + 1]) == 0xBF &&
            static_cast<uint8_t>(s[i + 2]) == 0xBD) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST_F(TokenizerTest, DecodeTokenDefaultMatchesExplicitTrue) {
    ASSERT_NE(tok_, nullptr);
    auto ids = tok_->encode("Hello", /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());
    for (int32_t id : ids) {
        EXPECT_EQ(tok_->decode_token(id),
                  tok_->decode_token(id, /*stream_safe_utf8=*/true));
    }
}

TEST_F(TokenizerTest, DecodeTokenStreamSafePreservesUtf8MultiByte) {
    ASSERT_NE(tok_, nullptr);

    // Pure CJK input — every codepoint is 3-byte UTF-8 and almost guaranteed
    // to be split across multiple byte-level tokens.
    //   \xe4\xbd\xa0\xe5\xa5\xbd          = 你好
    //   \xe4\xb8\x96\xe7\x95\x8c          = 世界
    //   \xf0\x9f\x9a\x80                  = 🚀  (4-byte)
    const std::string text =
        "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x9a\x80";

    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    std::string rebuilt;
    for (int32_t id : ids) rebuilt += tok_->decode_token(id, /*stream_safe_utf8=*/true);

    EXPECT_EQ(rebuilt, text);
    EXPECT_FALSE(contains_replacement_char(rebuilt));
}

TEST_F(TokenizerTest, DecodeTokenStreamSafeMixedAsciiAndUtf8) {
    ASSERT_NE(tok_, nullptr);

    const std::string text =
        "Hello, \xe4\xb8\x96\xe7\x95\x8c! \xf0\x9f\x9a\x80";

    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    std::string rebuilt;
    for (int32_t id : ids) rebuilt += tok_->decode_token(id, /*stream_safe_utf8=*/true);

    EXPECT_EQ(rebuilt, text);
    EXPECT_FALSE(contains_replacement_char(rebuilt));
}

TEST_F(TokenizerTest, DecodeTokenUnsafeMatchesLegacyDecode) {
    ASSERT_NE(tok_, nullptr);

    // OFF mode must be byte-identical to tok_->decode({id}).
    auto ids = tok_->encode("Hello, world!", /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    for (int32_t id : ids) {
        EXPECT_EQ(tok_->decode_token(id, /*stream_safe_utf8=*/false),
                  tok_->decode({id}, /*skip_special_tokens=*/true));
    }
}

TEST_F(TokenizerTest, DecodeTokenStreamSafeAsciiMatchesPlainText) {
    ASSERT_NE(tok_, nullptr);

    const std::string text = "Hello, world! 12345";
    auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());

    std::string rebuilt;
    for (int32_t id : ids) rebuilt += tok_->decode_token(id);
    EXPECT_EQ(rebuilt, text);
}
