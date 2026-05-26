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

// ─── Chat template ──────────────────────────────────────────────────────────
// Exercised against the Qwen2.5-0.5B tokenizer_config.json fixture downloaded
// by tests/CMakeLists.txt. Skips at runtime if the fixture is missing.

namespace {

class ChatTemplateTest : public ::testing::Test {
protected:
    static std::unique_ptr<geniex::Tokenizer> tok_;

    static void SetUpTestSuite() {
        const fs::path tok_path = GENIEXPROC_TEST_TOKENIZER_PATH;
        const fs::path cfg_path = GENIEXPROC_TEST_TOKENIZER_CONFIG_PATH;
        if (!fs::exists(tok_path) || !fs::exists(cfg_path)) {
            GTEST_SKIP() << "Chat-template fixtures not present "
                         << "(tokenizer.json / tokenizer_config.json).";
        }
        tok_ = geniex::Tokenizer::from_file(tok_path.string(), cfg_path.string());
        ASSERT_NE(tok_, nullptr);
    }

    static void TearDownTestSuite() { tok_.reset(); }
};

std::unique_ptr<geniex::Tokenizer> ChatTemplateTest::tok_{};

}  // namespace

TEST(ChatTemplateLoadingTest, NoTemplateWhenConfigPathEmpty) {
    const fs::path tok_path = GENIEXPROC_TEST_TOKENIZER_PATH;
    if (!fs::exists(tok_path)) GTEST_SKIP() << "tokenizer.json fixture missing";

    auto tok = geniex::Tokenizer::from_file(tok_path.string());
    ASSERT_NE(tok, nullptr);
    EXPECT_FALSE(tok->has_chat_template());
    EXPECT_THROW(tok->apply_chat_template({{geniex::Role::User, "hi"}}),
                 std::runtime_error);
}

TEST(ChatTemplateLoadingTest, MissingConfigFileThrows) {
    const fs::path tok_path = GENIEXPROC_TEST_TOKENIZER_PATH;
    if (!fs::exists(tok_path)) GTEST_SKIP() << "tokenizer.json fixture missing";

    EXPECT_THROW(
        geniex::Tokenizer::from_file(tok_path.string(),
                                     "/definitely/does/not/exist.json"),
        std::runtime_error);
}

TEST_F(ChatTemplateTest, HasTemplateWhenConfigPathProvided) {
    ASSERT_NE(tok_, nullptr);
    EXPECT_TRUE(tok_->has_chat_template());
}

TEST_F(ChatTemplateTest, RendersDefaultSystemPrompt) {
    ASSERT_NE(tok_, nullptr);

    // Qwen2.5's template injects "You are a helpful assistant." when no
    // explicit system message is given.
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "Hi"},
    };
    const auto out = tok_->apply_chat_template(msgs);
    EXPECT_NE(out.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"),
              std::string::npos) << out;
    EXPECT_NE(out.find("<|im_start|>user\nHi<|im_end|>\n"), std::string::npos) << out;
    EXPECT_NE(out.find("<|im_start|>assistant\n"), std::string::npos) << out;
}

TEST_F(ChatTemplateTest, RespectsExplicitSystem) {
    ASSERT_NE(tok_, nullptr);

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "You are a pirate."},
        {geniex::Role::User,   "Hi"},
    };
    const auto out = tok_->apply_chat_template(msgs);
    EXPECT_NE(out.find("<|im_start|>system\nYou are a pirate.<|im_end|>\n"),
              std::string::npos) << out;
    // Default fallback must NOT appear when an explicit system was given.
    EXPECT_EQ(out.find("You are a helpful assistant."), std::string::npos) << out;
}

TEST_F(ChatTemplateTest, AddGenerationPromptToggle) {
    ASSERT_NE(tok_, nullptr);

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "Hi"},
    };

    geniex::Tokenizer::ApplyChatTemplateOptions on;
    on.add_generation_prompt = true;
    const auto with_prompt = tok_->apply_chat_template(msgs, on);

    geniex::Tokenizer::ApplyChatTemplateOptions off;
    off.add_generation_prompt = false;
    const auto without_prompt = tok_->apply_chat_template(msgs, off);

    EXPECT_NE(with_prompt.length(), without_prompt.length());
    EXPECT_TRUE(with_prompt.size() > without_prompt.size()) << with_prompt;
    // The trailing assistant header is what the flag toggles.
    EXPECT_TRUE(with_prompt.rfind("<|im_start|>assistant\n") + 22 == with_prompt.size())
        << with_prompt;
}

TEST_F(ChatTemplateTest, ToolCallRoundtrip) {
    ASSERT_NE(tok_, nullptr);

    geniex::ChatMessage assistant_call;
    assistant_call.role = geniex::Role::Assistant;
    assistant_call.tool_calls.push_back(
        geniex::ToolCall{
            /*id=*/"call_1",
            /*name=*/"get_weather",
            /*arguments_json=*/"{\"location\":\"Paris\"}"});

    geniex::ChatMessage tool_response;
    tool_response.role         = geniex::Role::Tool;
    tool_response.tool_call_id = "call_1";
    tool_response.name         = "get_weather";
    tool_response.content      = "22C";

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "What's the weather?"},
        assistant_call,
        tool_response,
    };

    geniex::Tokenizer::ApplyChatTemplateOptions opts;
    opts.tools_json =
        R"([{"type":"function","function":{"name":"get_weather",)"
        R"("description":"Look up the weather.","parameters":)"
        R"({"type":"object","properties":{"location":{"type":"string"}},)"
        R"("required":["location"]}}}])";

    const auto out = tok_->apply_chat_template(msgs, opts);

    EXPECT_NE(out.find("<tools>"),         std::string::npos) << out;
    EXPECT_NE(out.find("get_weather"),     std::string::npos) << out;
    EXPECT_NE(out.find("<tool_call>"),     std::string::npos) << out;
    EXPECT_NE(out.find("\"location\""),    std::string::npos) << out;
    EXPECT_NE(out.find("<tool_response>\n22C\n</tool_response>"),
              std::string::npos) << out;
}

TEST_F(ChatTemplateTest, InvalidArgumentsJsonThrows) {
    ASSERT_NE(tok_, nullptr);

    geniex::ChatMessage assistant_call;
    assistant_call.role = geniex::Role::Assistant;
    assistant_call.tool_calls.push_back(
        geniex::ToolCall{"id1", "f", "definitely-not-json"});

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "Hi"},
        assistant_call,
    };

    try {
        tok_->apply_chat_template(msgs);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("messages[1]"),    std::string::npos) << what;
        EXPECT_NE(what.find("arguments_json"), std::string::npos) << what;
    }
}

TEST_F(ChatTemplateTest, InvalidToolsJsonThrows) {
    ASSERT_NE(tok_, nullptr);

    geniex::Tokenizer::ApplyChatTemplateOptions opts;
    opts.tools_json = "not-json";

    EXPECT_THROW(
        tok_->apply_chat_template({{geniex::Role::User, "Hi"}}, opts),
        std::runtime_error);
}

TEST_F(ChatTemplateTest, ChatTemplateOverride) {
    ASSERT_NE(tok_, nullptr);

    geniex::Tokenizer::ApplyChatTemplateOptions opts;
    opts.add_generation_prompt   = false;
    opts.chat_template_override =
        "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}";

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User,      "ping"},
        {geniex::Role::Assistant, "pong"},
    };
    const auto out = tok_->apply_chat_template(msgs, opts);
    EXPECT_EQ(out, "user:ping\nassistant:pong\n");
}
