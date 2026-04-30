// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for the public VLM processor surface used by geniex-qairt-plugin:
//   - geniex::Role / role_to_string
//   - geniex::kDefaultImageMarker / VisionProcessor::image_marker()
//   - Qwen2VLProcessor::create (with and without marker override)
//   - Qwen2VLProcessor::tokenizer()
//   - Qwen2VLProcessor::apply_chat_template (pure text path)
//   - Qwen2VLProcessor::process (end-to-end, text + image)
//
// A small PNG test image is generated at runtime via stb_image_write so the
// suite has no file fixture beyond tokenizer.json.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "geniex-proc/processor.h"
#include "geniex-proc/qwen2vl.h"
#include "geniex-proc/tokenizer.h"
#include "geniex-proc/types.h"

namespace fs = std::filesystem;

// ─── Role helpers ────────────────────────────────────────────────────────────

TEST(Role, StringifiesAllEnumValues) {
    EXPECT_STREQ(geniex::role_to_string(geniex::Role::System),    "system");
    EXPECT_STREQ(geniex::role_to_string(geniex::Role::User),      "user");
    EXPECT_STREQ(geniex::role_to_string(geniex::Role::Assistant), "assistant");
}

// ─── kDefaultImageMarker constant ────────────────────────────────────────────

TEST(ImageMarker, DefaultConstantHasExpectedValue) {
    // Plugin code (vlm_pipeline.cpp) doesn't hard-code this, but anyone who
    // matches on it — including our own process() tests below — relies on it
    // being non-empty and distinct enough not to collide with real vocab.
    EXPECT_STRNE(geniex::kDefaultImageMarker, "");
    EXPECT_STREQ(geniex::kDefaultImageMarker, "<__image__>");
}

// ─── Qwen2VLProcessor — construction & image_marker ──────────────────────────

namespace {

fs::path tokenizer_path() { return fs::path(GENIEXPROC_TEST_TOKENIZER_PATH); }

// Create a processor, skipping the test if the tokenizer fixture is absent.
std::unique_ptr<geniex::qwen2vl::Qwen2VLProcessor> make_processor(
    std::string marker_override = {}) {
    const auto path = tokenizer_path();
    if (!fs::exists(path)) {
        return nullptr;
    }
    geniex::qwen2vl::Qwen2VLConfig cfg;
    return geniex::qwen2vl::Qwen2VLProcessor::create(
        path.string(), cfg, std::move(marker_override));
}

}  // namespace

TEST(Qwen2VLProcessor, CreateUsesDefaultMarker) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";
    EXPECT_EQ(p->image_marker(), geniex::kDefaultImageMarker);
}

TEST(Qwen2VLProcessor, CreateAcceptsMarkerOverride) {
    auto p = make_processor("<|vision_start|>");
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";
    EXPECT_EQ(p->image_marker(), "<|vision_start|>");
}

TEST(Qwen2VLProcessor, TokenizerAccessorReturnsUsableRef) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";
    auto& tok = p->tokenizer();
    EXPECT_GT(tok.vocab_size(), 0);
    auto ids = tok.encode("hello", /*add_special_tokens=*/false);
    EXPECT_FALSE(ids.empty());
}

// ─── apply_chat_template ─────────────────────────────────────────────────────

TEST(Qwen2VLProcessor, ApplyChatTemplateWrapsRoleAndContent) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "hello", /*mm_content=*/{}},
    };
    auto text = p->apply_chat_template(msgs, /*add_generation_prompt=*/false);

    EXPECT_NE(text.find("<|im_start|>user\n"), std::string::npos)  << text;
    EXPECT_NE(text.find("hello"),              std::string::npos)  << text;
    EXPECT_NE(text.find("<|im_end|>"),         std::string::npos)  << text;
    // No assistant prompt without add_generation_prompt.
    EXPECT_EQ(text.find("<|im_start|>assistant"), std::string::npos) << text;
}

TEST(Qwen2VLProcessor, ApplyChatTemplateAppendsAssistantPrompt) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "hi", {}},
    };
    auto text = p->apply_chat_template(msgs, /*add_generation_prompt=*/true);

    // The assistant opener must appear exactly once and at the end.
    const std::string suffix = "<|im_start|>assistant\n";
    ASSERT_GE(text.size(), suffix.size());
    EXPECT_EQ(text.compare(text.size() - suffix.size(), suffix.size(), suffix), 0)
        << "text did not end with assistant prompt: " << text;
}

TEST(Qwen2VLProcessor, ApplyChatTemplateHandlesMultipleMessages) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System,    "you are helpful", {}},
        {geniex::Role::User,      "hello",           {}},
        {geniex::Role::Assistant, "hi there",        {}},
    };
    auto text = p->apply_chat_template(msgs, /*add_generation_prompt=*/false);

    EXPECT_NE(text.find("<|im_start|>system"),    std::string::npos);
    EXPECT_NE(text.find("<|im_start|>user"),      std::string::npos);
    EXPECT_NE(text.find("<|im_start|>assistant"), std::string::npos);
    EXPECT_NE(text.find("you are helpful"),       std::string::npos);
    EXPECT_NE(text.find("hi there"),              std::string::npos);
}

TEST(Qwen2VLProcessor, ApplyChatTemplateInsertsOneMarkerPerMMContent) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "describe these",
         {
             {geniex::Modality::Image, "a.png"},
             {geniex::Modality::Image, "b.png"},
         }},
    };
    auto text = p->apply_chat_template(msgs, /*add_generation_prompt=*/true);

    // Count occurrences of the default marker.
    const std::string marker = geniex::kDefaultImageMarker;
    size_t count = 0;
    for (size_t pos = 0;
         (pos = text.find(marker, pos)) != std::string::npos;
         pos += marker.size()) {
        ++count;
    }
    EXPECT_EQ(count, 2u) << text;
}

TEST(Qwen2VLProcessor, ApplyChatTemplateRejectsLiteralMarkerInContent) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    // Content that would confuse positional marker→image pairing during process().
    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User,
         std::string("sneaky ") + geniex::kDefaultImageMarker,
         {}},
    };
    EXPECT_THROW(p->apply_chat_template(msgs, /*add_generation_prompt=*/true),
                 std::runtime_error);
}

// ─── process() — end-to-end on a generated image ─────────────────────────────

namespace {

// Create a tiny solid-colour PNG in `dir` and return its path. The image is
// 64x64 so smart_resize bumps it up to at least the default min_pixels
// (4*28*28 = 3136 pixels with factor=56 \u2192 one merge unit).
fs::path write_test_png(const fs::path& dir) {
    constexpr int W = 64, H = 64;
    std::vector<uint8_t> rgb(W * H * 3, 128);
    // Add a diagonal gradient so it's not entirely uniform.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            rgb[(y * W + x) * 3 + 0] = static_cast<uint8_t>(x * 4);  // R
            rgb[(y * W + x) * 3 + 1] = static_cast<uint8_t>(y * 4);  // G
            rgb[(y * W + x) * 3 + 2] = 200;                          // B
        }
    }
    const fs::path path = dir / "proc_test_image.png";
    const int ok = stbi_write_png(path.string().c_str(), W, H, 3,
                                  rgb.data(), W * 3);
    if (ok == 0) {
        throw std::runtime_error("stbi_write_png failed for " + path.string());
    }
    return path;
}

}  // namespace

TEST(Qwen2VLProcessor, ProcessPopulatesAllBatchFields) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    // Stage a generated image under the test binary's working dir.
    const fs::path image = write_test_png(fs::temp_directory_path());

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "describe",
         {{geniex::Modality::Image, image.string()}}},
    };
    const auto formatted = p->apply_chat_template(msgs, /*add_generation_prompt=*/true);

    auto features = p->process(formatted, {image.string()});

    EXPECT_EQ(features.text, formatted);
    EXPECT_FALSE(features.input_ids.empty());
    // pixel_values should be [n_patches, C*T*P*P]; with our 64x64 input and
    // default config (factor=28*2=56, min_pixels=3136), smart_resize yields a
    // non-empty patch tensor.
    ASSERT_EQ(features.pixel_values.shape().size(), 2u);
    EXPECT_GT(features.pixel_values.shape()[0], 0u);
    EXPECT_GT(features.pixel_values.shape()[1], 0u);
    // image_grid_thw shape [1, 3]
    ASSERT_EQ(features.image_grid_thw.shape().size(), 2u);
    EXPECT_EQ(features.image_grid_thw.shape()[0], 1u);
    EXPECT_EQ(features.image_grid_thw.shape()[1], 3u);

    std::remove(image.string().c_str());
}

TEST(Qwen2VLProcessor, ProcessThrowsWhenMarkerCountMismatchesImages) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    const fs::path image = write_test_png(fs::temp_directory_path());

    // Formatted text has zero markers but we pass one image → must throw.
    EXPECT_THROW(p->process("no marker here", {image.string()}),
                 std::runtime_error);

    std::remove(image.string().c_str());
}

TEST(Qwen2VLProcessor, ProcessTextOnlyProducesNonEmptyIdsAndEmptyPixels) {
    auto p = make_processor();
    if (!p) GTEST_SKIP() << "Tokenizer fixture not present";

    std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "hello world", {}},
    };
    const auto formatted = p->apply_chat_template(msgs, /*add_generation_prompt=*/true);

    auto features = p->process(formatted, /*image_paths=*/{});

    EXPECT_EQ(features.text, formatted);
    EXPECT_FALSE(features.input_ids.empty());
    // With zero images, pixel_values and image_grid_thw remain default-empty.
    EXPECT_EQ(features.pixel_values.shape().size(),   0u);
    EXPECT_EQ(features.image_grid_thw.shape().size(), 0u);
}
