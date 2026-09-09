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
#include "geniex-proc/gemma4.h"
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
    auto text = p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/false});

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
    auto text = p->apply_chat_template(msgs);

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
    auto text = p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/false});

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
    auto text = p->apply_chat_template(msgs);

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
    EXPECT_THROW(p->apply_chat_template(msgs), std::runtime_error);
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
    const auto formatted = p->apply_chat_template(msgs);

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
    const auto formatted = p->apply_chat_template(msgs);

    auto features = p->process(formatted, /*image_paths=*/{});

    EXPECT_EQ(features.text, formatted);
    EXPECT_FALSE(features.input_ids.empty());
    // With zero images, pixel_values and image_grid_thw remain default-empty.
    EXPECT_EQ(features.pixel_values.shape().size(),   0u);
    EXPECT_EQ(features.image_grid_thw.shape().size(), 0u);
}

// ─── Gemma4Processor::apply_chat_template ────────────────────────────────────
//
// Gemma4's framing is hand-rolled (as Qwen2VL's and InternVL's are), so these
// assert the exact rendered string. The expectations were captured from the
// bundled Jinja template it replaced, against the E4B bundle, so a drift here is
// a drift away from upstream Gemma formatting.
//
// No tokenizer fixture is needed: the builder works off the message list alone.

namespace {

std::unique_ptr<geniex::gemma4::Gemma4Processor> make_gemma4_processor() {
    return geniex::gemma4::Gemma4Processor::create(
        /*tokenizer_path=*/"", /*tokenizer_config_path=*/"", geniex::gemma4::Gemma4Config{});
}

}  // namespace

TEST(Gemma4Processor, ApplyChatTemplateUserOnly) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    const std::vector<geniex::ChatMessage> msgs = {{geniex::Role::User, "hello", {}}};
    EXPECT_EQ(p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true}),
              "<bos><|turn>user\nhello<turn|>\n<|turn>model\n");
    EXPECT_EQ(p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/false}),
              "<bos><|turn>user\nhello<turn|>\n");
}

// BOS is emitted exactly once, before the first turn — not per turn as in
// ChatML. A caller re-rendering a transcript would therefore repeat it.
TEST(Gemma4Processor, ApplyChatTemplateEmitsBosOnlyOnce) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "q1", {}},
        {geniex::Role::Assistant, "a1", {}},
        {geniex::Role::User, "q2", {}},
    };
    const auto text = p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true});
    EXPECT_EQ(text,
              "<bos><|turn>user\nq1<turn|>\n<|turn>model\na1<turn|>\n<|turn>user\nq2<turn|>\n<|turn>model\n");

    size_t bos_count = 0;
    for (size_t i = text.find("<bos>"); i != std::string::npos; i = text.find("<bos>", i + 1)) ++bos_count;
    EXPECT_EQ(bos_count, 1u);
}

// Gemma names the assistant role "model".
TEST(Gemma4Processor, ApplyChatTemplateRendersAssistantAsModel) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    const std::vector<geniex::ChatMessage> msgs = {{geniex::Role::Assistant, "hi", {}}};
    const auto text = p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/false});
    EXPECT_EQ(text, "<bos><|turn>model\nhi<turn|>\n");
    EXPECT_EQ(text.find("assistant"), std::string::npos) << text;
}

TEST(Gemma4Processor, ApplyChatTemplateSystemTurn) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "You are helpful.", {}},
        {geniex::Role::User, "hello", {}},
    };
    EXPECT_EQ(p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true}),
              "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n");
}

// One marker per attachment, ahead of the text, so process() pairs the i-th
// marker with image_paths[i].
TEST(Gemma4Processor, ApplyChatTemplateEmitsOneMarkerPerImage) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    geniex::ChatMessage two{geniex::Role::User, "compare them", {}};
    two.mm_content.push_back({geniex::Modality::Image, "a.jpg"});
    two.mm_content.push_back({geniex::Modality::Image, "b.jpg"});

    EXPECT_EQ(p->apply_chat_template({two}, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true}),
              "<bos><|turn>user\n<__image__><__image__>compare them<turn|>\n<|turn>model\n");
}

// Message bodies are trimmed at the EDGES ONLY, matching the Jinja's
// `{{- message['content'] | trim -}}`. Skipping it would render a different
// prompt than upstream Gemma for any body with surrounding whitespace.
//
// Each expectation below was verified against the bundle's real Jinja template
// (rendered through Tokenizer::apply_chat_template on an E4B bundle) and is
// byte-identical to it.
TEST(Gemma4Processor, ApplyChatTemplateTrimsContentEdges) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    auto render = [&p](const std::string& body) {
        return p->apply_chat_template(
            {{geniex::Role::User, body, {}}}, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true});
    };

    // Leading and trailing whitespace goes, including newlines and tabs.
    EXPECT_EQ(render("  hello  \n"), "<bos><|turn>user\nhello<turn|>\n<|turn>model\n");
    EXPECT_EQ(render("\n\n  hi  \t\n"), "<bos><|turn>user\nhi<turn|>\n<|turn>model\n");
    EXPECT_EQ(render("\tlead and trail tab\t"), "<bos><|turn>user\nlead and trail tab<turn|>\n<|turn>model\n");

    // A body that is nothing but whitespace collapses to an empty turn.
    EXPECT_EQ(render("   "), "<bos><|turn>user\n<turn|>\n<|turn>model\n");
}

// Interior whitespace is NOT touched -- only the edges are. Kept separate from
// the case above because this is the half that is easy to get wrong: a builder
// that normalised or collapsed runs would still pass the edge cases.
TEST(Gemma4Processor, ApplyChatTemplatePreservesInteriorWhitespace) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    auto render = [&p](const std::string& body) {
        return p->apply_chat_template(
            {{geniex::Role::User, body, {}}}, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true});
    };

    // The edges are stripped but the single interior space survives.
    EXPECT_EQ(render("  hello world  \n"), "<bos><|turn>user\nhello world<turn|>\n<|turn>model\n");

    // Nothing to strip, and an interior run is passed through verbatim.
    EXPECT_EQ(render("hello   world"), "<bos><|turn>user\nhello   world<turn|>\n<|turn>model\n");

    // Interior newlines survive too, so multi-line bodies keep their shape.
    EXPECT_EQ(render("a\n\nb"), "<bos><|turn>user\na\n\nb<turn|>\n<|turn>model\n");
}

// A literal marker in user content would shift positional marker-to-image
// pairing in process(), so it is rejected rather than silently mispaired.
TEST(Gemma4Processor, ApplyChatTemplateRejectsLiteralMarkerInContent) {
    auto p = make_gemma4_processor();
    ASSERT_TRUE(p != nullptr);

    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, std::string("look ") + geniex::kDefaultImageMarker, {}},
    };
    EXPECT_THROW(
        p->apply_chat_template(msgs, geniex::ApplyChatTemplateOptions{/*add_generation_prompt=*/true}),
        std::runtime_error);
}
