// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "geniex-proc/qwen2vl.h"

namespace fs = std::filesystem;
namespace {

class ScopedPreprocessDeviceEnv {
   public:
    explicit ScopedPreprocessDeviceEnv(const char* value) {
        const char* previous = std::getenv("GENIEX_QAIRT_PREPROCESS_DEVICE");
        had_previous_ = previous != nullptr;
        if (previous) previous_ = previous;
        set(value);
    }

    ~ScopedPreprocessDeviceEnv() { set(had_previous_ ? previous_.c_str() : nullptr); }

   private:
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("GENIEX_QAIRT_PREPROCESS_DEVICE", value ? value : "");
#else
        if (value)
            setenv("GENIEX_QAIRT_PREPROCESS_DEVICE", value, 1);
        else
            unsetenv("GENIEX_QAIRT_PREPROCESS_DEVICE");
#endif
    }

    bool had_previous_ = false;
    std::string previous_;
};

fs::path tokenizer_path() { return fs::path(GENIEXPROC_TEST_TOKENIZER_PATH); }

bool has_tokenizer_fixture() { return fs::exists(tokenizer_path()); }

class TestImages {
   public:
    TestImages() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        dir_ = fs::temp_directory_path() / ("geniex-qwen2vl-device-" + std::to_string(nonce));
        fs::create_directories(dir_);
    }

    ~TestImages() { fs::remove_all(dir_); }

    fs::path write(const std::string& name, int width, int height, int seed) const {
        std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
                rgb[offset] = static_cast<uint8_t>((x * 17 + y * 3 + seed * 29) % 256);
                rgb[offset + 1] = static_cast<uint8_t>((x * 5 + y * 19 + seed * 47) % 256);
                rgb[offset + 2] = static_cast<uint8_t>((x * 11 + y * 7 + seed * 61) % 256);
            }
        }
        const fs::path path = dir_ / name;
        if (!stbi_write_png(path.string().c_str(), width, height, 3, rgb.data(), width * 3)) {
            throw std::runtime_error("stbi_write_png failed for " + path.string());
        }
        return path;
    }

   private:
    fs::path dir_;
};

std::unique_ptr<geniex::qwen2vl::Qwen2VLProcessor> make_processor(geniex::qwen2vl::Qwen2VLConfig config) {
    return geniex::qwen2vl::Qwen2VLProcessor::create(tokenizer_path().string(), config);
}

geniex::BatchFeatures process_images(geniex::qwen2vl::Qwen2VLProcessor& processor, const std::vector<fs::path>& paths) {
    geniex::ChatMessage message{geniex::Role::User, "Compare these images", {}};
    std::vector<std::string> image_paths;
    for (const auto& path : paths) {
        message.mm_content.push_back({geniex::Modality::Image, path.string()});
        image_paths.push_back(path.string());
    }
    const std::string text = processor.apply_chat_template({message});
    return processor.process(text, image_paths);
}

void expect_same_features(const geniex::BatchFeatures& expected, const geniex::BatchFeatures& actual,
                          float pixel_tolerance) {
    EXPECT_EQ(actual.text, expected.text);
    EXPECT_EQ(actual.input_ids, expected.input_ids);

    ASSERT_EQ(actual.pixel_values.dimension(), expected.pixel_values.dimension());
    ASSERT_EQ(actual.image_grid_thw.dimension(), expected.image_grid_thw.dimension());
    for (size_t axis = 0; axis < actual.pixel_values.dimension(); ++axis) {
        EXPECT_EQ(actual.pixel_values.shape(axis), expected.pixel_values.shape(axis)) << "pixel axis " << axis;
    }
    for (size_t axis = 0; axis < actual.image_grid_thw.dimension(); ++axis) {
        EXPECT_EQ(actual.image_grid_thw.shape(axis), expected.image_grid_thw.shape(axis)) << "grid axis " << axis;
    }
    ASSERT_EQ(actual.pixel_values.size(), expected.pixel_values.size());
    ASSERT_EQ(actual.image_grid_thw.size(), expected.image_grid_thw.size());
    for (size_t i = 0; i < actual.image_grid_thw.size(); ++i) {
        EXPECT_EQ(actual.image_grid_thw.data()[i], expected.image_grid_thw.data()[i]) << "grid value " << i;
    }

    float max_difference = 0.0f;
    size_t max_index = 0;
    for (size_t i = 0; i < actual.pixel_values.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual.pixel_values.data()[i])) << "non-finite device value at " << i;
        const float difference = std::abs(actual.pixel_values.data()[i] - expected.pixel_values.data()[i]);
        if (difference > max_difference) {
            max_difference = difference;
            max_index = i;
        }
    }
    EXPECT_LE(max_difference, pixel_tolerance)
        << "maximum normalized pixel difference " << max_difference << " at element " << max_index
        << " exceeds tolerance " << pixel_tolerance;
}

void compare_hardware_backend(geniex::PreprocessDevice device, int patch_size = 14, int merge_size = 2,
                              int temporal_patch_size = 2, int height = 56, int width = 56) {
    // Deliberately distinct from InternVL's default ImageNet normalization.
    // A backend that accidentally hard-codes InternVL's scale/mean/std will
    // produce a large patch-value difference even if its resize is correct.
    geniex::qwen2vl::Qwen2VLConfig cpu_config;
    cpu_config.fixed_height = height;
    cpu_config.fixed_width = width;
    cpu_config.patch_size = patch_size;
    cpu_config.merge_size = merge_size;
    cpu_config.temporal_patch_size = temporal_patch_size;
    cpu_config.rescale_factor = 0.37f;
    cpu_config.image_mean = {0.17f, -0.08f, 0.63f};
    cpu_config.image_std = {0.41f, 0.73f, 1.17f};
    cpu_config.preprocess_device = geniex::PreprocessDevice::CPU;

    auto device_config = cpu_config;
    device_config.preprocess_device = device;
    auto cpu = make_processor(cpu_config);
    auto accelerated = make_processor(device_config);

    TestImages images;
    const auto small = images.write("upsample.png", 28, 42, 1);
    const auto large = images.write("downsample.png", 112, 84, 2);
    const auto same = images.write("same-size.png", 56, 56, 3);
    const auto irregular = images.write("irregular.png", 37, 53, 4);

    // Both backends round resize output to uint8; allow one code value.
    const float tolerance = std::abs(cpu_config.rescale_factor) /
                                *std::min_element(cpu_config.image_std.begin(), cpu_config.image_std.end()) +
                            0.00002f;
    const std::array<std::pair<const char*, std::vector<fs::path>>, 5> cases{{
        {"upsampling", {small}},
        {"downsampling", {large}},
        {"unchanged dimensions", {same}},
        {"irregular scale", {irregular}},
        {"multiple images", {small, large, same}},
    }};
    for (const auto& [name, paths] : cases) {
        SCOPED_TRACE(name);
        const auto cpu_features = process_images(*cpu, paths);
        const auto device_features = process_images(*accelerated, paths);
        expect_same_features(cpu_features, device_features, tolerance);
        ASSERT_EQ(device_features.pixel_values.dimension(), 2u);
        ASSERT_EQ(device_features.image_grid_thw.dimension(), 2u);
        const size_t grid_height = static_cast<size_t>(height / patch_size);
        const size_t grid_width = static_cast<size_t>(width / patch_size);
        EXPECT_EQ(device_features.pixel_values.shape(0), paths.size() * grid_height * grid_width);
        EXPECT_EQ(device_features.pixel_values.shape(1),
                  3u * static_cast<size_t>(temporal_patch_size) * patch_size * patch_size);
        ASSERT_EQ(device_features.image_grid_thw.shape(0), paths.size());
        ASSERT_EQ(device_features.image_grid_thw.shape(1), 3u);
        for (size_t image = 0; image < paths.size(); ++image) {
            EXPECT_EQ(device_features.image_grid_thw(image, 0), 1u);
            EXPECT_EQ(device_features.image_grid_thw(image, 1), grid_height);
            EXPECT_EQ(device_features.image_grid_thw(image, 2), grid_width);
        }
    }
}

}  // namespace

TEST(Qwen2VLDevice, DefaultCpuMatchesExplicitCpuEndToEnd) {
    ScopedPreprocessDeviceEnv env(nullptr);
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();

    geniex::qwen2vl::Qwen2VLConfig default_config;
    EXPECT_EQ(default_config.preprocess_device, geniex::PreprocessDevice::Auto);
    auto explicit_config = default_config;
    explicit_config.preprocess_device = geniex::PreprocessDevice::CPU;
    auto default_processor = make_processor(default_config);
    auto explicit_processor = make_processor(explicit_config);

    TestImages images;
    const auto image = images.write("default-cpu.png", 64, 48, 5);
    const auto default_features = process_images(*default_processor, {image});
    const auto explicit_features = process_images(*explicit_processor, {image});
    expect_same_features(default_features, explicit_features, 0.0f);
}

TEST(Qwen2VLDevice, EnvironmentSelectsDeviceAndExplicitConfigWins) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    geniex::qwen2vl::Qwen2VLConfig config;
    {
        ScopedPreprocessDeviceEnv env("not-a-device");
        EXPECT_THROW(make_processor(config), std::invalid_argument);
        config.preprocess_device = geniex::PreprocessDevice::CPU;
        EXPECT_NO_THROW(make_processor(config));
    }
    config.preprocess_device = geniex::PreprocessDevice::Auto;
    {
        ScopedPreprocessDeviceEnv env("GpU");
#ifndef GENIEXPROC_TEST_ENABLE_OPENCL
        EXPECT_THROW(make_processor(config), std::runtime_error);
#else
        EXPECT_NO_THROW(make_processor(config));
#endif
    }
    {
        ScopedPreprocessDeviceEnv env("NpU");
#ifndef GENIEXPROC_TEST_ENABLE_HTP
        EXPECT_THROW(make_processor(config), std::runtime_error);
#else
        EXPECT_NO_THROW(make_processor(config));
#endif
    }
}

#if !defined(GENIEXPROC_TEST_ENABLE_OPENCL) && !defined(GENIEXPROC_TEST_ENABLE_HTP)
TEST(Qwen2VLDevice, CpuOnlyBuildRejectsGpuAndNpuClearly) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    for (const auto& [device, message] :
         std::array{std::pair{geniex::PreprocessDevice::GPU, "requires GENIEXPROC_ENABLE_OPENCL"},
                    std::pair{geniex::PreprocessDevice::NPU, "requires GENIEXPROC_ENABLE_HTP"}}) {
        auto config = geniex::qwen2vl::Qwen2VLConfig{};
        config.preprocess_device = device;
        try {
            (void)make_processor(config);
            FAIL() << "Unsupported device unexpectedly constructed";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
        }
    }
}
#endif

#if defined(GENIEXPROC_TEST_ENABLE_OPENCL)
TEST(Qwen2VLDevice, OpenClMatchesCpuEndToEnd) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    compare_hardware_backend(geniex::PreprocessDevice::GPU);
}

TEST(Qwen2VLDevice, OpenClMatchesCpuWithNondefaultTemporalAndMerge) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    compare_hardware_backend(geniex::PreprocessDevice::GPU, 7, 3, 3, 63, 84);
}
#else
TEST(Qwen2VLDevice, OpenClMatchesCpuEndToEnd) { GTEST_SKIP() << "GENIEXPROC_ENABLE_OPENCL is disabled in this build"; }
TEST(Qwen2VLDevice, OpenClMatchesCpuWithNondefaultTemporalAndMerge) {
    GTEST_SKIP() << "GENIEXPROC_ENABLE_OPENCL is disabled in this build";
}
#endif

#if defined(GENIEXPROC_TEST_ENABLE_HTP)
TEST(Qwen2VLDevice, HtpMatchesCpuEndToEnd) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    compare_hardware_backend(geniex::PreprocessDevice::NPU);
}

TEST(Qwen2VLDevice, HtpMatchesCpuWithNondefaultTemporalAndMerge) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    compare_hardware_backend(geniex::PreprocessDevice::NPU, 7, 3, 3, 63, 84);
}

TEST(Qwen2VLDevice, HtpMatchesCpuWithSmallPatches) {
    if (!has_tokenizer_fixture()) GTEST_SKIP() << "Tokenizer fixture not present: " << tokenizer_path();
    compare_hardware_backend(geniex::PreprocessDevice::NPU, 14, 2, 2, 28, 28);
}
#else
TEST(Qwen2VLDevice, HtpMatchesCpuEndToEnd) { GTEST_SKIP() << "GENIEXPROC_ENABLE_HTP is disabled in this build"; }
TEST(Qwen2VLDevice, HtpMatchesCpuWithNondefaultTemporalAndMerge) {
    GTEST_SKIP() << "GENIEXPROC_ENABLE_HTP is disabled in this build";
}
TEST(Qwen2VLDevice, HtpMatchesCpuWithSmallPatches) {
    GTEST_SKIP() << "GENIEXPROC_ENABLE_HTP is disabled in this build";
}
#endif
