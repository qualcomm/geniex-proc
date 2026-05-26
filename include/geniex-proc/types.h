// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// geniex-proc/types.h
//
// All shared types used across geniex-proc: token types, sampling parameters,
// conversation types, and utility macros.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include "geniex-proc/export.h"

// ============================================================
// Error / debug macros — logging will be revisited in a future refactor.
// ============================================================

#define GENIEXPROC_THROW(msg) \
    throw std::runtime_error(std::string(__FILE__) + ':' + std::to_string(__LINE__) + ' ' + (msg))
#define GENIEXPROC_CHECK(cond) \
    do { if (!(cond)) GENIEXPROC_THROW("check failed (" #cond ")"); } while(0)

// ============================================================
// Core token type
// ============================================================

#define GENIEX_DEFAULT_SEED UINT32_MAX
#define GENIEX_TOKEN_NULL   -1

typedef int32_t geniex_token;

// ============================================================
// Sampler types
// ============================================================

enum GENIEXPROC_API sampler_type {
    SAMPLER_TYPE_NONE        = 0,
    SAMPLER_TYPE_DRY         = 1,
    SAMPLER_TYPE_TOP_K       = 2,
    SAMPLER_TYPE_TOP_P       = 3,
    SAMPLER_TYPE_MIN_P       = 4,
    SAMPLER_TYPE_TYPICAL_P   = 6,
    SAMPLER_TYPE_TEMPERATURE = 7,
    SAMPLER_TYPE_XTC         = 8,
    SAMPLER_TYPE_INFILL      = 9,
    SAMPLER_TYPE_PENALTIES   = 10,
    SAMPLER_TYPE_TOP_N_SIGMA = 11,
};

// ============================================================
// Sampling parameters
// ============================================================

struct GENIEXPROC_API geniex_sampler_params {
    uint32_t seed = GENIEX_DEFAULT_SEED;

    int32_t n_prev    = 64;      // number of previous tokens to remember
    int32_t top_k     = 40;      // <= 0 to use vocab size
    float   top_p     = 0.95f;   // 1.0 = disabled
    float   min_p     = 0.05f;   // 0.0 = disabled
    float   temp      = 0.80f;   // <= 0.0 to sample greedily
    float   typical_p = 1.00f;   // 1.0 = disabled
    size_t  min_keep  = 0;       // minimum number of tokens to keep

    // Repetition penalties
    int32_t penalty_last_n    = 64;     // last n tokens to penalize (0 = disable, -1 = context size)
    float   penalty_repeat    = 1.00f;  // 1.0 = disabled
    float   penalty_freq      = 0.00f;  // 0.0 = disabled
    float   penalty_present   = 0.00f;  // 0.0 = disabled

    // DRY sampling
    float       dry_multiplier      = 0.0f;
    float       dry_base            = 1.75f;
    int32_t     dry_allowed_length  = 2;
    int32_t     dry_penalty_last_n  = -1;  // -1 = context size
    // Pre-tokenized sequence breakers — caller tokenizes strings once before constructing Sampler.
    // Empty = DRY disabled even if dry_multiplier > 0.
    std::vector<std::vector<geniex_token>> dry_sequence_breaker_tokens = {};

    // XTC sampling
    float xtc_probability = 0.00f;  // 0.0 = disabled
    float xtc_threshold   = 0.10f;

    // Mirostat
    int32_t mirostat     = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau = 5.00f;
    float   mirostat_eta = 0.10f;

    // Extended temperature
    float dynatemp_range    = 0.00f;  // 0.0 = disabled
    float dynatemp_exponent = 1.00f;

    // Top-N-Sigma sampling
    float top_n_sigma = -1.00f;  // -1.0 = disabled

    // EOG (end-of-generation) token IDs — caller populates from tokenizer->token_eos() etc.
    // Used by Sampler::is_eog(). Empty = is_eog() always returns false.
    std::vector<geniex_token> eog_tokens = {};

    // Grammar — string form; attach a Grammar object to Sampler for enforcement
    std::string grammar_str;
    std::string grammar_root = "root";

    // Performance
    bool no_perf = false;

    // Logit bias
    std::vector<std::pair<geniex_token, float>> logit_bias;

    // Sampler chain order
    std::vector<enum sampler_type> samplers = {
        SAMPLER_TYPE_PENALTIES, SAMPLER_TYPE_DRY,       SAMPLER_TYPE_TOP_N_SIGMA,
        SAMPLER_TYPE_TOP_K,     SAMPLER_TYPE_TYPICAL_P, SAMPLER_TYPE_TOP_P,
        SAMPLER_TYPE_MIN_P,     SAMPLER_TYPE_XTC,       SAMPLER_TYPE_TEMPERATURE,
    };

    std::string print() const;
};

// ============================================================
// Conversation types
// ============================================================

namespace geniex {

/// Role of a conversation participant.
enum class Role {
    System,
    User,
    Assistant,
    Tool,
};

/// Returns the string representation of a Role
inline const char* role_to_string(Role role) {
    switch (role) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
        case Role::Tool:      return "tool";
    }
    return "unknown";
}

/// Type of a multimodal content attachment.
enum class Modality {
    Image,
    Audio,
    Video,
};

/// A single multimodal content entry: a typed file path.
struct MMContent {
    Modality    modality;
    std::string path;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    Role        role    = Role::User;
    std::string content;

    // Multimodal attachments. Consumed by VLM/Omni Processors;
    // Tokenizer::apply_chat_template() ignores them.
    std::vector<MMContent> mm_content;

    // Tool calls issued on a prior `Assistant` turn.
    std::vector<ToolCall> tool_calls;

    // Links a `Tool` response back to `tool_calls[i].id`. Required by
    // some templates (Mistral, Cohere, GPT-OSS).
    std::string tool_call_id;

    // Function name on `Tool` responses. Required by some templates
    // (Llama 3.1, EXAONE).
    std::string name;

    // Separate reasoning stream on `Assistant` turns, so reasoning
    // models (Qwen3, DeepSeek-R1, Gemma 4, EXAONE 4) see their own prior
    // chain-of-thought without it bleeding into `content`.
    std::string reasoning_content;
};

// ============================================================
// Multimodal batch output
// ============================================================

/// Output of a model Processor's process() call.
/// Phase 2 (Vision): contains text, token IDs, image pixel values, and grid metadata.
/// Future phases will add audio_features, video_grid_thw, etc.
struct GENIEXPROC_API BatchFeatures {
    /// Formatted prompt string (after chat template applied).
    std::string text;

    /// Token IDs for the full prompt (including image placeholder tokens).
    std::vector<int32_t> input_ids;

    /// Flattened image patches. Shape: [total_patches, channels * temporal_patch_size * patch_size * patch_size].
    /// Empty if no images were provided.
    xt::xarray<float> pixel_values;

    /// Per-image grid dimensions (T, H, W). Shape: [n_images, 3].
    /// Empty if no images were provided.
    xt::xarray<size_t> image_grid_thw;
};

// ============================================================
// Utilities
// ============================================================

template <typename T>
void print_vector(const std::vector<T> &vec, const std::string &name = "vector") {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << vec[i];
    }
    std::cout << "]" << std::endl;
}

} // namespace geniex
