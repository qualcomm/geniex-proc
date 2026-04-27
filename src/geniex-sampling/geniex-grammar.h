// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp)
// Original work Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License (https://opensource.org/licenses/MIT)

#pragma once

#include <map>
#include <regex>
#include <string>
#include <vector>

#include "geniex-sampling.h"

// Forward declarations
struct geniex_vocab_interface;

// Grammar element type
enum geniex_gretype {
    // end of rule definition
    GENIEX_GRETYPE_END = 0,
    // start of alternate definition for rule
    GENIEX_GRETYPE_ALT = 1,
    // non-terminal element: reference to rule
    GENIEX_GRETYPE_RULE_REF = 2,
    // terminal element: character (code point)
    GENIEX_GRETYPE_CHAR = 3,
    // inverse char(s) ([^a], [^a-b] [^abc])
    GENIEX_GRETYPE_CHAR_NOT = 4,
    // modifies a preceding GENIEX_GRETYPE_CHAR or GENIEX_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    GENIEX_GRETYPE_CHAR_RNG_UPPER = 5,
    // modifies a preceding GENIEX_GRETYPE_CHAR or
    // GENIEX_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    GENIEX_GRETYPE_CHAR_ALT = 6,
    // any character (.)
    GENIEX_GRETYPE_CHAR_ANY = 7,
};

typedef struct geniex_grammar_element {
    enum geniex_gretype type;
    uint32_t value;  // Unicode code point or rule ID
} geniex_grammar_element;

struct geniex_partial_utf8 {
    uint32_t value;  // bit value so far (unshifted)
    int n_remain;    // num bytes remaining; -1 indicates invalid sequence
};

struct geniex_grammar_candidate {
    size_t index;
    const uint32_t *code_points;
    geniex_partial_utf8 partial_utf8;
};

using geniex_grammar_rule = std::vector<geniex_grammar_element>;
using geniex_grammar_stack = std::vector<const geniex_grammar_element *>;

using geniex_grammar_rules = std::vector<geniex_grammar_rule>;
using geniex_grammar_stacks = std::vector<geniex_grammar_stack>;
using geniex_grammar_candidates = std::vector<geniex_grammar_candidate>;

// Grammar parser
struct geniex_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;
    geniex_grammar_rules rules;

    geniex_grammar_stack c_rules() const;
    uint32_t get_symbol_id(const char *src, size_t len);
    uint32_t generate_symbol_id(const std::string &base_name);
    void add_rule(uint32_t rule_id, const geniex_grammar_rule &rule);

    const char *parse_alternates(const char *src, const std::string &rule_name, uint32_t rule_id, bool is_nested);
    const char *parse_sequence(const char *src, const std::string &rule_name, geniex_grammar_rule &rule, bool is_nested);
    const char *parse_rule(const char *src);

    bool parse(const char *src);
    void print(FILE *file);
};

// Grammar trigger pattern for lazy grammars
struct geniex_grammar_trigger_pattern {
    std::string pattern;
    std::regex regex;
};

// Main grammar struct
struct geniex_grammar {
    geniex_vocab_interface *vocab;

    const geniex_grammar_rules rules;
    geniex_grammar_stacks stacks;

    // Buffer for partially generated UTF-8 sequence from accepted tokens
    geniex_partial_utf8 partial_utf8;

    // Lazy grammar support
    bool lazy;
    bool awaiting_trigger;
    std::string trigger_buffer;
    std::vector<geniex_token> trigger_tokens;
    std::vector<geniex_grammar_trigger_pattern> trigger_patterns;
};

// Main grammar API
struct geniex_grammar *geniex_grammar_init(geniex_vocab_interface *vocab, const geniex_grammar_element **rules, size_t n_rules,
                                       size_t start_rule_index);

struct geniex_grammar *geniex_grammar_init_from_string(geniex_vocab_interface *vocab, const char *grammar_str,
                                                   const char *grammar_root, bool lazy = false,
                                                   const char **trigger_patterns = nullptr,
                                                   size_t num_trigger_patterns = 0,
                                                   const geniex_token *trigger_tokens = nullptr,
                                                   size_t num_trigger_tokens = 0);

void geniex_grammar_free(struct geniex_grammar *grammar);

struct geniex_grammar *geniex_grammar_clone(const struct geniex_grammar &grammar);

// Grammar application functions
void geniex_grammar_apply(const struct geniex_grammar &grammar, geniex_token_data_array *cur_p);

void geniex_grammar_accept(struct geniex_grammar &grammar, geniex_token token);

void geniex_grammar_accept_str(struct geniex_grammar &grammar, const std::string &piece);

// Helper functions
void geniex_grammar_accept_impl(struct geniex_grammar &grammar, uint32_t chr);

geniex_grammar_candidates geniex_grammar_reject_candidates_for_stack(const geniex_grammar_rules &rules,
                                                                 const geniex_grammar_stack &stack,
                                                                 const geniex_grammar_candidates &candidates);

// Internal helper functions
std::vector<geniex_grammar_candidate> geniex_grammar_reject_candidates(const geniex_grammar_rules &rules,
                                                                   const geniex_grammar_stacks &stacks,
                                                                   const geniex_grammar_candidates &candidates);

void geniex_grammar_advance_stack(const geniex_grammar_rules &rules, const geniex_grammar_stack &stack,
                                geniex_grammar_stacks &new_stacks);

// UTF-8 decoding helpers
std::pair<uint32_t, const char *> geniex_decode_utf8(const char *src);

std::pair<std::vector<uint32_t>, geniex_partial_utf8> geniex_decode_utf8(const std::string &src,
                                                                     geniex_partial_utf8 partial_start);