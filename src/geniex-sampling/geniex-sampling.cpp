// Adapted from llama.cpp (https://github.com/ggml-org/llama.cpp)
// Original work Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License (https://opensource.org/licenses/MIT)

#include "geniex-sampling.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "geniex-grammar.h"
#include "sampling-vocab.h"

// Simple time measurement utility
struct time_meas {
    std::chrono::high_resolution_clock::time_point t_start;
    int64_t &t_us;
    bool no_perf;

    time_meas(int64_t &t_us, bool no_perf = false) : t_us(t_us), no_perf(no_perf) {
        if (!no_perf) {
            t_start = std::chrono::high_resolution_clock::now();
        }
    }

    ~time_meas() {
        if (!no_perf) {
            auto t_end = std::chrono::high_resolution_clock::now();
            t_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        }
    }
};

// Ring buffer implementation from llama-sampling.cpp
template <typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T &front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T &front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T &back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T &back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T &value) {
        if (capacity == 0) {
            throw std::runtime_error("ring buffer: capacity is zero");
        }

        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T &rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const { return sz == 0; }

    size_t size() const { return sz; }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;

    std::vector<T> data;
};

// Core sampler implementation functions

static void geniex_sampler_top_k_impl(geniex_token_data_array *cur_p, int32_t k) {
    if (k <= 0) {
        return;
    }

    k = std::min(k, (int)cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        auto comp = [](const geniex_token_data &a, const geniex_token_data &b) { return a.logit > b.logit; };
        if (k <= 128) {
            std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size, comp);
        } else {
            constexpr int nbuckets = 128;
            constexpr float bucket_low = -10.0f;
            constexpr float bucket_high = 10.0f;
            constexpr float bucket_scale = nbuckets / (bucket_high - bucket_low);
            constexpr float bucket_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(cur_p->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int)cur_p->size; ++i) {
                const float val = cur_p->data[i].logit;
                int ib = int(bucket_scale * val + bucket_inter);
                ib = std::max(0, std::min(nbuckets - 1, ib));
                bucket_idx[i] = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib = nbuckets - 1;
            for (; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) {
                    break;
                }
            }
            std::vector<geniex_token_data> tmp_tokens(nhave);
            auto *ptr = tmp_tokens.data();
            std::vector<geniex_token_data *> bucket_ptrs;
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int)cur_p->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets - 1 - j]++ = cur_p->data[i];
                }
            }

            ptr = tmp_tokens.data();
            int ndone = 0;
            for (int j = nbuckets - 1; j > ib; --j) {
                std::sort(ptr, ptr + histo[j], comp);
                ptr += histo[j];
                ndone += histo[j];
            }
            std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

            std::memcpy(cur_p->data, tmp_tokens.data(), k * sizeof(geniex_token_data));
        }
        cur_p->sorted = true;
    }

    cur_p->size = k;
}

uint32_t get_rng_seed(uint32_t seed) {
    if (seed == GENIEX_DEFAULT_SEED) {
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return (uint32_t)std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

void geniex_sampler_temp_impl(geniex_token_data_array *cur_p, float temp) {
    if (temp <= 0.0f) {
        size_t max_i = 0;
        float max_l = cur_p->data[0].logit;

        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit > max_l) {
                cur_p->data[max_i].logit = -INFINITY;
                max_i = i;
                max_l = cur_p->data[i].logit;
            } else {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}

void geniex_sampler_softmax_impl(geniex_token_data_array *cur_p) {
    assert(cur_p->size > 0);

    // Sort the logits in descending order
    if (!cur_p->sorted) {
        std::sort(cur_p->data, cur_p->data + cur_p->size,
                  [](const geniex_token_data &a, const geniex_token_data &b) { return a.logit > b.logit; });
        cur_p->sorted = true;
    }

    float max_l = cur_p->data[0].logit;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= cum_sum;
    }
}

int geniex_sample_dist(geniex_token_data_array *cur_p, std::mt19937 &rng) {
    struct probs_iterator {
        typedef std::input_iterator_tag iterator_category;
        typedef float value_type;
        typedef float *pointer;
        typedef float &reference;
        typedef ptrdiff_t difference_type;

        const geniex_token_data *data;

        bool operator==(const probs_iterator &other) const { return data == other.data; }
        bool operator!=(const probs_iterator &other) const { return data != other.data; }
        const float &operator*() const { return data->p; }
        probs_iterator &operator++() {
            ++data;
            return *this;
        }
        probs_iterator operator++(int) {
            probs_iterator tmp = *this;
            ++data;
            return tmp;
        }
    };

    std::discrete_distribution<int> dist(probs_iterator{cur_p->data}, probs_iterator{cur_p->data + cur_p->size});
    return dist(rng);
}

// Core sampler API implementation
struct geniex_sampler *geniex_sampler_init(const struct geniex_sampler_i *iface, geniex_sampler_context_t ctx) {
    return new geniex_sampler{
        /* .iface = */ iface,
        /* .ctx   = */ ctx,
    };
}

const char *geniex_sampler_name(const struct geniex_sampler *smpl) {
    if (!smpl->iface) {
        return "(null)";
    }
    return smpl->iface->name(smpl);
}

void geniex_sampler_accept(struct geniex_sampler *smpl, geniex_token token) {
    if (smpl->iface->accept) {
        smpl->iface->accept(smpl, token);
    }
}

void geniex_sampler_apply(struct geniex_sampler *smpl, struct geniex_token_data_array *cur_p) {
    assert(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}

void geniex_sampler_reset(struct geniex_sampler *smpl) {
    if (smpl->iface->reset) {
        smpl->iface->reset(smpl);
    }
}

struct geniex_sampler *geniex_sampler_clone(const struct geniex_sampler *smpl) {
    if (smpl->iface->clone) {
        return smpl->iface->clone(smpl);
    }

    if (smpl->ctx == nullptr) {
        return geniex_sampler_init(smpl->iface, nullptr);
    }

    // Cannot clone without clone implementation
    return nullptr;
}

void geniex_sampler_free(struct geniex_sampler *smpl) {
    if (smpl == nullptr) {
        return;
    }

    if (smpl->iface->free) {
        smpl->iface->free(smpl);
    }

    delete smpl;
}

// Sampler chain implementation
struct geniex_sampler_chain {
    geniex_sampler_chain_params params;
    std::vector<struct geniex_sampler *> samplers;
    mutable int64_t t_sample_us;
    mutable int32_t n_sample;
};

static const char *geniex_sampler_chain_name(const struct geniex_sampler * /*smpl*/) { return "chain"; }

static void geniex_sampler_chain_accept(struct geniex_sampler *smpl, geniex_token token) {
    auto *chain = (geniex_sampler_chain *)smpl->ctx;
    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto *smpl : chain->samplers) {
        geniex_sampler_accept(smpl, token);
    }
    chain->n_sample++;
}

static void geniex_sampler_chain_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *chain = (geniex_sampler_chain *)smpl->ctx;
    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto *smpl : chain->samplers) {
        geniex_sampler_apply(smpl, cur_p);
    }
}

static void geniex_sampler_chain_reset(struct geniex_sampler *smpl) {
    auto *chain = (geniex_sampler_chain *)smpl->ctx;

    for (auto *smpl : chain->samplers) {
        geniex_sampler_reset(smpl);
    }

    chain->t_sample_us = 0;
    chain->n_sample = 0;
}

static struct geniex_sampler *geniex_sampler_chain_clone(const struct geniex_sampler *smpl) {
    const auto *chain_src = (const geniex_sampler_chain *)smpl->ctx;
    auto *result = geniex_sampler_chain_init(chain_src->params);

    for (auto *smpl : chain_src->samplers) {
        geniex_sampler_chain_add(result, geniex_sampler_clone(smpl));
    }

    return result;
}

static void geniex_sampler_chain_free(struct geniex_sampler *smpl) {
    auto *chain = (geniex_sampler_chain *)smpl->ctx;

    for (auto *smpl : chain->samplers) {
        geniex_sampler_free(smpl);
    }

    delete chain;
}

static struct geniex_sampler_i geniex_sampler_chain_i = {
    /* .name   = */ geniex_sampler_chain_name,
    /* .accept = */ geniex_sampler_chain_accept,
    /* .apply  = */ geniex_sampler_chain_apply,
    /* .reset  = */ geniex_sampler_chain_reset,
    /* .clone  = */ geniex_sampler_chain_clone,
    /* .free   = */ geniex_sampler_chain_free,
};

struct geniex_sampler *geniex_sampler_chain_init(struct geniex_sampler_chain_params params) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_chain_i,
        /* .ctx   = */ new geniex_sampler_chain{
            /* .params      = */ params,
            /* .samplers    = */ {},
            /* .t_sample_us = */ 0,
            /* .n_sample    = */ 0,
        });
}

void geniex_sampler_chain_add(struct geniex_sampler *chain, struct geniex_sampler *smpl) {
    auto *p = (geniex_sampler_chain *)chain->ctx;
    p->samplers.push_back(smpl);
}

struct geniex_sampler *geniex_sampler_chain_get(const struct geniex_sampler *chain, int32_t i) {
    const auto *p = (const geniex_sampler_chain *)chain->ctx;

    if (i < 0 || (size_t)i >= p->samplers.size()) {
        return nullptr;
    }

    return p->samplers[i];
}

struct geniex_sampler *geniex_sampler_chain_remove(struct geniex_sampler *chain, int32_t i) {
    auto *p = (geniex_sampler_chain *)chain->ctx;

    if (i < 0 || (size_t)i >= p->samplers.size()) {
        return nullptr;
    }

    auto *result = p->samplers[i];
    p->samplers.erase(p->samplers.begin() + i);

    return result;
}

int geniex_sampler_chain_n(const struct geniex_sampler *chain) {
    const auto *p = (const geniex_sampler_chain *)chain->ctx;
    return p->samplers.size();
}

// Individual sampler implementations

// Greedy sampler
static const char *geniex_sampler_greedy_name(const struct geniex_sampler * /*smpl*/) { return "greedy"; }

static void geniex_sampler_greedy_apply(struct geniex_sampler * /*smpl*/, geniex_token_data_array *cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}

static struct geniex_sampler_i geniex_sampler_greedy_i = {
    /* .name   = */ geniex_sampler_greedy_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_greedy_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ nullptr,
    /* .free   = */ nullptr,
};

struct geniex_sampler *geniex_sampler_init_greedy() {
    return geniex_sampler_init(&geniex_sampler_greedy_i, nullptr);
}

// Dist sampler
struct geniex_sampler_dist {
    const uint32_t seed;
    uint32_t seed_cur;
    std::mt19937 rng;
};

static const char *geniex_sampler_dist_name(const struct geniex_sampler * /*smpl*/) { return "dist"; }

static void geniex_sampler_dist_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_dist *)smpl->ctx;
    geniex_sampler_softmax_impl(cur_p);
    cur_p->selected = geniex_sample_dist(cur_p, ctx->rng);
}

static struct geniex_sampler *geniex_sampler_dist_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_dist *)smpl->ctx;
    auto *result = geniex_sampler_init_dist(ctx->seed);
    auto *result_ctx = (geniex_sampler_dist *)result->ctx;
    result_ctx->rng = ctx->rng;
    return result;
}

static void geniex_sampler_dist_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_dist *)smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static void geniex_sampler_dist_free(struct geniex_sampler *smpl) { delete (geniex_sampler_dist *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_dist_i = {
    /* .name   = */ geniex_sampler_dist_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_dist_apply,
    /* .reset  = */ geniex_sampler_dist_reset,
    /* .clone  = */ geniex_sampler_dist_clone,
    /* .free   = */ geniex_sampler_dist_free,
};

struct geniex_sampler *geniex_sampler_init_dist(uint32_t seed) {
    auto seed_cur = get_rng_seed(seed);
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_dist_i,
        /* .ctx   = */ new geniex_sampler_dist{
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .rng      = */ std::mt19937(seed_cur),
        });
}

// Softmax sampler
static const char *geniex_sampler_softmax_name(const struct geniex_sampler * /*smpl*/) { return "softmax"; }

static void geniex_sampler_softmax_apply(struct geniex_sampler * /*smpl*/, geniex_token_data_array *cur_p) {
    geniex_sampler_softmax_impl(cur_p);
}

static struct geniex_sampler_i geniex_sampler_softmax_i = {
    /* .name   = */ geniex_sampler_softmax_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_softmax_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ nullptr,
    /* .free   = */ nullptr,
};

struct geniex_sampler *geniex_sampler_init_softmax() {
    return geniex_sampler_init(&geniex_sampler_softmax_i, nullptr);
}

// Top-k sampler
struct geniex_sampler_top_k {
    const int32_t k;
};

static const char *geniex_sampler_top_k_name(const struct geniex_sampler * /*smpl*/) { return "top-k"; }

static void geniex_sampler_top_k_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_top_k *)smpl->ctx;
    geniex_sampler_top_k_impl(cur_p, ctx->k);
}

static struct geniex_sampler *geniex_sampler_top_k_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_top_k *)smpl->ctx;
    return geniex_sampler_init_top_k(ctx->k);
}

static void geniex_sampler_top_k_free(struct geniex_sampler *smpl) { delete (geniex_sampler_top_k *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_top_k_i = {
    /* .name   = */ geniex_sampler_top_k_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_top_k_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_top_k_clone,
    /* .free   = */ geniex_sampler_top_k_free,
};

struct geniex_sampler *geniex_sampler_init_top_k(int32_t k) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_top_k_i,
        /* .ctx   = */ new geniex_sampler_top_k{
            /* .k = */ k,
        });
}

// Top-p sampler
struct geniex_sampler_top_p {
    const float p;
    const size_t min_keep;
};

static const char *geniex_sampler_top_p_name(const struct geniex_sampler * /*smpl*/) { return "top-p"; }

static void geniex_sampler_top_p_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_top_p *)smpl->ctx;

    if (ctx->p >= 1.0f) {
        return;
    }

    geniex_sampler_softmax_impl(cur_p);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        if (cum_sum >= ctx->p && i + 1 >= ctx->min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    cur_p->size = last_idx;
}

static struct geniex_sampler *geniex_sampler_top_p_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_top_p *)smpl->ctx;
    return geniex_sampler_init_top_p(ctx->p, ctx->min_keep);
}

static void geniex_sampler_top_p_free(struct geniex_sampler *smpl) { delete (geniex_sampler_top_p *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_top_p_i = {
    /* .name   = */ geniex_sampler_top_p_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_top_p_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_top_p_clone,
    /* .free   = */ geniex_sampler_top_p_free,
};

struct geniex_sampler *geniex_sampler_init_top_p(float p, size_t min_keep) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_top_p_i,
        /* .ctx   = */ new geniex_sampler_top_p{
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        });
}

// Min-p sampler
struct geniex_sampler_min_p {
    const float p;
    const size_t min_keep;
};

static const char *geniex_sampler_min_p_name(const struct geniex_sampler * /*smpl*/) { return "min-p"; }

static void geniex_sampler_min_p_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_min_p *)smpl->ctx;

    if (ctx->p <= 0.0f || !cur_p->size) {
        return;
    }

    bool min_p_applied = false;

    // if the cur_p aren't sorted, try the unsorted implementation first
    if (!cur_p->sorted) {
        std::vector<geniex_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < cur_p->size; ++i) {
            max_logit = std::max(max_logit, cur_p->data[i].logit);
        }
        const float min_logit = max_logit + logf(ctx->p);  // min logit for p_i >= p * p_max

        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit >= min_logit) {
                filtered_tokens.push_back(cur_p->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (!filtered_tokens.empty() && filtered_tokens.size() >= ctx->min_keep) {
            memcpy(cur_p->data, filtered_tokens.data(), filtered_tokens.size() * sizeof(geniex_token_data));
            cur_p->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the cur_p are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!cur_p->sorted) {
            std::sort(cur_p->data, cur_p->data + cur_p->size,
                      [](const geniex_token_data &a, const geniex_token_data &b) { return a.logit > b.logit; });
            cur_p->sorted = true;
        }

        const float min_logit = cur_p->data[0].logit + logf(ctx->p);  // min logit for p_i >= p * p_max
        size_t i = 1;                                                 // first token always matches

        for (; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit < min_logit && i >= ctx->min_keep) {
                break;  // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        cur_p->size = i;
    }
}

static struct geniex_sampler *geniex_sampler_min_p_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_min_p *)smpl->ctx;
    return geniex_sampler_init_min_p(ctx->p, ctx->min_keep);
}

static void geniex_sampler_min_p_free(struct geniex_sampler *smpl) { delete (geniex_sampler_min_p *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_min_p_i = {
    /* .name   = */ geniex_sampler_min_p_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_min_p_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_min_p_clone,
    /* .free   = */ geniex_sampler_min_p_free,
};

struct geniex_sampler *geniex_sampler_init_min_p(float p, size_t min_keep) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_min_p_i,
        /* .ctx   = */ new geniex_sampler_min_p{
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        });
}

// Penalties sampler
struct geniex_sampler_penalties {
    const int32_t penalty_last_n;
    const float penalty_repeat;
    const float penalty_freq;
    const float penalty_present;

    ring_buffer<geniex_token> prev;
    std::unordered_map<geniex_token, int> token_count;
};

static const char *geniex_sampler_penalties_name(const struct geniex_sampler * /*smpl*/) { return "penalties"; }

static void geniex_sampler_penalties_accept(struct geniex_sampler *smpl, geniex_token token) {
    auto *ctx = (geniex_sampler_penalties *)smpl->ctx;
    if (ctx->penalty_last_n == 0) {
        return;
    }

    ctx->token_count[token]++;

    // if the ring buffer is full, remove the oldest token
    if (ctx->prev.size() >= (size_t)ctx->penalty_last_n) {
        const auto old = ctx->prev.front();

        ctx->token_count[old]--;
        if (ctx->token_count[old] == 0) {
            ctx->token_count.erase(old);
        }
    }

    ctx->prev.push_back(token);
}

static void geniex_sampler_penalties_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_penalties *)smpl->ctx;

    if ((ctx->penalty_last_n == 0) ||
        (ctx->penalty_repeat == 1.0f && ctx->penalty_freq == 0.0f && ctx->penalty_present == 0.0f)) {
        return;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token_iter = ctx->token_count.find(cur_p->data[i].id);
        if (token_iter == ctx->token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        assert(count > 0 && count <= ctx->penalty_last_n);

        // The academic publication that described this technique actually just only divided, but that would cause
        // tokens with negative logits to become more likely, which is obviously wrong. This is common fix for this
        // problem, which is to multiply by the penalty instead of dividing.
        if (cur_p->data[i].logit <= 0) {
            cur_p->data[i].logit *= ctx->penalty_repeat;
        } else {
            cur_p->data[i].logit /= ctx->penalty_repeat;
        }

        cur_p->data[i].logit -= float(count) * ctx->penalty_freq + float(count > 0) * ctx->penalty_present;
    }

    cur_p->sorted = false;
}

static void geniex_sampler_penalties_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_penalties *)smpl->ctx;
    ctx->prev.clear();
    ctx->token_count.clear();
}

static struct geniex_sampler *geniex_sampler_penalties_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_penalties *)smpl->ctx;
    auto *result =
        geniex_sampler_init_penalties(ctx->penalty_last_n, ctx->penalty_repeat, ctx->penalty_freq, ctx->penalty_present);

    // copy the state
    {
        auto *result_ctx = (geniex_sampler_penalties *)result->ctx;
        result_ctx->prev = ctx->prev;
    }

    return result;
}

static void geniex_sampler_penalties_free(struct geniex_sampler *smpl) { delete (geniex_sampler_penalties *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_penalties_i = {
    /* .name   = */ geniex_sampler_penalties_name,
    /* .accept = */ geniex_sampler_penalties_accept,
    /* .apply  = */ geniex_sampler_penalties_apply,
    /* .reset  = */ geniex_sampler_penalties_reset,
    /* .clone  = */ geniex_sampler_penalties_clone,
    /* .free   = */ geniex_sampler_penalties_free,
};

struct geniex_sampler *geniex_sampler_init_penalties(int32_t penalty_last_n, float penalty_repeat, float penalty_freq,
                                                 float penalty_present) {
    penalty_last_n = std::max(penalty_last_n, 0);

    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_penalties_i,
        /* .ctx   = */ new geniex_sampler_penalties{
            /* .penalty_last_n  = */ penalty_last_n,
            /* .penalty_repeat  = */ penalty_repeat,
            /* .penalty_freq    = */ penalty_freq,
            /* .penalty_present = */ penalty_present,
            /* .prev            = */ ring_buffer<geniex_token>(penalty_last_n),
            /* .token_count     = */ {},
        });
}

// XTC sampler implementation
struct geniex_sampler_xtc {
    const float probability;
    const float threshold;
    const size_t min_keep;

    const uint32_t seed;
    uint32_t seed_cur;

    std::mt19937 rng;
};

static const char *geniex_sampler_xtc_name(const struct geniex_sampler * /*smpl*/) { return "xtc"; }

static void geniex_sampler_xtc_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_xtc *)smpl->ctx;

    if (ctx->probability <= 0.0f || ctx->threshold <= 0.0f || cur_p->size < 2) {
        return;
    }

    // Ensure probabilities are computed via softmax
    geniex_sampler_softmax_impl(cur_p);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    size_t pos = 0;
    while (pos < cur_p->size) {
        if (dist(ctx->rng) < ctx->probability && cur_p->data[pos].p < ctx->threshold) {
            // Remove the token
            for (size_t i = pos; i < cur_p->size - 1; i++) {
                cur_p->data[i] = cur_p->data[i + 1];
            }
            cur_p->size--;
            if (cur_p->size <= ctx->min_keep) {
                break;
            }
        } else {
            pos++;
        }
    }
}

static struct geniex_sampler *geniex_sampler_xtc_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_xtc *)smpl->ctx;
    auto *result = geniex_sampler_init_xtc(ctx->probability, ctx->threshold, ctx->min_keep, ctx->seed);
    auto *result_ctx = (geniex_sampler_xtc *)result->ctx;
    result_ctx->rng = ctx->rng;
    return result;
}

static void geniex_sampler_xtc_free(struct geniex_sampler *smpl) { delete (geniex_sampler_xtc *)smpl->ctx; }

static void geniex_sampler_xtc_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_xtc *)smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static struct geniex_sampler_i geniex_sampler_xtc_i = {
    /* .name   = */ geniex_sampler_xtc_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_xtc_apply,
    /* .reset  = */ geniex_sampler_xtc_reset,
    /* .clone  = */ geniex_sampler_xtc_clone,
    /* .free   = */ geniex_sampler_xtc_free,
};

struct geniex_sampler *geniex_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed) {
    auto seed_cur = get_rng_seed(seed);
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_xtc_i,
        /* .ctx   = */ new geniex_sampler_xtc{
            /* .probability = */ p,
            /* .threshold   = */ t,
            /* .min_keep    = */ min_keep,
            /* .seed        = */ seed,
            /* .seed_cur    = */ seed_cur,
            /* .rng         = */ std::mt19937(seed_cur),
        });
}

// Mirostat sampler implementation
struct geniex_sampler_mirostat {
    const int32_t n_vocab;

    const uint32_t seed;
    uint32_t seed_cur;

    const float tau;
    const float eta;

    const int32_t m;

    float mu;

    std::mt19937 rng;
};

static const char *geniex_sampler_mirostat_name(const struct geniex_sampler * /*smpl*/) { return "mirostat"; }

static void geniex_sampler_mirostat_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_mirostat *)smpl->ctx;

    geniex_sampler_softmax_impl(cur_p);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(ctx->m - 1) && i < cur_p->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(cur_p->data[i].p / cur_p->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the desired value of tau, and bail out if we don't have enough data
    float temp = ctx->eta * s_hat + ctx->tau;
    size_t k = size_t(temp);
    if (ctx->m - 1 < 0 || k < 2 || k >= cur_p->size) {
        // fallback to greedy sampling
        cur_p->selected = 0;
        return;
    }

    // Sample the next word X using top-k sampling
    size_t idx = k;
    geniex_sampler_top_k_impl(cur_p, int(k));
    cur_p->selected = geniex_sample_dist(cur_p, ctx->rng);
    float observed_surprise = -log2f(cur_p->data[cur_p->selected].p);
    float e = observed_surprise - ctx->tau;

    // Update mu using the learning rate and error
    ctx->mu = ctx->mu - ctx->eta * e;
}

static struct geniex_sampler *geniex_sampler_mirostat_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_mirostat *)smpl->ctx;
    auto *result = geniex_sampler_init_mirostat(ctx->n_vocab, ctx->seed, ctx->tau, ctx->eta, ctx->m);
    auto *result_ctx = (geniex_sampler_mirostat *)result->ctx;
    result_ctx->mu = ctx->mu;
    result_ctx->rng = ctx->rng;
    return result;
}

static void geniex_sampler_mirostat_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_mirostat *)smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
    ctx->mu = 2.0f * ctx->tau;
}

static void geniex_sampler_mirostat_free(struct geniex_sampler *smpl) { delete (geniex_sampler_mirostat *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_mirostat_i = {
    /* .name   = */ geniex_sampler_mirostat_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_mirostat_apply,
    /* .reset  = */ geniex_sampler_mirostat_reset,
    /* .clone  = */ geniex_sampler_mirostat_clone,
    /* .free   = */ geniex_sampler_mirostat_free,
};

struct geniex_sampler *geniex_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m) {
    auto seed_cur = get_rng_seed(seed);
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_mirostat_i,
        /* .ctx   = */ new geniex_sampler_mirostat{
            /* .n_vocab  = */ n_vocab,
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .tau      = */ tau,
            /* .eta      = */ eta,
            /* .m        = */ m,
            /* .mu       = */ 2.0f * tau,
            /* .rng      = */ std::mt19937(seed_cur),
        });
}

// Mirostat v2 sampler implementation
struct geniex_sampler_mirostat_v2 {
    const uint32_t seed;
    uint32_t seed_cur;

    const float tau;
    const float eta;

    float mu;

    std::mt19937 rng;
};

static const char *geniex_sampler_mirostat_v2_name(const struct geniex_sampler * /*smpl*/) { return "mirostat-v2"; }

static void geniex_sampler_mirostat_v2_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_mirostat_v2 *)smpl->ctx;

    geniex_sampler_softmax_impl(cur_p);

    // Truncate the words with surprise values greater than mu
    cur_p->size = std::distance(
        cur_p->data, std::find_if(cur_p->data, cur_p->data + cur_p->size,
                                  [&](const geniex_token_data &candidate) { return -log2f(candidate.p) > ctx->mu; }));

    if (cur_p->size == 0) {
        cur_p->size = 1;
    }

    // Normalize the probabilities of the remaining words
    geniex_sampler_softmax_impl(cur_p);

    // Sample the next word X from the remaining words
    cur_p->selected = geniex_sample_dist(cur_p, ctx->rng);
    float observed_surprise = -log2f(cur_p->data[cur_p->selected].p);
    float e = observed_surprise - ctx->tau;

    // Update mu using the learning rate and error
    ctx->mu = ctx->mu - ctx->eta * e;
}

static void geniex_sampler_mirostat_v2_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_mirostat_v2 *)smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
    ctx->mu = 2.0f * ctx->tau;
}

static struct geniex_sampler *geniex_sampler_mirostat_v2_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_mirostat_v2 *)smpl->ctx;
    auto *result = geniex_sampler_init_mirostat_v2(ctx->seed, ctx->tau, ctx->eta);
    auto *result_ctx = (geniex_sampler_mirostat_v2 *)result->ctx;
    result_ctx->mu = ctx->mu;
    result_ctx->rng = ctx->rng;
    return result;
}

static void geniex_sampler_mirostat_v2_free(struct geniex_sampler *smpl) { delete (geniex_sampler_mirostat_v2 *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_mirostat_v2_i = {
    /* .name   = */ geniex_sampler_mirostat_v2_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_mirostat_v2_apply,
    /* .reset  = */ geniex_sampler_mirostat_v2_reset,
    /* .clone  = */ geniex_sampler_mirostat_v2_clone,
    /* .free   = */ geniex_sampler_mirostat_v2_free,
};

struct geniex_sampler *geniex_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta) {
    auto seed_cur = get_rng_seed(seed);
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_mirostat_v2_i,
        /* .ctx   = */ new geniex_sampler_mirostat_v2{
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .tau      = */ tau,
            /* .eta      = */ eta,
            /* .mu       = */ 2.0f * tau,
            /* .rng      = */ std::mt19937(seed_cur),
        });
}

// Top-N-Sigma sampler implementation
struct geniex_sampler_top_n_sigma {
    const float n;
};

static const char *geniex_sampler_top_n_sigma_name(const struct geniex_sampler * /*smpl*/) { return "top-n-sigma"; }

static void geniex_sampler_top_n_sigma_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_top_n_sigma *)smpl->ctx;

    if (ctx->n <= 0.0f || cur_p->size < 2) {
        return;
    }

    // Sort tokens by descending probability
    if (!cur_p->sorted) {
        geniex_sampler_softmax_impl(cur_p);
    }

    // Compute mean and variance of log probabilities
    float sum_log_p = 0.0f;
    float sum_log_p_sq = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float log_p = logf(cur_p->data[i].p);
        sum_log_p += log_p;
        sum_log_p_sq += log_p * log_p;
    }

    float mean_log_p = sum_log_p / cur_p->size;
    float variance_log_p = sum_log_p_sq / cur_p->size - mean_log_p * mean_log_p;
    float std_log_p = sqrtf(variance_log_p);

    // Compute threshold
    float threshold = mean_log_p - ctx->n * std_log_p;

    // Remove tokens below threshold
    size_t new_size = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (logf(cur_p->data[i].p) >= threshold) {
            cur_p->data[new_size] = cur_p->data[i];
            new_size++;
        }
    }

    cur_p->size = std::max(new_size, size_t(1));
}

static struct geniex_sampler *geniex_sampler_top_n_sigma_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_top_n_sigma *)smpl->ctx;
    return geniex_sampler_init_top_n_sigma(ctx->n);
}

static void geniex_sampler_top_n_sigma_free(struct geniex_sampler *smpl) { delete (geniex_sampler_top_n_sigma *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_top_n_sigma_i = {
    /* .name   = */ geniex_sampler_top_n_sigma_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_top_n_sigma_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_top_n_sigma_clone,
    /* .free   = */ geniex_sampler_top_n_sigma_free,
};

struct geniex_sampler *geniex_sampler_init_top_n_sigma(float n) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_top_n_sigma_i,
        /* .ctx   = */ new geniex_sampler_top_n_sigma{
            /* .n = */ n,
        });
}

// DRY sampler implementation
struct geniex_sampler_dry {
    int32_t total_context_size;

    const float dry_multiplier;
    const float dry_base;
    const int32_t dry_allowed_length;
    const int32_t dry_penalty_last_n;

    geniex_vocab_interface *vocab;

    std::unordered_multimap<geniex_token, std::vector<geniex_token>> dry_processed_breakers;
    std::vector<int> dry_repeat_count;
    std::unordered_map<geniex_token, int> dry_max_token_repeat;
    ring_buffer<geniex_token> last_tokens;
};

// Ported from llama-sampling.cpp - finds overlapping token sequences for sequence breakers
static void get_overlapping_token_sequences(
    geniex_vocab_interface *vocab, const std::string &str,
    std::unordered_multimap<geniex_token, std::vector<geniex_token>> &token_sequences, int max_tail_len = -1) {
    for (geniex_token token_id = 0; token_id < vocab->n_tokens(); token_id++) {
        std::string word = vocab->detokenize({token_id}, true);
        if (word.find(str) != std::string::npos) {
            token_sequences.emplace(token_id, std::vector<geniex_token>());
        } else {
            size_t word_len = word.size();
            size_t str_len = str.size();
            size_t pos = -1;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool match = true;
                size_t i;
                for (i = 1; i < str_len && i + pos < word_len; ++i) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    std::vector<geniex_token> tokenization = vocab->tokenize(str.substr(i), false, false);
                    if (max_tail_len >= 0 && tokenization.size() > (size_t)max_tail_len) {
                        tokenization.resize(max_tail_len);
                    }

                    // Ensure we don't already have a duplicate matching tokenization
                    auto its = token_sequences.equal_range(token_id);
                    bool found = false;
                    for (auto it = its.first; it != its.second; ++it) {
                        if (tokenization == it->second) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        token_sequences.emplace(token_id, tokenization);
                    }
                }
            }
        }
    }
}

static const char *geniex_sampler_dry_name(const struct geniex_sampler * /*smpl*/) { return "dry"; }

static void geniex_sampler_dry_accept(struct geniex_sampler *smpl, geniex_token token) {
    auto *ctx = (geniex_sampler_dry *)smpl->ctx;
    if (ctx->dry_multiplier == 0.0f || ctx->dry_base < 1.0f || ctx->dry_penalty_last_n == 0) {
        return;
    }

    ctx->last_tokens.push_back(token);
}

// Ported from llama-sampling.cpp - full DRY implementation with Z-algorithm
static void geniex_sampler_dry_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_dry *)smpl->ctx;

    if (ctx->dry_multiplier == 0.0f || ctx->dry_base < 1.0f || ctx->dry_penalty_last_n == 0) {
        return;
    }

    int32_t effective_dry_penalty_last_n =
        (ctx->dry_penalty_last_n == -1) ? ctx->total_context_size : std::max(ctx->dry_penalty_last_n, 0);
    int last_n_repeat =
        std::min(std::min((int)ctx->last_tokens.size(), effective_dry_penalty_last_n), ctx->total_context_size);

    if (last_n_repeat <= ctx->dry_allowed_length) {
        return;
    }

    ctx->dry_repeat_count.assign(last_n_repeat, 0);
    ctx->dry_max_token_repeat.clear();

    // Step 1: Look for restart sequences to limit the maximum repetition length.
    // Work backwards through the context looking for any token that begins a restart sequence.
    //
    // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
    // sequences that together comprise a restart sequence. This allows us to quickly check
    // whether each token is the head of a complete sequence. Most restart sequences are actually
    // a single token, and for these the "tail" is an empty vector.
    //
    // If the token is a "head", test all restart sequences that begin with this token
    // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
    // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
    // longest matching sequence (if any) is used to limit the maximum repetition length.
    //
    // Note that in the case case of a short sequence contained in a longer one, this might fail to
    // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
    // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
    // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
    //
    // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
    // have already clamped the maximum tail sequence length when generating `restart_sequences`.
    // With clamping, this scan is O(N) in the context length.

    int rep_limit = last_n_repeat;
    for (int i = 0; i < last_n_repeat; ++i) {
        geniex_token token = ctx->last_tokens.rat(i);
        auto its = ctx->dry_processed_breakers.equal_range(token);
        if (its.first == ctx->dry_processed_breakers.end()) {
            continue;
        }
        int longest_match = -1;
        for (auto it = its.first; it != its.second; ++it) {
            // Note that (*it) does not contain the head character, so seq_len will be
            // the restart sequence length minus 1.
            // In the common case of a single-token restart sequence, (*it) will be empty
            // and we will trivially match.
            int seq_len = (int)it->second.size();
            if (seq_len > longest_match && seq_len <= (int)i) {
                bool match = true;
                for (int offset = 0; offset < seq_len; ++offset) {
                    // The -1 when indexing `last_tokens` is because we already matched the head.
                    if (it->second[offset] != ctx->last_tokens.rat(i - offset - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    longest_match = seq_len;
                }
            }
        }
        if (longest_match >= 0) {
            // We found a restart sequence starting `i` tokens from the end and continuing for
            // `longest_match` tokens.
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < ctx->dry_allowed_length) {
        return;
    }

    // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
    // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
    // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
    //
    // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
    // https://ivanyu.me/blog/2013/10/15/z-algorithm/
    //
    // The goal is to find the longest repeating sequence, ending at the last token of the context.
    // For each of the last `rep_limit` characters, we compute the length of the longest substring
    // ending at that position that is also a suffix of the context.
    //
    // For example, with context "a b c c b c y a b c", the longest suffixes are:
    // Last token: "c" (substring of length 1)
    // Second-to-last: "b c" (substring of length 2)
    // Third-to-last: "a b c" (substring of length 3)
    // Fourth-to-last: "y a b c" (substring of length 0, as "y a b c" is not a suffix)
    // And so on.
    //
    // The Z-algorithm computes these lengths efficiently in O(N) time.

    {
        const int last = last_n_repeat - 1;
        int rt = 0, lt = 0;

        for (int k = 1; k < last_n_repeat; ++k) {
            if (k > rt) {
                // If k is outside the current Z-box, do naive computation.
                int n = 0;
                while (n + k < last_n_repeat && ctx->last_tokens.rat(n) == ctx->last_tokens.rat(n + k)) {
                    ++n;
                }
                ctx->dry_repeat_count[last - k] = std::min(n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            } else {
                // If k is inside the current Z-box, consider two cases.

                int p = k - lt;  // Pair index.
                int right_part_len = rt - k + 1;

                if (ctx->dry_repeat_count[last - p] < right_part_len) {
                    int n = std::min(ctx->dry_repeat_count[last - p], rep_limit);
                    ctx->dry_repeat_count[last - k] = n;
                } else {
                    int i = rt + 1;
                    while (i < last_n_repeat && ctx->last_tokens.rat(i) == ctx->last_tokens.rat(i - k)) {
                        i += 1;
                    }

                    int n = std::min(i - k, rep_limit);
                    ctx->dry_repeat_count[last - k] = n;
                    lt = k;
                    rt = i - 1;
                }
            }
        }
    }

    // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
    // that would be generated by emitting each new token that would extend a sequence.
    //
    // Following the same example as above:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //
    // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
    // c: 3 -> 4 (from `a b c` to `a b c c`)
    // b: 1 -> 2 (from `c` to `c b`)
    // y: 2 -> 3 (from `b c` to `b c y`)

    for (int i = 0; i < last_n_repeat - 1; ++i) {
        int repeat_len = ctx->dry_repeat_count[i];
        if (repeat_len >= ctx->dry_allowed_length) {
            // This token ends a repeat, so the next token would continue one.
            // By convention, the value of `repeat_len` only includes the tokens currently
            // in the context, not the new token that would be added.
            geniex_token token = ctx->last_tokens.rat(last_n_repeat - 2 - i);
            // Track the maximum sequence ending in this token.
            const auto &it = ctx->dry_max_token_repeat.find(token);
            if (it == ctx->dry_max_token_repeat.end() || it->second < repeat_len) {
                ctx->dry_max_token_repeat[token] = repeat_len;
            }
        }
    }

    // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

    // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
    // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exponent = 0;
    if (ctx->dry_base > 1.000001f) {
        max_exponent = FLOAT_MAX_LOG / std::log(ctx->dry_base);
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto &af_kvp = ctx->dry_max_token_repeat.find(cur_p->data[i].id);
        if (af_kvp != ctx->dry_max_token_repeat.end()) {
            // Check all sequence breakers starting with this token
            auto range = ctx->dry_processed_breakers.equal_range(cur_p->data[i].id);
            bool is_single_token_breaker = false;

            for (auto it = range.first; it != range.second; ++it) {
                if (it->second.empty()) {
                    is_single_token_breaker = true;
                    break;
                }
            }

            // Apply penalty only if it's not a single-token sequence breaker
            if (!is_single_token_breaker) {
                int repeat_exp = af_kvp->second - ctx->dry_allowed_length;
                if (max_exponent > 0 && repeat_exp > max_exponent) {
                    repeat_exp = max_exponent;
                }
                float penalty = ctx->dry_multiplier * std::pow(ctx->dry_base, repeat_exp);
                cur_p->data[i].logit -= penalty;
            }
        }
    }

    cur_p->sorted = false;
}

static void geniex_sampler_dry_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_dry *)smpl->ctx;
    ctx->last_tokens.clear();
    ctx->dry_repeat_count.clear();
    ctx->dry_max_token_repeat.clear();
}

static struct geniex_sampler *geniex_sampler_dry_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_dry *)smpl->ctx;

    // Create a new DRY sampler with the same vocab interface and no sequence breakers
    // (we'll copy the processed breakers directly)
    auto *result = geniex_sampler_init_dry(ctx->vocab, ctx->total_context_size, ctx->dry_multiplier, ctx->dry_base,
                                         ctx->dry_allowed_length, ctx->dry_penalty_last_n, nullptr, 0);

    // Copy the state, including the processed breakers
    auto *result_ctx = (geniex_sampler_dry *)result->ctx;
    result_ctx->dry_processed_breakers = ctx->dry_processed_breakers;
    result_ctx->dry_repeat_count = ctx->dry_repeat_count;
    result_ctx->dry_max_token_repeat = ctx->dry_max_token_repeat;
    result_ctx->last_tokens = ctx->last_tokens;

    return result;
}

static void geniex_sampler_dry_free(struct geniex_sampler *smpl) { delete (geniex_sampler_dry *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_dry_i = {
    /* .name   = */ geniex_sampler_dry_name,
    /* .accept = */ geniex_sampler_dry_accept,
    /* .apply  = */ geniex_sampler_dry_apply,
    /* .reset  = */ geniex_sampler_dry_reset,
    /* .clone  = */ geniex_sampler_dry_clone,
    /* .free   = */ geniex_sampler_dry_free,
};

struct geniex_sampler *geniex_sampler_init_dry(geniex_vocab_interface *vocab, int32_t context_size, float dry_multiplier,
                                           float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n,
                                           const char **seq_breakers, size_t num_breakers) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    std::unordered_multimap<geniex_token, std::vector<geniex_token>> processed_breakers;
    const int MAX_CHAR_LEN = 40;
    const int MAX_SEQ_LEN = 20;

    const bool dry_enabled = (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

    if (dry_enabled && vocab && seq_breakers != nullptr && num_breakers > 0) {
        // Process sequence breakers
        for (size_t i = 0; i < num_breakers; ++i) {
            if (seq_breakers[i] == nullptr || std::strlen(seq_breakers[i]) == 0) {
                continue;
            }

            std::string sequence_break(seq_breakers[i]);
            if (sequence_break.empty()) {
                continue;
            }

            if (sequence_break.size() > MAX_CHAR_LEN) {
                sequence_break.resize(MAX_CHAR_LEN);
            }

            get_overlapping_token_sequences(vocab, sequence_break, processed_breakers, MAX_SEQ_LEN);
        }
    }

    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_dry_i,
        /* .ctx   = */ new geniex_sampler_dry{
            /* .total_context_size     = */ context_size,
            /* .dry_multiplier         = */ dry_multiplier,
            /* .dry_base               = */ dry_base,
            /* .dry_allowed_length     = */ dry_allowed_length,
            /* .dry_penalty_last_n     = */ dry_penalty_last_n,
            /* .vocab                  = */ vocab,
            /* .dry_processed_breakers = */ std::move(processed_breakers),
            /* .dry_repeat_count       = */
            dry_enabled ? std::vector<int>(effective_dry_penalty_last_n, 0) : std::vector<int>{},
            /* .dry_max_token_repeat   = */ {},
            /* .last_tokens            = */
            dry_enabled ? ring_buffer<geniex_token>(effective_dry_penalty_last_n) : ring_buffer<geniex_token>(0),
        });
}

// Variant that accepts pre-tokenized sequence breakers — no vocab required.
// Each sequence [t0, t1, t2, ...] maps to head t0 → tail [t1, t2, ...].
struct geniex_sampler *geniex_sampler_init_dry_tokenized(
        int32_t context_size, float dry_multiplier, float dry_base,
        int32_t dry_allowed_length, int32_t dry_penalty_last_n,
        const geniex_token * const *breaker_seqs, const size_t *breaker_lens, size_t num_breakers) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    std::unordered_multimap<geniex_token, std::vector<geniex_token>> processed_breakers;

    const bool dry_enabled = (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

    if (dry_enabled && breaker_seqs != nullptr && num_breakers > 0) {
        for (size_t i = 0; i < num_breakers; ++i) {
            if (breaker_lens[i] == 0 || breaker_seqs[i] == nullptr) continue;
            geniex_token head = breaker_seqs[i][0];
            std::vector<geniex_token> tail(breaker_seqs[i] + 1, breaker_seqs[i] + breaker_lens[i]);
            processed_breakers.emplace(head, std::move(tail));
        }
    }

    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_dry_i,
        /* .ctx   = */ new geniex_sampler_dry{
            /* .total_context_size     = */ context_size,
            /* .dry_multiplier         = */ dry_multiplier,
            /* .dry_base               = */ dry_base,
            /* .dry_allowed_length     = */ dry_allowed_length,
            /* .dry_penalty_last_n     = */ dry_penalty_last_n,
            /* .vocab                  = */ nullptr,
            /* .dry_processed_breakers = */ std::move(processed_breakers),
            /* .dry_repeat_count       = */
            dry_enabled ? std::vector<int>(effective_dry_penalty_last_n, 0) : std::vector<int>{},
            /* .dry_max_token_repeat   = */ {},
            /* .last_tokens            = */
            dry_enabled ? ring_buffer<geniex_token>(effective_dry_penalty_last_n) : ring_buffer<geniex_token>(0),
        });
}

// Temperature sampler
struct geniex_sampler_temp {
    const float temp;
};

static const char *geniex_sampler_temp_name(const struct geniex_sampler * /*smpl*/) { return "temp"; }

static void geniex_sampler_temp_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_temp *)smpl->ctx;
    geniex_sampler_temp_impl(cur_p, ctx->temp);
}

static struct geniex_sampler *geniex_sampler_temp_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_temp *)smpl->ctx;
    return geniex_sampler_init_temp(ctx->temp);
}

static void geniex_sampler_temp_free(struct geniex_sampler *smpl) { delete (geniex_sampler_temp *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_temp_i = {
    /* .name   = */ geniex_sampler_temp_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_temp_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_temp_clone,
    /* .free   = */ geniex_sampler_temp_free,
};

struct geniex_sampler *geniex_sampler_init_temp(float temp) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_temp_i,
        /* .ctx   = */ new geniex_sampler_temp{
            /*.temp = */ temp,
        });
}

// Typical sampler
struct geniex_sampler_typical {
    const float p;
    const size_t min_keep;
};

static const char *geniex_sampler_typical_name(const struct geniex_sampler * /*smpl*/) { return "typical"; }

static void geniex_sampler_typical_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_typical *)smpl->ctx;

    if (ctx->p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    geniex_sampler_softmax_impl(cur_p);

    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float shifted_score = fabsf(-logf(cur_p->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(cur_p->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return shifted_scores[a] < shifted_scores[b]; });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += cur_p->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > ctx->p && (ctx->min_keep == 0 || i >= ctx->min_keep - 1)) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<geniex_token_data> cur_p_new;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        cur_p_new.push_back(cur_p->data[idx]);
    }

    // Replace the data in cur_p with the cur_p_new data
    std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
    cur_p->size = cur_p_new.size();
    cur_p->sorted = false;
}

static struct geniex_sampler *geniex_sampler_typical_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_typical *)smpl->ctx;
    return geniex_sampler_init_typical(ctx->p, ctx->min_keep);
}

static void geniex_sampler_typical_free(struct geniex_sampler *smpl) { delete (geniex_sampler_typical *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_typical_i = {
    /* .name   = */ geniex_sampler_typical_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_typical_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_typical_clone,
    /* .free   = */ geniex_sampler_typical_free,
};

struct geniex_sampler *geniex_sampler_init_typical(float p, size_t min_keep) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_typical_i,
        /* .ctx   = */ new geniex_sampler_typical{
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        });
}

// Extended temperature sampler
struct geniex_sampler_temp_ext {
    const float temp;
    const float delta;
    const float exponent;
};

static const char *geniex_sampler_temp_ext_name(const struct geniex_sampler * /*smpl*/) { return "temp-ext"; }

static void geniex_sampler_temp_ext_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    const auto *ctx = (geniex_sampler_temp_ext *)smpl->ctx;
    if (ctx->delta > 0) {
        const float min_temp = std::max(0.0f, ctx->temp - ctx->delta);
        const float max_temp = ctx->temp + ctx->delta;

        float exponent_val = ctx->exponent;

        // no need to do anything if there is only one (or zero) candidates
        if (cur_p->size <= 1) {
            return;
        }

        // Calculate maximum possible entropy
        float max_entropy = -logf(1.0f / cur_p->size);

        geniex_sampler_softmax_impl(cur_p);

        // Calculate entropy of the softmax probabilities
        float entropy = 0.0f;
        for (size_t i = 0; i < cur_p->size; ++i) {
            float prob = cur_p->data[i].p;
            if (prob > 0.0f) {  // Ensure no log(0)
                entropy -= prob * logf(prob);
            }
        }

        // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
        float normalized_entropy = entropy / max_entropy;

        // Map the normalized entropy to the desired temperature range using the power function
        float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

        // Apply the dynamically calculated temperature scaling
        geniex_sampler_temp_impl(cur_p, dyn_temp);

        // Re-compute softmax probabilities after scaling logits with dynamic temperature
        const double max_l_double = cur_p->data[0].logit;

        double cum_sum_double = 0.0;
        for (size_t i = 0; i < cur_p->size; ++i) {
            double p = exp(cur_p->data[i].logit - max_l_double);
            cur_p->data[i].p = p;  // Store the scaled probability
            cum_sum_double += p;
        }

        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= cum_sum_double;  // Re-normalize the probabilities
        }
    } else {
        geniex_sampler_temp_impl(cur_p, ctx->temp);
    }
}

static struct geniex_sampler *geniex_sampler_temp_ext_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_temp_ext *)smpl->ctx;
    return geniex_sampler_init_temp_ext(ctx->temp, ctx->delta, ctx->exponent);
}

static void geniex_sampler_temp_ext_free(struct geniex_sampler *smpl) { delete (geniex_sampler_temp_ext *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_temp_ext_i = {
    /* .name   = */ geniex_sampler_temp_ext_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_temp_ext_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_temp_ext_clone,
    /* .free   = */ geniex_sampler_temp_ext_free,
};

struct geniex_sampler *geniex_sampler_init_temp_ext(float temp, float delta, float exponent) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_temp_ext_i,
        /* .ctx   = */ new geniex_sampler_temp_ext{
            /* .temp     = */ temp,
            /* .delta    = */ delta,
            /* .exponent = */ exponent,
        });
}

// Logit bias sampler
struct geniex_sampler_logit_bias {
    const int32_t n_vocab;
    const std::vector<geniex_logit_bias> logit_bias;
    std::vector<geniex_logit_bias> to_search;
};

static const char *geniex_sampler_logit_bias_name(const struct geniex_sampler * /*smpl*/) { return "logit-bias"; }

static void geniex_sampler_logit_bias_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_logit_bias *)smpl->ctx;

    if (ctx->logit_bias.empty()) {
        return;
    }

    ctx->to_search.clear();

    // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
    for (const auto &lb : ctx->logit_bias) {
        if (lb.token >= 0 && cur_p->size > (size_t)lb.token && cur_p->data[lb.token].id == lb.token) {
            cur_p->data[lb.token].logit += lb.bias;
        } else {
            ctx->to_search.push_back(lb);
        }
    }

    if (ctx->to_search.empty()) {
        return;
    }

    // search for the remaining candidates that were not found in the previous step
    for (size_t i = 0; i < cur_p->size; ++i) {
        for (const auto &lb : ctx->to_search) {
            if (cur_p->data[i].id == lb.token) {
                cur_p->data[i].logit += lb.bias;
                break;
            }
        }
    }
}

static struct geniex_sampler *geniex_sampler_logit_bias_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_logit_bias *)smpl->ctx;
    return geniex_sampler_init_logit_bias(ctx->n_vocab, ctx->logit_bias.size(), ctx->logit_bias.data());
}

static void geniex_sampler_logit_bias_free(struct geniex_sampler *smpl) { delete (geniex_sampler_logit_bias *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_logit_bias_i = {
    /* .name   = */ geniex_sampler_logit_bias_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_logit_bias_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_logit_bias_clone,
    /* .free   = */ geniex_sampler_logit_bias_free,
};

struct geniex_sampler *geniex_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias,
                                                  const geniex_logit_bias *logit_bias) {
    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_logit_bias_i,
        /* .ctx   = */ new geniex_sampler_logit_bias{
            /* .n_vocab    = */ n_vocab,
            /* .logit_bias = */ std::vector<geniex_logit_bias>(logit_bias, logit_bias + n_logit_bias),
            /* .to_search  = */ {},
        });
}

// Grammar sampler - requires vocab interface
struct geniex_sampler_grammar {
    geniex_vocab_interface *vocab;
    std::string grammar_str;
    std::string grammar_root;
    struct geniex_grammar *grammar;  // Actual grammar implementation
};

static const char *geniex_sampler_grammar_name(const struct geniex_sampler * /*smpl*/) { return "grammar"; }

static void geniex_sampler_grammar_accept(struct geniex_sampler *smpl, geniex_token token) {
    auto *ctx = (geniex_sampler_grammar *)smpl->ctx;
    if (!ctx->vocab || !ctx->grammar) {
        return;
    }

    // Accept token in grammar
    geniex_grammar_accept(*ctx->grammar, token);
}

static void geniex_sampler_grammar_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_grammar *)smpl->ctx;
    if (!ctx->vocab || !ctx->grammar) {
        return;
    }

    // Apply grammar constraints
    geniex_grammar_apply(*ctx->grammar, cur_p);
}

static void geniex_sampler_grammar_reset(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_grammar *)smpl->ctx;
    if (!ctx->vocab || !ctx->grammar) {
        return;
    }

    // Reset grammar - recreate from string
    geniex_grammar_free(ctx->grammar);
    ctx->grammar = geniex_grammar_init_from_string(ctx->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str());
}

static struct geniex_sampler *geniex_sampler_grammar_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_grammar *)smpl->ctx;

    auto *result = geniex_sampler_init_grammar(ctx->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str());

    return result;
}

static void geniex_sampler_grammar_free(struct geniex_sampler *smpl) {
    auto *ctx = (geniex_sampler_grammar *)smpl->ctx;

    if (ctx->grammar) {
        geniex_grammar_free(ctx->grammar);
    }

    delete ctx;
}

static struct geniex_sampler_i geniex_sampler_grammar_i = {
    /* .name   = */ geniex_sampler_grammar_name,
    /* .accept = */ geniex_sampler_grammar_accept,
    /* .apply  = */ geniex_sampler_grammar_apply,
    /* .reset  = */ geniex_sampler_grammar_reset,
    /* .clone  = */ geniex_sampler_grammar_clone,
    /* .free   = */ geniex_sampler_grammar_free,
};

struct geniex_sampler *geniex_sampler_init_grammar(geniex_vocab_interface *vocab, const char *grammar_str,
                                               const char *grammar_root) {
    if (!vocab || !grammar_str) {
        return nullptr;
    }

    auto *ctx = new geniex_sampler_grammar{
        /* .vocab        = */ vocab,
        /* .grammar_str  = */ grammar_str,
        /* .grammar_root = */ grammar_root ? grammar_root : "root",
        /* .grammar      = */ nullptr,
    };

    // Initialize grammar from string
    ctx->grammar = geniex_grammar_init_from_string(vocab, grammar_str, ctx->grammar_root.c_str());

    if (!ctx->grammar) {
        delete ctx;
        return nullptr;
    }

    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_grammar_i,
        /* .ctx   = */ ctx);
}

// Infill sampler - requires vocab interface
struct geniex_sampler_infill {
    geniex_vocab_interface *vocab;
    std::vector<char> buf0;
    std::vector<char> buf1;
};

static const char *geniex_sampler_infill_name(const struct geniex_sampler * /*smpl*/) { return "infill"; }

static void geniex_sampler_infill_apply(struct geniex_sampler *smpl, geniex_token_data_array *cur_p) {
    auto *ctx = (geniex_sampler_infill *)smpl->ctx;
    if (!ctx->vocab) {
        return;
    }

    geniex_sampler_softmax_impl(cur_p);

    float p_txt_sum = 0.0f;
    float p_eog_sum = 0.0f;

    // Calculate probabilities for text vs end-of-generation tokens
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (ctx->vocab->is_eog(cur_p->data[i].id)) {
            p_eog_sum += cur_p->data[i].p;
        } else {
            p_txt_sum += cur_p->data[i].p;
        }
    }

    // If ratio of text to EOG is too low, prefer EOG tokens
    if (3 * p_eog_sum * cur_p->size > p_txt_sum && p_eog_sum > 0.0f) {
        // Keep only EOG tokens
        const auto size_org = cur_p->size;
        cur_p->size = 0;
        float p_sum = 0.0f;

        for (size_t i = 0; i < size_org; ++i) {
            if (ctx->vocab->is_eog(cur_p->data[i].id)) {
                p_sum += cur_p->data[i].p;
                cur_p->data[cur_p->size++] = cur_p->data[i];
            }
        }

        // Normalize probabilities
        if (p_sum > 0.0f) {
            for (size_t i = 0; i < cur_p->size; ++i) {
                cur_p->data[i].p /= p_sum;
            }
        }
        return;
    }

    size_t n_combined = 0;

    // Combine tokens with common prefixes
    for (size_t i0 = 0; i0 < cur_p->size; ++i0) {
        if (cur_p->data[i0].logit == -INFINITY) {
            continue;
        }

        for (size_t i1 = i0 + 1; i1 < cur_p->size; ++i1) {
            if (cur_p->data[i1].logit == -INFINITY) {
                continue;
            }

            int len0 = ctx->vocab->token_to_piece(cur_p->data[i0].id, ctx->buf0.data(), (int)ctx->buf0.size());
            if (len0 < 0) {
                ctx->buf0.resize(-len0);
                len0 = ctx->vocab->token_to_piece(cur_p->data[i0].id, ctx->buf0.data(), (int)ctx->buf0.size());
            }

            int len1 = ctx->vocab->token_to_piece(cur_p->data[i1].id, ctx->buf1.data(), (int)ctx->buf1.size());
            if (len1 < 0) {
                ctx->buf1.resize(-len1);
                len1 = ctx->vocab->token_to_piece(cur_p->data[i1].id, ctx->buf1.data(), (int)ctx->buf1.size());
            }

            // Check if token i0 is a prefix of token i1
            if (len0 > 0 && len0 <= len1 && memcmp(ctx->buf0.data(), ctx->buf1.data(), len0) == 0) {
                int dst = i0;
                int src = i1;

                // Merge into token with higher probability
                if (cur_p->data[i1].p > cur_p->data[i0].p) {
                    std::swap(dst, src);
                }

                cur_p->data[dst].p += cur_p->data[src].p;
                cur_p->data[src].logit = -INFINITY;
                cur_p->data[src].p = 0.0f;

                n_combined++;
            }
        }
    }

    // Apply threshold filtering
    size_t n_non_eog = 0;
    size_t size_org = cur_p->size;
    float p_sum = 0.0f;
    float thold = 0.2f;

    cur_p->size = 0;

    for (size_t i = 0; i < size_org; ++i) {
        if (cur_p->data[i].logit == -INFINITY) {
            continue;
        }

        const bool is_eog = ctx->vocab->is_eog(cur_p->data[i].id);

        if (cur_p->data[i].p < thold && !is_eog) {
            continue;
        }

        if (!is_eog) {
            ++n_non_eog;
        }

        p_sum += cur_p->data[i].p;
        cur_p->data[cur_p->size++] = cur_p->data[i];
    }

    // If no non-EOG tokens left, use single EOT token
    if (n_non_eog == 0) {
        cur_p->size = 1;
        cur_p->data[0].id = ctx->vocab->token_eot();
        cur_p->data[0].logit = 1.0f;
        cur_p->data[0].p = 1.0f;
        return;
    }

    // Normalize probabilities
    if (p_sum > 0.0f) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= p_sum;
        }
    }

    // Apply second threshold
    size_org = cur_p->size;
    p_sum = 0.0f;
    thold = 1.0f / (n_non_eog + 1);

    cur_p->size = 0;

    for (size_t i = 0; i < size_org; ++i) {
        const bool is_eog = ctx->vocab->is_eog(cur_p->data[i].id);

        if (cur_p->data[i].p < thold && !is_eog) {
            continue;
        }

        p_sum += cur_p->data[i].p;
        cur_p->data[cur_p->size++] = cur_p->data[i];
    }

    // Final normalization
    if (p_sum > 0.0f) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= p_sum;
        }
    }
}

static struct geniex_sampler *geniex_sampler_infill_clone(const struct geniex_sampler *smpl) {
    const auto *ctx = (const geniex_sampler_infill *)smpl->ctx;
    return geniex_sampler_init_infill(ctx->vocab);
}

static void geniex_sampler_infill_free(struct geniex_sampler *smpl) { delete (geniex_sampler_infill *)smpl->ctx; }

static struct geniex_sampler_i geniex_sampler_infill_i = {
    /* .name   = */ geniex_sampler_infill_name,
    /* .accept = */ nullptr,
    /* .apply  = */ geniex_sampler_infill_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ geniex_sampler_infill_clone,
    /* .free   = */ geniex_sampler_infill_free,
};

struct geniex_sampler *geniex_sampler_init_infill(geniex_vocab_interface *vocab) {
    if (!vocab) {
        return nullptr;
    }

    return geniex_sampler_init(
        /* .iface = */ &geniex_sampler_infill_i,
        /* .ctx   = */ new geniex_sampler_infill{
            /* .vocab = */ vocab,
            /* .buf0  = */ std::vector<char>(512),
            /* .buf1  = */ std::vector<char>(512),
        });
}

// Performance and utility functions
uint32_t geniex_sampler_get_seed(const struct geniex_sampler *smpl) {
    if (smpl->iface == &geniex_sampler_dist_i) {
        return ((const geniex_sampler_dist *)smpl->ctx)->seed_cur;
    }

    if (smpl->iface == &geniex_sampler_chain_i) {
        const auto *ctx = (const geniex_sampler_chain *)smpl->ctx;
        for (auto it = ctx->samplers.rbegin(); it != ctx->samplers.rend(); ++it) {
            const uint32_t seed = geniex_sampler_get_seed(*it);
            if (seed != GENIEX_DEFAULT_SEED) {
                return seed;
            }
        }
    }

    return GENIEX_DEFAULT_SEED;
}

struct geniex_perf_sampler_data geniex_perf_sampler(const struct geniex_sampler *chain) {
    struct geniex_perf_sampler_data data = {};

    if (chain == nullptr || chain->iface != &geniex_sampler_chain_i) {
        return data;
    }

    const auto *ctx = (const struct geniex_sampler_chain *)chain->ctx;

    data.t_sample_ms = 1e-3 * ctx->t_sample_us;
    data.n_sample = std::max(0, ctx->n_sample);

    return data;
}

void geniex_perf_sampler_print(const struct geniex_sampler *chain) {
    const auto data = geniex_perf_sampler(chain);

    fprintf(stdout, "\nsampling time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            data.t_sample_ms, data.n_sample, data.t_sample_ms / data.n_sample, 1e3 / data.t_sample_ms * data.n_sample);
}

void geniex_perf_sampler_reset(struct geniex_sampler *chain) {
    if (chain == nullptr || chain->iface != &geniex_sampler_chain_i) {
        return;
    }

    auto *ctx = (struct geniex_sampler_chain *)chain->ctx;
    ctx->t_sample_us = ctx->n_sample = 0;
}

struct geniex_sampler_chain_params geniex_sampler_chain_default_params() {
    return {
        /*.no_perf = */ false,
    };
}

//
// sampler context implementation (similar to common_sampler in llama.cpp)
//

struct geniex_sampler_context {
    geniex_sampler_params params;
    geniex_vocab_interface *vocab;

    struct geniex_sampler *grammar;  // Grammar sampler (if any)
    struct geniex_sampler *chain;    // Main sampler chain

    ring_buffer<geniex_token> prev;  // Token history

    std::vector<geniex_token_data> cur;  // Current candidate tokens
    geniex_token_data_array cur_p;       // Current candidate array

    void set_logits(const float *logits, int32_t n_vocab) {
        if (!logits || n_vocab <= 0) {
            return;
        }

        // Reserve capacity if needed
        if (cur.capacity() < (size_t)n_vocab) {
            cur.reserve(n_vocab);
        }

        if (cur.size() != (size_t)n_vocab) {
            cur.resize(n_vocab);
        }

        for (geniex_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = geniex_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = {
            /* .data     = */ cur.data(),
            /* .size     = */ cur.size(),
            /* .selected = */ -1,
            /* .sorted   = */ false,
        };
    }
};

static struct geniex_sampler *create_sampler_chain_from_params(const geniex_sampler_params &params,
                                                             geniex_vocab_interface *vocab) {
    auto sparam = geniex_sampler_chain_default_params();
    sparam.no_perf = params.no_perf;

    auto *chain = geniex_sampler_chain_init(sparam);

    // Add logit bias sampler first (if any biases exist)
    if (!params.logit_bias.empty()) {
        std::vector<geniex_logit_bias> biases;
        for (const auto &bias : params.logit_bias) {
            biases.push_back({bias.first, bias.second});
        }
        geniex_sampler_chain_add(
            chain, geniex_sampler_init_logit_bias(vocab ? vocab->n_tokens() : 32000, biases.size(), biases.data()));
    }

    if (params.mirostat == 0) {
        for (const auto &sampler_type : params.samplers) {
            switch (sampler_type) {
                case SAMPLER_TYPE_PENALTIES:
                    // Add penalties sampler
                    if (params.penalty_last_n > 0 &&
                        (params.penalty_repeat > 1.0f || params.penalty_freq > 0.0f || params.penalty_present > 0.0f)) {
                        geniex_sampler_chain_add(
                            chain, geniex_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat,
                                                               params.penalty_freq, params.penalty_present));
                    }
                    break;
                case SAMPLER_TYPE_DRY:
                    // Add DRY sampler — uses pre-tokenized breakers from params (no vocab needed)
                    if (params.dry_multiplier > 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0
                            && !params.dry_sequence_breaker_tokens.empty()) {
                        std::vector<const geniex_token*> seqs;
                        std::vector<size_t> lens;
                        seqs.reserve(params.dry_sequence_breaker_tokens.size());
                        lens.reserve(params.dry_sequence_breaker_tokens.size());
                        for (const auto& seq : params.dry_sequence_breaker_tokens) {
                            seqs.push_back(seq.data());
                            lens.push_back(seq.size());
                        }
                        geniex_sampler_chain_add(
                            chain,
                            geniex_sampler_init_dry_tokenized(
                                vocab ? vocab->n_tokens() : 32000,
                                params.dry_multiplier, params.dry_base,
                                params.dry_allowed_length, params.dry_penalty_last_n,
                                seqs.data(), lens.data(), seqs.size()));
                    }
                    break;
                case SAMPLER_TYPE_TOP_N_SIGMA:
                    // Add top-n-sigma sampler
                    if (params.top_n_sigma > 0.0f) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_top_n_sigma(params.top_n_sigma));
                    }
                    break;
                case SAMPLER_TYPE_TOP_K:
                    // Add top-k sampler
                    if (params.top_k > 0) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_top_k(params.top_k));
                    }
                    break;
                case SAMPLER_TYPE_TYPICAL_P:
                    // Add typical sampler
                    if (params.typical_p < 1.0f && params.typical_p > 0.0f) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_typical(params.typical_p, params.min_keep));
                    }
                    break;
                case SAMPLER_TYPE_TOP_P:
                    // Add top-p sampler
                    if (params.top_p < 1.0f && params.top_p > 0.0f) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_top_p(params.top_p, params.min_keep));
                    }
                    break;
                case SAMPLER_TYPE_MIN_P:
                    // Add min-p sampler
                    if (params.min_p > 0.0f && params.min_p < 1.0f) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_min_p(params.min_p, params.min_keep));
                    }
                    break;
                case SAMPLER_TYPE_XTC:
                    // Add XTC sampler
                    if (params.xtc_probability > 0.0f) {
                        geniex_sampler_chain_add(
                            chain, geniex_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep,
                                                         params.seed));
                    }
                    break;
                case SAMPLER_TYPE_TEMPERATURE:
                    // Add temperature sampler
                    if (params.dynatemp_range > 0.0f) {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_temp_ext(params.temp, params.dynatemp_range,
                                                                                 params.dynatemp_exponent));
                    } else {
                        geniex_sampler_chain_add(chain, geniex_sampler_init_temp(params.temp));
                    }
                    break;
            }
        }
        // Add distribution sampler as final step
        geniex_sampler_chain_add(chain, geniex_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        geniex_sampler_chain_add(chain, geniex_sampler_init_temp(params.temp));
        geniex_sampler_chain_add(chain, geniex_sampler_init_mirostat(vocab ? vocab->n_tokens() : 32000, params.seed,
                                                                 params.mirostat_tau, params.mirostat_eta, 100));

    } else if (params.mirostat == 2) {
        geniex_sampler_chain_add(chain, geniex_sampler_init_temp(params.temp));
        geniex_sampler_chain_add(chain,
                               geniex_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    }

    return chain;
}

std::string geniex_sampler_params::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
             "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
             "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
             "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = "
             "%.3f, top_n_sigma = %.3f, temp = %.3f\n"
             "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
             penalty_last_n, penalty_repeat, penalty_freq, penalty_present, dry_multiplier, dry_base,
             dry_allowed_length, dry_penalty_last_n, top_k, top_p, min_p, xtc_probability, xtc_threshold, typical_p,
             top_n_sigma, temp, mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

std::string geniex_sampler_context_print(const struct geniex_sampler_context *sctx) {
    std::string result = "logits ";

    for (int i = 0; i < geniex_sampler_chain_n(sctx->chain); i++) {
        const auto *smpl = geniex_sampler_chain_get(sctx->chain, i);
        result += std::string("-> ") + geniex_sampler_name(smpl) + " ";
    }

    return result;
}

struct geniex_sampler_context *geniex_sampler_init_context(const geniex_sampler_params &params, geniex_vocab_interface *vocab) {
    // Create grammar sampler if needed
    struct geniex_sampler *grammar = nullptr;
    if (!params.grammar_str.empty() && vocab) {
        grammar = geniex_sampler_init_grammar(vocab, params.grammar_str.c_str(), params.grammar_root.c_str());
    }

    // Create main sampler chain
    auto *chain = create_sampler_chain_from_params(params, vocab);
    if (!chain) {
        if (grammar) {
            geniex_sampler_free(grammar);
        }
        return nullptr;
    }

    // Create unified sampler
    auto *context = new geniex_sampler_context{
        /* .params  = */ params,
        /* .vocab   = */ vocab,
        /* .grammar = */ grammar,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<geniex_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return context;
}

struct geniex_sampler_context *geniex_sampler_context_clone(const struct geniex_sampler_context *sctx) {
    // create a new context
    auto *result = geniex_sampler_init_context(sctx->params, sctx->vocab);
    if (!result) return nullptr;

    // copy state
    result->prev = sctx->prev;
    result->cur = sctx->cur;
    result->cur_p = sctx->cur_p;

    // Fix pointer after copying
    if (!result->cur.empty()) {
        result->cur_p.data = result->cur.data();
    }

    return result;
}

void geniex_sampler_context_free(struct geniex_sampler_context *sctx) {
    if (!sctx) return;

    if (sctx->grammar) {
        geniex_sampler_free(sctx->grammar);
    }

    if (sctx->chain) {
        geniex_sampler_free(sctx->chain);
    }

    delete sctx;
}

void geniex_sampler_context_set_grammar(struct geniex_sampler_context *sctx, struct geniex_sampler *grammar_sampler) {
    if (!sctx) return;
    if (sctx->grammar) {
        geniex_sampler_free(sctx->grammar);
    }
    sctx->grammar = grammar_sampler;
}

void geniex_sampler_context_set_logits(struct geniex_sampler_context *sctx, const float *logits, int32_t n_vocab) {
    sctx->set_logits(logits, n_vocab);
}

geniex_token geniex_sampler_context_sample(struct geniex_sampler_context *sctx, const float *logits, int32_t n_vocab,
                                       bool grammar_first) {
    geniex_token id = geniex_sampler_context_sample_no_accept(sctx, logits, n_vocab, grammar_first);
    geniex_sampler_context_accept(sctx, id);
    return id;
}

geniex_token geniex_sampler_context_sample_no_accept(struct geniex_sampler_context *sctx, const float *logits,
                                                 int32_t n_vocab, bool grammar_first) {
    sctx->set_logits(logits, n_vocab);

    if (grammar_first && sctx->grammar) {
        geniex_sampler_apply(sctx->grammar, &sctx->cur_p);
    }

    // Apply main sampler chain
    geniex_sampler_apply(sctx->chain, &sctx->cur_p);

    assert((sctx->cur_p.selected != -1 && sctx->cur_p.selected < (int32_t)sctx->cur_p.size) &&
           "no selected token during sampling - check your sampling configuration");

    const geniex_token id = sctx->cur_p.data[sctx->cur_p.selected].id;

    if (grammar_first || (sctx->grammar == nullptr)) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        geniex_token_data single_token_data = {id, 1.0f, 0.0f};
        geniex_token_data_array single_token_data_array = {&single_token_data, 1, -1, false};

        geniex_sampler_apply(sctx->grammar, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    sctx->set_logits(logits, n_vocab);

    geniex_sampler_apply(sctx->grammar, &sctx->cur_p);
    geniex_sampler_apply(sctx->chain, &sctx->cur_p);

    assert((sctx->cur_p.selected != -1 && sctx->cur_p.selected < (int32_t)sctx->cur_p.size) &&
           "no selected token during re-sampling - check your sampling configuration");

    return sctx->cur_p.data[sctx->cur_p.selected].id;
}

void geniex_sampler_context_accept(struct geniex_sampler_context *sctx, geniex_token token) {
    if (!sctx) return;

    // Accept token in grammar
    if (sctx->grammar) {
        geniex_sampler_accept(sctx->grammar, token);
    }

    // Accept token in chain
    geniex_sampler_accept(sctx->chain, token);

    // Add to history
    sctx->prev.push_back(token);
}

void geniex_sampler_context_reset(struct geniex_sampler_context *sctx) {
    if (!sctx) return;

    if (sctx->grammar) {
        geniex_sampler_reset(sctx->grammar);
    }

    if (sctx->chain) {
        geniex_sampler_reset(sctx->chain);
    }

    sctx->prev.clear();
}

void geniex_perf_sampler_context_print(const struct geniex_sampler_context *sctx) { geniex_perf_sampler_print(sctx->chain); }

geniex_token_data_array *geniex_sampler_context_get_candidates(struct geniex_sampler_context *sctx) {
    if (!sctx) return nullptr;

    return &sctx->cur_p;
}

