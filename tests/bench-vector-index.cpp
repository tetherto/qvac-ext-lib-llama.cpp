// bench-vector-index.cpp - f32 vs q8 vector-index quality and latency smoke bench.

#include "ggml-vector-index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct BenchConfig {
    int n_vec   = 2048;
    int dim     = 256;
    int n_query = 64;
    int k       = 10;
    int warmups = 2;
    int repeats = 7;
};

struct TimedSearch {
    double ms = 0.0;
    std::vector<float> scores;
    std::vector<uint64_t> ids;
};

const char * q8_kernel_name() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "arm-neon";
#elif defined(__AVX2__)
    return "avx2";
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

std::vector<float> make_normalized_vectors(int n, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int row = 0; row < n; ++row) {
        float norm2 = 0.0f;
        float * v = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int i = 0; i < dim; ++i) {
            v[i] = dist(rng);
            norm2 += v[i] * v[i];
        }
        const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (int i = 0; i < dim; ++i) {
            v[i] *= inv_norm;
        }
    }
    return vectors;
}

TimedSearch run_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    for (int i = 0; i < warmups; ++i) {
        CHECK(ggml_vec_index_search(
            idx,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ggml_vec_index_search(
            idx,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    result.ms = times[times.size() / 2];
    return result;
}

float dot_exact(
        const std::vector<float> & vectors,
        const float * query,
        uint64_t id,
        int dim) {
    const size_t row = static_cast<size_t>(id - 1);
    const float * v = vectors.data() + row * static_cast<size_t>(dim);
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += query[i] * v[i];
    }
    return acc;
}

std::filesystem::path write_index_file(ggml_vec_index_t * idx, const char * name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    CHECK(ggml_vec_index_write(idx, path.string().c_str()) == GGML_VEC_INDEX_OK);
    return path;
}

} // namespace

int main() {
    const BenchConfig cfg;

    std::vector<float> vectors = make_normalized_vectors(cfg.n_vec, cfg.dim, 0xdeadbeef);
    std::vector<float> queries = make_normalized_vectors(cfg.n_query, cfg.dim, 0xc001d00d);
    std::vector<uint64_t> ids(static_cast<size_t>(cfg.n_vec));
    for (int i = 0; i < cfg.n_vec; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i) + 1;
    }

    ggml_vec_index_t * f32 = ggml_vec_index_create(cfg.dim, 32);
    ggml_vec_index_t * q8  = ggml_vec_index_create(cfg.dim, 8);
    CHECK(f32 != nullptr);
    CHECK(q8 != nullptr);
    CHECK(ggml_vec_index_add(f32, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q8,  vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);

    const TimedSearch f32_res = run_search(
        f32, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q8_res = run_search(
        q8, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);

    double mean_abs_drift = 0.0;
    double max_abs_drift = 0.0;
    int drift_count = 0;
    int overlap = 0;

    for (int q = 0; q < cfg.n_query; ++q) {
        std::unordered_set<uint64_t> f32_topk;
        for (int j = 0; j < cfg.k; ++j) {
            f32_topk.insert(f32_res.ids[static_cast<size_t>(q) * cfg.k + j]);
        }

        const float * query = queries.data() + static_cast<size_t>(q) * static_cast<size_t>(cfg.dim);
        for (int j = 0; j < cfg.k; ++j) {
            const size_t pos = static_cast<size_t>(q) * static_cast<size_t>(cfg.k) + static_cast<size_t>(j);
            const uint64_t id = q8_res.ids[pos];
            if (f32_topk.count(id) != 0) {
                ++overlap;
            }
            const float exact = dot_exact(vectors, query, id, cfg.dim);
            const double drift = std::fabs(static_cast<double>(exact) - q8_res.scores[pos]);
            mean_abs_drift += drift;
            max_abs_drift = std::max(max_abs_drift, drift);
            ++drift_count;
        }
    }
    mean_abs_drift /= static_cast<double>(drift_count);
    const double recall_at_k = static_cast<double>(overlap) /
                               static_cast<double>(cfg.n_query * cfg.k);

    const auto f32_path = write_index_file(f32, "ggml-vector-index-bench-f32.tvim");
    const auto q8_path  = write_index_file(q8,  "ggml-vector-index-bench-q8.tvim");
    const uintmax_t f32_file_size = std::filesystem::file_size(f32_path);
    const uintmax_t q8_file_size  = std::filesystem::file_size(q8_path);
    std::filesystem::remove(f32_path);
    std::filesystem::remove(q8_path);

    const size_t f32_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t q8_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(int8_t) +
        static_cast<size_t>(cfg.n_vec) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);

    std::printf("bench-vector-index\n");
    std::printf("  q8 kernel=%s\n", q8_kernel_name());
    std::printf("  n_vec=%d dim=%d n_query=%d k=%d warmups=%d repeats=%d\n",
        cfg.n_vec, cfg.dim, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    std::printf("  estimated memory: f32=%zu bytes q8=%zu bytes ratio=%.3f\n",
        f32_memory_bytes, q8_memory_bytes,
        static_cast<double>(q8_memory_bytes) / static_cast<double>(f32_memory_bytes));
    std::printf("  file size:        f32=%llu bytes q8=%llu bytes ratio=%.3f\n",
        static_cast<unsigned long long>(f32_file_size),
        static_cast<unsigned long long>(q8_file_size),
        static_cast<double>(q8_file_size) / static_cast<double>(f32_file_size));
    std::printf("  median latency:   f32=%.3f ms q8=%.3f ms ratio=%.3f\n",
        f32_res.ms, q8_res.ms, q8_res.ms / f32_res.ms);
    std::printf("  quality:          recall@%d=%.4f mean_abs_score_drift=%.6f max_abs_score_drift=%.6f\n",
        cfg.k, recall_at_k, mean_abs_drift, max_abs_drift);

    ggml_vec_index_free(f32);
    ggml_vec_index_free(q8);
    return 0;
}
