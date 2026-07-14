// bench-vector-index.cpp - f32 vs q8 vector-index quality and latency smoke bench.

#include "ggml-vector-index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    int delta_ops = 256;
    int ivf_lists = 64;
    int ivf_iters = 4;
    int ivf_nprobe = 4;
};

struct TimedSearch {
    double ms = 0.0;
    std::vector<float> scores;
    std::vector<uint64_t> ids;
};

template <typename Fn>
double median_time_ms(int warmups, int repeats, Fn fn) {
    for (int i = 0; i < warmups; ++i) {
        fn();
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

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

const char * q4_kernel_name() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "arm-neon";
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

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search(
            idx,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_ivf_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int nprobe,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_ivf(
            idx,
            queries.data(),
            n_query,
            k,
            nprobe,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_filtered_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        const std::vector<uint64_t> & allowed_ids,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_filtered(
            idx,
            queries.data(),
            n_query,
            k,
            allowed_ids.data(),
            static_cast<int>(allowed_ids.size()),
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_prepared_filtered_search(
        const ggml_vec_index_t * idx,
        const ggml_vec_index_filter_t * filter,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx,
            filter,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
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

std::vector<uint8_t> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    const auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        CHECK(f.gcount() == static_cast<std::streamsize>(bytes.size()));
    }
    return bytes;
}

void write_file_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

std::vector<uint64_t> make_allowlist(int n_vec, int n_allowed) {
    std::vector<uint64_t> allowed;
    allowed.reserve(static_cast<size_t>(n_allowed));
    for (int i = 0; i < n_allowed; ++i) {
        const int row = (i * 37) % n_vec;
        allowed.push_back(static_cast<uint64_t>(row) + 1);
    }
    std::sort(allowed.begin(), allowed.end());
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return allowed;
}

double recall_against(const TimedSearch & exact, const TimedSearch & candidate, int n_query, int k) {
    int overlap = 0;
    for (int q = 0; q < n_query; ++q) {
        std::unordered_set<uint64_t> exact_topk;
        for (int j = 0; j < k; ++j) {
            exact_topk.insert(exact.ids[static_cast<size_t>(q) * k + j]);
        }
        for (int j = 0; j < k; ++j) {
            const uint64_t id = candidate.ids[static_cast<size_t>(q) * k + j];
            if (exact_topk.count(id) != 0) {
                ++overlap;
            }
        }
    }
    return static_cast<double>(overlap) / static_cast<double>(n_query * k);
}

struct DeltaBenchResult {
    double snapshot_load_ms = 0.0;
    double replay_load_ms = 0.0;
    double compact_ms = 0.0;
    double post_compact_load_ms = 0.0;
    uintmax_t snapshot_bytes_before = 0;
    uintmax_t delta_bytes_before = 0;
    uintmax_t snapshot_bytes_after = 0;
    uintmax_t delta_bytes_after = 0;
};

DeltaBenchResult run_delta_bench(
        int bit_width,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids,
        const BenchConfig & cfg,
        const char * snapshot_name,
        const char * delta_name) {
    CHECK(cfg.delta_ops > 0 && cfg.delta_ops * 2 < cfg.n_vec);
    const int base_n = cfg.n_vec - cfg.delta_ops;
    const std::filesystem::path snapshot_path =
        std::filesystem::temp_directory_path() / snapshot_name;
    const std::filesystem::path delta_path =
        std::filesystem::temp_directory_path() / delta_name;
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);

    ggml_vec_index_t * idx = ggml_vec_index_create(cfg.dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), base_n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.string().c_str()) == GGML_VEC_INDEX_OK);

    for (int i = 0; i < cfg.delta_ops; ++i) {
        const int row = base_n + i;
        CHECK(ggml_vec_index_add_logged(
            idx,
            vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(cfg.dim),
            1,
            ids.data() + row,
            delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
    }
    for (int i = 0; i < cfg.delta_ops / 2; ++i) {
        CHECK(ggml_vec_index_remove_logged(
            idx,
            ids[static_cast<size_t>(i)],
            delta_path.string().c_str()) == 1);
    }

    DeltaBenchResult result;
    result.snapshot_bytes_before = std::filesystem::file_size(snapshot_path);
    result.delta_bytes_before = std::filesystem::file_size(delta_path);
    const std::vector<uint8_t> dirty_delta = read_file_bytes(delta_path);

    result.snapshot_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load(snapshot_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    result.replay_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_with_delta(
            snapshot_path.string().c_str(),
            delta_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    result.compact_ms = median_time_ms(0, cfg.repeats, [&]() {
        write_file_bytes(delta_path, dirty_delta);
        CHECK(ggml_vec_index_compact_delta(
            idx,
            snapshot_path.string().c_str(),
            delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
    });
    result.snapshot_bytes_after = std::filesystem::file_size(snapshot_path);
    result.delta_bytes_after = std::filesystem::file_size(delta_path);
    result.post_compact_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_with_delta(
            snapshot_path.string().c_str(),
            delta_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });

    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    return result;
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
    ggml_vec_index_t * q4  = ggml_vec_index_create(cfg.dim, 4);
    CHECK(f32 != nullptr);
    CHECK(q8 != nullptr);
    CHECK(q4 != nullptr);
    CHECK(ggml_vec_index_add(f32, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q8,  vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q4,  vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);

    const double f32_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(f32, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double q8_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(q8, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double q4_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(q4, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });

    const TimedSearch f32_res = run_search(
        f32, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q8_res = run_search(
        q8, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q4_res = run_search(
        q4, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch f32_ivf_res = run_ivf_search(
        f32, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch q8_ivf_res = run_ivf_search(
        q8, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch q4_ivf_res = run_ivf_search(
        q4, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);

    const std::vector<int> filter_sizes = {
        32,
        128,
        512,
        cfg.n_vec / 2,
    };
    std::vector<std::vector<uint64_t>> allowlists;
    std::vector<TimedSearch> f32_filtered;
    std::vector<TimedSearch> q8_filtered;
    std::vector<TimedSearch> f32_prepared_filtered;
    std::vector<TimedSearch> q8_prepared_filtered;
    for (int requested : filter_sizes) {
        const int n_allowed = std::min(requested, cfg.n_vec);
        allowlists.push_back(make_allowlist(cfg.n_vec, n_allowed));
        f32_filtered.push_back(run_filtered_search(
            f32,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        q8_filtered.push_back(run_filtered_search(
            q8,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        ggml_vec_index_filter_t * f32_filter = ggml_vec_index_filter_create(
            f32, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        ggml_vec_index_filter_t * q8_filter = ggml_vec_index_filter_create(
            q8, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        CHECK(f32_filter != nullptr);
        CHECK(q8_filter != nullptr);
        f32_prepared_filtered.push_back(run_prepared_filtered_search(
            f32,
            f32_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        q8_prepared_filtered.push_back(run_prepared_filtered_search(
            q8,
            q8_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        ggml_vec_index_filter_free(f32_filter);
        ggml_vec_index_filter_free(q8_filter);
    }

    double q8_mean_abs_drift = 0.0;
    double q8_max_abs_drift = 0.0;
    double q4_mean_abs_drift = 0.0;
    double q4_max_abs_drift = 0.0;
    int q8_overlap = 0;
    int q4_overlap = 0;
    int drift_count = 0;

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
                ++q8_overlap;
            }
            const float exact = dot_exact(vectors, query, id, cfg.dim);
            const double drift = std::fabs(static_cast<double>(exact) - q8_res.scores[pos]);
            q8_mean_abs_drift += drift;
            q8_max_abs_drift = std::max(q8_max_abs_drift, drift);

            const uint64_t q4_id = q4_res.ids[pos];
            if (f32_topk.count(q4_id) != 0) {
                ++q4_overlap;
            }
            const float q4_exact = dot_exact(vectors, query, q4_id, cfg.dim);
            const double q4_drift = std::fabs(static_cast<double>(q4_exact) - q4_res.scores[pos]);
            q4_mean_abs_drift += q4_drift;
            q4_max_abs_drift = std::max(q4_max_abs_drift, q4_drift);
            ++drift_count;
        }
    }
    q8_mean_abs_drift /= static_cast<double>(drift_count);
    q4_mean_abs_drift /= static_cast<double>(drift_count);
    const double q8_recall_at_k = static_cast<double>(q8_overlap) /
                                  static_cast<double>(cfg.n_query * cfg.k);
    const double q4_recall_at_k = static_cast<double>(q4_overlap) /
                                  static_cast<double>(cfg.n_query * cfg.k);
    const double f32_ivf_recall_at_k =
        recall_against(f32_res, f32_ivf_res, cfg.n_query, cfg.k);
    const double q8_ivf_recall_at_k =
        recall_against(f32_res, q8_ivf_res, cfg.n_query, cfg.k);
    const double q4_ivf_recall_at_k =
        recall_against(f32_res, q4_ivf_res, cfg.n_query, cfg.k);

    const auto f32_path = write_index_file(f32, "ggml-vector-index-bench-f32.tvim");
    const auto q8_path  = write_index_file(q8,  "ggml-vector-index-bench-q8.tvim");
    const auto q4_path  = write_index_file(q4,  "ggml-vector-index-bench-q4.tvim");
    const uintmax_t f32_file_size = std::filesystem::file_size(f32_path);
    const uintmax_t q8_file_size  = std::filesystem::file_size(q8_path);
    const uintmax_t q4_file_size  = std::filesystem::file_size(q4_path);
    std::filesystem::remove(f32_path);
    std::filesystem::remove(q8_path);
    std::filesystem::remove(q4_path);

    const DeltaBenchResult f32_delta = run_delta_bench(
        32,
        vectors,
        ids,
        cfg,
        "ggml-vector-index-bench-delta-f32.tvim",
        "ggml-vector-index-bench-delta-f32.tvid");
    const DeltaBenchResult q8_delta = run_delta_bench(
        8,
        vectors,
        ids,
        cfg,
        "ggml-vector-index-bench-delta-q8.tvim",
        "ggml-vector-index-bench-delta-q8.tvid");
    const DeltaBenchResult q4_delta = run_delta_bench(
        4,
        vectors,
        ids,
        cfg,
        "ggml-vector-index-bench-delta-q4.tvim",
        "ggml-vector-index-bench-delta-q4.tvid");

    const size_t f32_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t q8_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(int8_t) +
        static_cast<size_t>(cfg.n_vec) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t q4_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * ((static_cast<size_t>(cfg.dim) + 1) / 2) +
        static_cast<size_t>(cfg.n_vec) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);

    std::printf("bench-vector-index\n");
    std::printf("  q8 kernel=%s\n", q8_kernel_name());
    std::printf("  q4 kernel=%s\n", q4_kernel_name());
    std::printf("  n_vec=%d dim=%d n_query=%d k=%d warmups=%d repeats=%d\n",
        cfg.n_vec, cfg.dim, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    std::printf("  estimated memory: f32=%zu bytes q8=%zu bytes q4=%zu bytes q8/f32=%.3f q4/f32=%.3f\n",
        f32_memory_bytes, q8_memory_bytes, q4_memory_bytes,
        static_cast<double>(q8_memory_bytes) / static_cast<double>(f32_memory_bytes),
        static_cast<double>(q4_memory_bytes) / static_cast<double>(f32_memory_bytes));
    std::printf("  file size:        f32=%llu bytes q8=%llu bytes q4=%llu bytes q8/f32=%.3f q4/f32=%.3f\n",
        static_cast<unsigned long long>(f32_file_size),
        static_cast<unsigned long long>(q8_file_size),
        static_cast<unsigned long long>(q4_file_size),
        static_cast<double>(q8_file_size) / static_cast<double>(f32_file_size),
        static_cast<double>(q4_file_size) / static_cast<double>(f32_file_size));
    std::printf("  median latency:   f32=%.3f ms q8=%.3f ms q4=%.3f ms q8/f32=%.3f q4/f32=%.3f\n",
        f32_res.ms, q8_res.ms, q4_res.ms, q8_res.ms / f32_res.ms, q4_res.ms / f32_res.ms);
    std::printf(
        "  ivf build:        lists=%d iters=%d f32=%.3f ms q8=%.3f ms q4=%.3f ms\n",
        cfg.ivf_lists,
        cfg.ivf_iters,
        f32_ivf_build_ms,
        q8_ivf_build_ms,
        q4_ivf_build_ms);
    std::printf(
        "  ivf latency:      nprobe=%d f32=%.3f ms q8=%.3f ms q4=%.3f ms f32/full=%.3f q8/full=%.3f q4/full=%.3f\n",
        cfg.ivf_nprobe,
        f32_ivf_res.ms,
        q8_ivf_res.ms,
        q4_ivf_res.ms,
        f32_ivf_res.ms / f32_res.ms,
        q8_ivf_res.ms / q8_res.ms,
        q4_ivf_res.ms / q4_res.ms);
    std::printf(
        "  ivf recall:       f32@%d=%.4f q8@%d=%.4f q4@%d=%.4f against exact f32\n",
        cfg.k,
        f32_ivf_recall_at_k,
        cfg.k,
        q8_ivf_recall_at_k,
        cfg.k,
        q4_ivf_recall_at_k);
    for (size_t i = 0; i < allowlists.size(); ++i) {
        std::printf(
            "  filtered latency: allowed=%zu f32=%.3f ms q8=%.3f ms f32/prepared=%.3f ms q8/prepared=%.3f ms\n",
            allowlists[i].size(),
            f32_filtered[i].ms,
            q8_filtered[i].ms,
            f32_prepared_filtered[i].ms,
            q8_prepared_filtered[i].ms);
        std::printf(
            "  filtered ratio:   allowed=%zu f32/full=%.3f q8/full=%.3f f32/prep_speedup=%.3f q8/prep_speedup=%.3f\n",
            allowlists[i].size(),
            f32_filtered[i].ms / f32_res.ms,
            q8_filtered[i].ms / q8_res.ms,
            f32_filtered[i].ms / f32_prepared_filtered[i].ms,
            q8_filtered[i].ms / q8_prepared_filtered[i].ms);
    }
    std::printf(
        "  delta load f32:    snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        f32_delta.snapshot_load_ms,
        f32_delta.replay_load_ms,
        f32_delta.compact_ms,
        f32_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes f32:   snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(f32_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(f32_delta.delta_bytes_before),
        static_cast<unsigned long long>(f32_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(f32_delta.delta_bytes_after));
    std::printf(
        "  delta load q8:     snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        q8_delta.snapshot_load_ms,
        q8_delta.replay_load_ms,
        q8_delta.compact_ms,
        q8_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes q8:    snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(q8_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(q8_delta.delta_bytes_before),
        static_cast<unsigned long long>(q8_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(q8_delta.delta_bytes_after));
    std::printf(
        "  delta load q4:     snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        q4_delta.snapshot_load_ms,
        q4_delta.replay_load_ms,
        q4_delta.compact_ms,
        q4_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes q4:    snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(q4_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(q4_delta.delta_bytes_before),
        static_cast<unsigned long long>(q4_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(q4_delta.delta_bytes_after));
    std::printf("  quality q8:       recall@%d=%.4f mean_abs_score_drift=%.6f max_abs_score_drift=%.6f\n",
        cfg.k, q8_recall_at_k, q8_mean_abs_drift, q8_max_abs_drift);
    std::printf("  quality q4:       recall@%d=%.4f mean_abs_score_drift=%.6f max_abs_score_drift=%.6f\n",
        cfg.k, q4_recall_at_k, q4_mean_abs_drift, q4_max_abs_drift);

    ggml_vec_index_free(f32);
    ggml_vec_index_free(q8);
    ggml_vec_index_free(q4);
    return 0;
}
