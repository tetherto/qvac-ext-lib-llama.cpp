// test-vector-index.cpp - standalone C-API smoke test for the vector
// index. Exercises lifecycle, add, search, remove, contains, write, load,
// search-after-load. No model, no llama; only the new ggml-vector-index
// public C API.

#include "ggml-vector-index.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

constexpr int kDim = 4;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

std::vector<float> normalize(std::vector<float> v) {
    double sumsq = 0.0;
    for (float x : v) sumsq += static_cast<double>(x) * x;
    const float n = static_cast<float>(std::sqrt(sumsq));
    if (n > 0.0f) for (float & x : v) x /= n;
    return v;
}

float q8_dot_reference(const std::vector<float> & vector, const std::vector<float> & query) {
    CHECK(vector.size() == query.size());

    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;

    float acc = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int code = max_abs == 0.0f ?
            0 : static_cast<int>(std::nearbyint(vector[i] / scale));
        code = std::max(-127, std::min(127, code));
        acc += query[i] * (static_cast<float>(code) * scale);
    }
    return acc;
}

float q4_dot_reference(const std::vector<float> & vector, const std::vector<float> & query) {
    CHECK(vector.size() == query.size());

    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 7.0f;

    float acc = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int code = max_abs == 0.0f ?
            0 : static_cast<int>(std::nearbyint(vector[i] / scale));
        code = std::max(-7, std::min(7, code));
        acc += query[i] * (static_cast<float>(code) * scale);
    }
    return acc;
}

uint8_t read_file_byte(const std::string & path, std::streamoff offset) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    f.seekg(offset);
    char c = 0;
    f.read(&c, 1);
    CHECK(f.good());
    return static_cast<uint8_t>(c);
}

std::vector<uint8_t> read_file_bytes(const std::string & path) {
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

void write_file_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

void append_file_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::app);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

void append_u32_le(std::vector<uint8_t> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_u64_le(std::vector<uint8_t> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_f32_le(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32_le(bytes, bits);
}

void write_v1_index(
        const std::string & path,
        int dim,
        int bit_width,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));

    std::vector<uint8_t> bytes = { 'T', 'V', 'P', 'I', 1,
                                   static_cast<uint8_t>(bit_width), 0, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, static_cast<uint32_t>(ids.size()));
    for (float value : vectors) {
        append_f32_le(bytes, value);
    }
    for (uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    write_file_bytes(path, bytes);
}

template <typename Fn>
void expect_corrupt_load_fails(
        const std::string & source_path,
        const std::string & corrupt_path,
        Fn mutate) {
    std::vector<uint8_t> bytes = read_file_bytes(source_path);
    mutate(bytes);
    write_file_bytes(corrupt_path, bytes);

    auto * bad = ggml_vec_index_load(corrupt_path.c_str());
    CHECK(bad == nullptr);
    ggml_vec_index_free(bad);
    std::filesystem::remove(corrupt_path);
}

} // namespace

int main() {
    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);

    // Non-finite vectors are rejected without mutation.
    {
        const std::array<float, kDim> bad_vector = {
            1.0f, 0.0f, std::numeric_limits<float>::infinity(), 0.0f,
        };
        const uint64_t bad_id = 777ULL;
        CHECK(ggml_vec_index_add(idx, bad_vector.data(), 1, &bad_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Add 4 well-separated unit vectors. IDs are non-trivial uint64 to
    // catch sign-extension or BigInt round-trip bugs at the JS boundary
    // when this codepath is later exercised from Bare.
    std::vector<float> vecs;
    std::vector<uint64_t> ids = {
        42ULL,
        (1ULL << 40) + 7ULL,
        (1ULL << 62) + 11ULL,
        UINT64_MAX - 13ULL,
    };
    std::vector<std::vector<float>> seeds = {
        normalize({1.0f, 0.0f, 0.0f, 0.0f}),
        normalize({0.0f, 1.0f, 0.0f, 0.0f}),
        normalize({0.0f, 0.0f, 1.0f, 0.0f}),
        normalize({0.0f, 0.0f, 0.0f, 1.0f}),
    };
    for (const auto & s : seeds) {
        vecs.insert(vecs.end(), s.begin(), s.end());
    }
    CHECK(ggml_vec_index_add(
        idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(ggml_vec_index_contains(idx, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(idx, 999ULL) == 0);

    // Non-finite queries are rejected before search.
    {
        const std::array<float, kDim> bad_query = {
            1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f,
        };
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, bad_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

    // Duplicate add must fail without mutating state.
    {
        const std::vector<uint64_t> dup_ids = { ids[0] };
        std::vector<float> dup_vec(seeds[0]);
        CHECK(ggml_vec_index_add(idx, dup_vec.data(), 1, dup_ids.data())
              == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // In-batch duplicate ids must also fail atomically.
    {
        const uint64_t new_id = (1ULL << 50) + 123ULL;
        const std::vector<uint64_t> dup_ids = { new_id, new_id };
        std::vector<float> dup_vecs;
        dup_vecs.insert(dup_vecs.end(), seeds[0].begin(), seeds[0].end());
        dup_vecs.insert(dup_vecs.end(), seeds[1].begin(), seeds[1].end());
        CHECK(ggml_vec_index_add(idx, dup_vecs.data(), 2, dup_ids.data())
              == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
        CHECK(ggml_vec_index_contains(idx, new_id) == 0);
    }

    // IVF-flat ANN search is explicit and stale builds are rejected.
    {
        auto * ann = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(ann != nullptr);
        CHECK(ggml_vec_index_add(
            ann, vecs.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

        const std::vector<float> query = normalize({0.9f, 0.3f, 0.1f, -0.2f});
        std::array<float, 4> exact_scores{};
        std::array<float, 4> ann_scores{};
        std::array<uint64_t, 4> exact_ids{};
        std::array<uint64_t, 4> ann_ids{};

        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/0, /*n_iter=*/1)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/16, /*n_iter=*/3)
              == GGML_VEC_INDEX_OK);

        CHECK(ggml_vec_index_search(
            ann, query.data(), 1, /*k=*/4,
            exact_scores.data(), exact_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/4, /*nprobe=*/16,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        for (int i = 0; i < 4; ++i) {
            CHECK(ann_ids[i] == exact_ids[i]);
            CHECK(std::fabs(ann_scores[i] - exact_scores[i]) < 1e-5f);
        }

        CHECK(ggml_vec_index_search_ivf(
            ann, seeds[0].data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ann_ids[0] == ids[0]);

        const uint64_t ann_new_id = 9999991ULL;
        CHECK(ggml_vec_index_add(ann, seeds[3].data(), 1, &ann_new_id)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/16, /*n_iter=*/3)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/16,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(ann);
    }

    // Read-only APIs on one handle can run concurrently.
    {
        constexpr int n_rows = 16;
        std::vector<float> rows;
        std::vector<uint64_t> row_ids;
        rows.reserve(static_cast<size_t>(n_rows) * kDim);
        row_ids.reserve(n_rows);
        for (int row = 0; row < n_rows; ++row) {
            const std::vector<float> v = normalize({
                static_cast<float>((row % 5) - 2),
                static_cast<float>(((row + 1) % 7) - 3),
                static_cast<float>(((row * 3) % 11) - 5),
                1.0f,
            });
            rows.insert(rows.end(), v.begin(), v.end());
            row_ids.push_back(static_cast<uint64_t>(7000 + row));
        }

        auto * concurrent = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(concurrent != nullptr);
        CHECK(ggml_vec_index_add(concurrent, rows.data(), n_rows, row_ids.data())
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_build_ivf(concurrent, /*n_lists=*/4, /*n_iter=*/2)
              == GGML_VEC_INDEX_OK);

        const std::array<uint64_t, 5> allowed = {
            row_ids[0], row_ids[2], row_ids[4], row_ids[6], row_ids[8],
        };
        ggml_vec_index_filter_t * filter = ggml_vec_index_filter_create(
            concurrent, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);

        std::atomic<int> ready{ 0 };
        std::atomic<bool> start{ false };
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t]() {
                std::array<float, 4> query = {
                    rows[static_cast<size_t>((t % n_rows) * kDim + 0)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 1)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 2)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 3)],
                };
                std::array<float, 3> scores{};
                std::array<uint64_t, 3> out_ids{};
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                for (int iter = 0; iter < 200; ++iter) {
                    CHECK(ggml_vec_index_search(
                        concurrent, query.data(), 1, /*k=*/3,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(ggml_vec_index_search_filtered(
                        concurrent, query.data(), 1, /*k=*/3,
                        allowed.data(), static_cast<int>(allowed.size()),
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(ggml_vec_index_search_prepared_filtered(
                        concurrent, filter, query.data(), 1, /*k=*/3,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(ggml_vec_index_search_ivf(
                        concurrent, query.data(), 1, /*k=*/3, /*nprobe=*/4,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(ggml_vec_index_contains(concurrent, row_ids[static_cast<size_t>(t % n_rows)]) == 1);
                    CHECK(ggml_vec_index_len(concurrent) == n_rows);
                    CHECK(ggml_vec_index_dim(concurrent) == kDim);
                    CHECK(ggml_vec_index_bit_width(concurrent) == 8);
                }
            });
        }
        while (ready.load() != 8) {
            std::this_thread::yield();
        }
        start.store(true);
        for (std::thread & thread : threads) {
            thread.join();
        }

        ggml_vec_index_filter_free(filter);
        ggml_vec_index_free(concurrent);
    }

    // Top-1 of querying with each unit vector should retrieve itself with
    // score very close to 1.0 (full f32, no quantization noise).
    {
        std::array<float, 4> scores{};
        std::array<uint64_t, 4> out_ids{};
        for (size_t i = 0; i < seeds.size(); ++i) {
            CHECK(ggml_vec_index_search(
                idx, seeds[i].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == 0);
            CHECK(out_ids[0] == ids[i]);
            CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        }
    }

    // Top-k > len returns sentinel-padded tail.
    {
        std::array<float, 8> scores{};
        std::array<uint64_t, 8> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), 1, /*k=*/8,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        // Tail entries (positions 4..7) use sentinel score/id values.
        for (int i = 4; i < 8; ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
    }

    // Filtered search only considers ids present in the allowlist. Missing
    // and duplicate filter ids do not produce duplicate result rows.
    {
        const uint64_t missing_id = (1ULL << 60) + 99ULL;
        const std::array<uint64_t, 4> allowed = {
            ids[2], missing_id, ids[0], ids[0],
        };
        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/3,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[0]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[2] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/2,
            nullptr, /*n_allowed=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/1,
            nullptr, /*n_allowed=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        auto * filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[0]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[2] == -FLT_MAX);
        ggml_vec_index_filter_free(filter);

        auto * empty_filter = ggml_vec_index_filter_create(
            idx, nullptr, /*n_allowed=*/0);
        CHECK(empty_filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, empty_filter, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);
        ggml_vec_index_filter_free(empty_filter);

        CHECK(ggml_vec_index_filter_create(
            idx, nullptr, /*n_allowed=*/1) == nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, nullptr, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        auto * stale_filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(stale_filter != nullptr);
        const uint64_t stale_new_id = (1ULL << 60) + 100ULL;
        CHECK(ggml_vec_index_add(
            idx, seeds[3].data(), 1, &stale_new_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, stale_filter, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(idx, stale_new_id) == 1);
        ggml_vec_index_filter_free(stale_filter);
    }

    // Remove + search: the removed id must no longer surface.
    {
        CHECK(ggml_vec_index_remove(idx, ids[1]) == 1);
        CHECK(ggml_vec_index_remove(idx, ids[1]) == 0); // already gone
        CHECK(ggml_vec_index_len(idx) == 3);
        CHECK(ggml_vec_index_contains(idx, ids[1]) == 0);

        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, seeds[1].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == 0);
        for (int i = 0; i < 3; ++i) {
            CHECK(out_ids[i] != ids[1]);
        }
    }

    // Persistence round-trip: write, free, load, re-query.
    const auto tmp = std::filesystem::temp_directory_path() /
                     "ggml-vector-index-test.tvim";
    const std::string path = tmp.string();
    CHECK(ggml_vec_index_write(idx, path.c_str()) == 0);
#ifndef _WIN32
    CHECK(::chmod(path.c_str(), 0600) == 0);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == 0);
    struct stat persisted_stat;
    CHECK(::stat(path.c_str(), &persisted_stat) == 0);
    CHECK((persisted_stat.st_mode & 0777) == 0600);
#endif

    ggml_vec_index_free(idx);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_dim(loaded) == kDim);
    CHECK(ggml_vec_index_len(loaded) == 3);
    CHECK(ggml_vec_index_bit_width(loaded) == 32);
    CHECK(ggml_vec_index_contains(loaded, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[1]) == 0); // stayed deleted
    CHECK(ggml_vec_index_contains(loaded, ids[2]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[3]) == 1);

    // Top-1 self-match after reload.
    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            loaded, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
    }

    ggml_vec_index_free(loaded);
    std::filesystem::remove(path);

    // Incremental persistence: replay add/remove deltas on top of a snapshot.
    {
        const std::string snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-base.tvim").string();
        const std::string delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-log.tvid").string();
        const std::string missing_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-missing-delta-log.tvid").string();
        const std::string mismatched_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-mismatch.tvim").string();
        const std::string corrupt_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-corrupt.tvid").string();
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(missing_delta_path);
        std::filesystem::remove(mismatched_snapshot_path);
        std::filesystem::remove(corrupt_delta_path);

        auto * base = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(base != nullptr);
        std::vector<float> base_vecs;
        base_vecs.insert(base_vecs.end(), seeds[0].begin(), seeds[0].end());
        base_vecs.insert(base_vecs.end(), seeds[1].begin(), seeds[1].end());
        CHECK(ggml_vec_index_add(base, base_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

        auto * base_only = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), missing_delta_path.c_str());
        CHECK(base_only != nullptr);
        CHECK(ggml_vec_index_len(base_only) == 2);
        ggml_vec_index_free(base_only);

        const uint64_t delta_id = (1ULL << 41) + 7ULL;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &delta_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove_logged(
            base, ids[0], delta_path.c_str()) == 1);
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_DUPLICATE);

        auto * replayed = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed != nullptr);
        CHECK(ggml_vec_index_len(replayed) == 2);
        CHECK(ggml_vec_index_contains(replayed, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(replayed, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(replayed, delta_id) == 1);

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            replayed, seeds[2].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == delta_id);

        std::vector<uint8_t> corrupt_delta = read_file_bytes(delta_path);
        corrupt_delta[16 + 20] ^= 1; // state CRC is covered by record CRC
        write_file_bytes(corrupt_delta_path, corrupt_delta);
        auto * corrupt_delta_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(corrupt_delta_loaded == nullptr);
        ggml_vec_index_free(corrupt_delta_loaded);

        auto * mismatch = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(mismatch != nullptr);
        std::vector<float> mismatch_vecs;
        mismatch_vecs.insert(mismatch_vecs.end(), seeds[0].begin(), seeds[0].end());
        mismatch_vecs.insert(mismatch_vecs.end(), seeds[3].begin(), seeds[3].end());
        CHECK(ggml_vec_index_add(
            mismatch, mismatch_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(
            mismatch, mismatched_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * mismatch_loaded = ggml_vec_index_load_with_delta(
            mismatched_snapshot_path.c_str(), delta_path.c_str());
        CHECK(mismatch_loaded == nullptr);
        ggml_vec_index_free(mismatch_loaded);
        const uint64_t mismatch_new_id = (1ULL << 41) + 9ULL;
        CHECK(ggml_vec_index_add_logged(
            mismatch, seeds[2].data(), 1, &mismatch_new_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_IO);
        CHECK(ggml_vec_index_contains(mismatch, mismatch_new_id) == 0);
        ggml_vec_index_free(mismatch);

        const std::vector<uint8_t> pre_compact_delta = read_file_bytes(delta_path);
        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 16);

        auto * compacted = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(compacted != nullptr);
        CHECK(ggml_vec_index_len(compacted) == 2);
        CHECK(ggml_vec_index_contains(compacted, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(compacted, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(compacted, delta_id) == 1);
        ggml_vec_index_free(compacted);

        // Crash window: the compacted snapshot is durable but the old log
        // survived. Replay must remain idempotent.
        write_file_bytes(delta_path, pre_compact_delta);
        auto * compacted_with_old_log = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(compacted_with_old_log != nullptr);
        CHECK(ggml_vec_index_len(compacted_with_old_log) == 2);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, delta_id) == 1);
        ggml_vec_index_free(compacted_with_old_log);

        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
        const uint64_t post_compact_id = (1ULL << 41) + 8ULL;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[3].data(), 1, &post_compact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_compact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_compact != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_compact) == 3);
        CHECK(ggml_vec_index_contains(replayed_after_compact, post_compact_id) == 1);
        ggml_vec_index_free(replayed_after_compact);

        append_file_bytes(delta_path, { 0x01, 0x00, 0x00 });
        auto * replayed_with_torn_tail = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_with_torn_tail != nullptr);
        CHECK(ggml_vec_index_contains(replayed_with_torn_tail, delta_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_with_torn_tail, post_compact_id) == 1);

        ggml_vec_index_free(replayed_with_torn_tail);
        ggml_vec_index_free(replayed);
        ggml_vec_index_free(base);
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(mismatched_snapshot_path);
        std::filesystem::remove(corrupt_delta_path);
    }

    // v1 f32 snapshots migrate to q8 only for legacy bit_width=8; other
    // legacy widths, including bit_width=4, migrate to f32.
    {
        const std::vector<uint64_t> v1_ids = {
            (1ULL << 37) + 1ULL,
            (1ULL << 37) + 2ULL,
        };
        std::vector<float> v1_vectors;
        v1_vectors.insert(v1_vectors.end(), seeds[0].begin(), seeds[0].end());
        v1_vectors.insert(v1_vectors.end(), seeds[1].begin(), seeds[1].end());

        for (int bit_width : { 32, 8, 4 }) {
            const auto v1_tmp = std::filesystem::temp_directory_path() /
                                ("ggml-vector-index-v1-" + std::to_string(bit_width) + ".tvim");
            const std::string v1_path = v1_tmp.string();
            write_v1_index(v1_path, kDim, bit_width, v1_vectors, v1_ids);

            auto * v1 = ggml_vec_index_load(v1_path.c_str());
            CHECK(v1 != nullptr);
            CHECK(ggml_vec_index_dim(v1) == kDim);
            CHECK(ggml_vec_index_len(v1) == 2);
            CHECK(ggml_vec_index_bit_width(v1) == (bit_width == 8 ? 8 : 32));

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                v1, seeds[1].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == v1_ids[1]);
            CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);

            ggml_vec_index_free(v1);
            std::filesystem::remove(v1_path);
        }
    }

    // q8 score parity for a dimension that exercises the SIMD tail.
    {
        constexpr int tail_dim = 13;
        const std::vector<float> tail_vector = {
            -1.0f, 0.75f, -0.5f, 0.25f, 0.125f, -0.875f, 0.625f,
            -0.375f, 0.9f, -0.7f, 0.3f, -0.2f, 0.05f,
        };
        const std::vector<float> tail_query = {
            0.2f, -0.4f, 0.6f, -0.8f, 1.0f, 0.3f, -0.5f,
            0.7f, -0.9f, 0.11f, -0.22f, 0.33f, -0.44f,
        };
        const uint64_t tail_id = (1ULL << 55) + 321ULL;

        auto * tail_idx = ggml_vec_index_create(tail_dim, /*bit_width=*/8);
        CHECK(tail_idx != nullptr);
        CHECK(ggml_vec_index_add(
            tail_idx, tail_vector.data(), 1, &tail_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            tail_idx, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == tail_id);

        const float expected = q8_dot_reference(tail_vector, tail_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        ggml_vec_index_free(tail_idx);

        std::vector<float> zero_vector(tail_dim, 0.0f);
        auto * zero_idx = ggml_vec_index_create(tail_dim, /*bit_width=*/8);
        CHECK(zero_idx != nullptr);
        CHECK(ggml_vec_index_add(
            zero_idx, zero_vector.data(), 1, &tail_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            zero_idx, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(scores[0] == 0.0f);
        ggml_vec_index_free(zero_idx);
    }

    // Applying the q8 scale after accumulation can overflow even when the
    // dequantized dot product is finite.
    {
        constexpr int overflow_dim = 8;
        const std::vector<float> small_vector(overflow_dim, 1e-30f);
        const std::vector<float> large_query(overflow_dim, 1e38f);
        const uint64_t overflow_id = 123456789ULL;

        auto * overflow_idx = ggml_vec_index_create(overflow_dim, /*bit_width=*/8);
        CHECK(overflow_idx != nullptr);
        CHECK(ggml_vec_index_add(
            overflow_idx, small_vector.data(), 1, &overflow_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            overflow_idx, large_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        const float expected = static_cast<float>(
            overflow_dim * static_cast<double>(small_vector[0]) * large_query[0]);
        CHECK(out_ids[0] == overflow_id);
        CHECK(std::isfinite(scores[0]));
        CHECK(std::fabs(scores[0] - expected) <= std::fabs(expected) * 1e-5f);

        ggml_vec_index_free(overflow_idx);
    }

    // q4 production path: packed nibbles with one f32 scale per vector.
    {
        constexpr int tail_dim = 13;
        const std::vector<float> tail_vector = {
            -1.0f, 0.75f, -0.5f, 0.25f, 0.125f, -0.875f, 0.625f,
            -0.375f, 0.9f, -0.7f, 0.3f, -0.2f, 0.05f,
        };
        const std::vector<float> tail_query = {
            0.2f, -0.4f, 0.6f, -0.8f, 1.0f, 0.3f, -0.5f,
            0.7f, -0.9f, 0.11f, -0.22f, 0.33f, -0.44f,
        };
        const uint64_t q4_id = (1ULL << 55) + 654ULL;

        auto * q4 = ggml_vec_index_create(tail_dim, /*bit_width=*/4);
        CHECK(q4 != nullptr);
        CHECK(ggml_vec_index_bit_width(q4) == 4);
        CHECK(ggml_vec_index_add(q4, tail_vector.data(), 1, &q4_id) == GGML_VEC_INDEX_OK);

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            q4, tail_query.data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);

        const float expected = q4_dot_reference(tail_vector, tail_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        const std::string q4_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q4-test.tvim").string();
        const std::string corrupt_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q4-corrupt-test.tvim").string();
        CHECK(ggml_vec_index_write(q4, q4_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(read_file_byte(q4_path, 4) == 2); // .tvim v2
        CHECK(read_file_byte(q4_path, 5) == 4); // q4 bit width
        CHECK(read_file_byte(q4_path, 6) == 3); // q4 storage kind
        CHECK(read_file_byte(q4_path, 24) == 0); // packed components

        auto * q4_loaded = ggml_vec_index_load(q4_path.c_str());
        CHECK(q4_loaded != nullptr);
        CHECK(ggml_vec_index_bit_width(q4_loaded) == 4);
        CHECK(ggml_vec_index_len(q4_loaded) == 1);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search(
            q4_loaded, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        constexpr size_t q4_vector_offset = 32 + sizeof(float);
        constexpr size_t q4_row_bytes = (tail_dim + 1) / 2;
        expect_corrupt_load_fails(q4_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[q4_vector_offset] =
                static_cast<uint8_t>(bytes[q4_vector_offset] & 0xf0u); // low nibble 0 is invalid
        });
        expect_corrupt_load_fails(q4_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            const size_t tail_byte = q4_vector_offset + q4_row_bytes - 1;
            bytes[tail_byte] =
                static_cast<uint8_t>((bytes[tail_byte] & 0x0fu) | 0x90u); // odd tail high nibble must be zero-code
        });

        ggml_vec_index_free(q4_loaded);
        ggml_vec_index_free(q4);
        std::filesystem::remove(q4_path);
    }

    // q4 parity for a dimension that exercises the optimized loop and tail.
    {
        constexpr int q4_dim = 33;
        std::vector<float> q4_vector(q4_dim);
        std::vector<float> q4_query(q4_dim);
        for (int i = 0; i < q4_dim; ++i) {
            q4_vector[static_cast<size_t>(i)] =
                static_cast<float>((i % 11) - 5) / 5.0f;
            q4_query[static_cast<size_t>(i)] =
                static_cast<float>(((i * 7) % 13) - 6) / 7.0f;
        }
        const uint64_t q4_id = (1ULL << 56) + 123ULL;
        auto * q4 = ggml_vec_index_create(q4_dim, /*bit_width=*/4);
        CHECK(q4 != nullptr);
        CHECK(ggml_vec_index_add(q4, q4_vector.data(), 1, &q4_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            q4, q4_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);

        const float expected = q4_dot_reference(q4_vector, q4_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);
        ggml_vec_index_free(q4);
    }

    // Delta replay keeps q8 storage q8 and reuses the normal q8 quantization path.
    {
        const std::string snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q8-delta-base.tvim").string();
        const std::string delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q8-delta-log.tvid").string();
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);

        const uint64_t base_id = (1ULL << 42) + 1ULL;
        const uint64_t delta_id = (1ULL << 42) + 2ULL;
        auto * q8_delta = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(q8_delta != nullptr);
        CHECK(ggml_vec_index_add(
            q8_delta, seeds[0].data(), 1, &base_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(q8_delta, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add_logged(
            q8_delta, seeds[3].data(), 1, &delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);

        auto * replayed_q8 = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_q8 != nullptr);
        CHECK(ggml_vec_index_bit_width(replayed_q8) == 8);
        CHECK(ggml_vec_index_len(replayed_q8) == 2);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            replayed_q8, seeds[3].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == delta_id);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);

        CHECK(ggml_vec_index_compact_delta(
            q8_delta, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 16);
        auto * compacted_q8 = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(compacted_q8 != nullptr);
        CHECK(ggml_vec_index_bit_width(compacted_q8) == 8);
        CHECK(ggml_vec_index_len(compacted_q8) == 2);
        CHECK(ggml_vec_index_contains(compacted_q8, delta_id) == 1);

        ggml_vec_index_free(compacted_q8);
        ggml_vec_index_free(replayed_q8);
        ggml_vec_index_free(q8_delta);
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
    }

    // Header metadata is protected even when all payload sections are empty.
    {
        auto * empty_idx = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(empty_idx != nullptr);
        const std::string empty_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-empty-test.tvim").string();
        const std::string corrupt_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-empty-corrupt-test.tvim").string();
        CHECK(ggml_vec_index_write(empty_idx, empty_path.c_str()) == GGML_VEC_INDEX_OK);
        expect_corrupt_load_fails(
            empty_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[8] += 1; // valid dimension change must fail the header CRC
            });
        ggml_vec_index_free(empty_idx);
        std::filesystem::remove(empty_path);
    }

    // q8 production path: stores quantized codes, searches directly against
    // q8 storage, and persists as .tvim v2 with q8 metadata.
    {
        auto * q8 = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(q8 != nullptr);
        CHECK(ggml_vec_index_dim(q8) == kDim);
        CHECK(ggml_vec_index_bit_width(q8) == 8);

        const std::vector<uint64_t> q8_ids = {
            (1ULL << 33) + 99ULL,
            (1ULL << 48) + 77ULL,
        };
        std::vector<float> q8_vecs;
        q8_vecs.insert(q8_vecs.end(), seeds[0].begin(), seeds[0].end());
        q8_vecs.insert(q8_vecs.end(), seeds[2].begin(), seeds[2].end());
        CHECK(ggml_vec_index_add(q8, q8_vecs.data(), 2, q8_ids.data()) == 0);

        std::array<float, 4> scores{};
        std::array<uint64_t, 4> out_ids{};
        CHECK(ggml_vec_index_search(
            q8, seeds[2].data(), 1, /*k=*/4,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == q8_ids[1]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        CHECK(scores[2] == -FLT_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[3] == -FLT_MAX);
        CHECK(out_ids[3] == UINT64_MAX);

        const uint64_t q8_missing_id = (1ULL << 59) + 17ULL;
        const std::array<uint64_t, 3> q8_allowed = {
            q8_missing_id, q8_ids[0], q8_ids[0],
        };
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_filtered(
            q8, seeds[2].data(), 1, /*k=*/3,
            q8_allowed.data(), static_cast<int>(q8_allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(scores[2] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            q8, seeds[2].data(), 1, /*k=*/2,
            nullptr, /*n_allowed=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);

        auto * q8_filter = ggml_vec_index_filter_create(
            q8, q8_allowed.data(), static_cast<int>(q8_allowed.size()));
        CHECK(q8_filter != nullptr);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_prepared_filtered(
            q8, q8_filter, seeds[2].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(scores[2] == -FLT_MAX);
        ggml_vec_index_filter_free(q8_filter);

        const auto q8_tmp = std::filesystem::temp_directory_path() /
                            "ggml-vector-index-q8-test.tvim";
        const std::string q8_path = q8_tmp.string();
        CHECK(ggml_vec_index_write(q8, q8_path.c_str()) == 0);
        CHECK(read_file_byte(q8_path, 4) == 2); // .tvim v2
        CHECK(read_file_byte(q8_path, 5) == 8); // q8 bit width
        CHECK(read_file_byte(q8_path, 6) == 2); // q8 storage kind
        CHECK(read_file_byte(q8_path, 7) == 1); // checksum trailer present

        const auto corrupt_tmp = std::filesystem::temp_directory_path() /
                                 "ggml-vector-index-corrupt-test.tvim";
        const std::string corrupt_path = corrupt_tmp.string();

        // Legacy v2 files without a checksum remain readable.
        const auto legacy_v2_tmp = std::filesystem::temp_directory_path() /
                                   "ggml-vector-index-legacy-v2-test.tvim";
        const std::string legacy_v2_path = legacy_v2_tmp.string();
        std::vector<uint8_t> legacy_v2 = read_file_bytes(q8_path);
        legacy_v2[7] = 0;
        legacy_v2.resize(legacy_v2.size() - 4 * sizeof(uint32_t));
        write_file_bytes(legacy_v2_path, legacy_v2);
        auto * legacy_loaded = ggml_vec_index_load(legacy_v2_path.c_str());
        CHECK(legacy_loaded != nullptr);
        CHECK(ggml_vec_index_len(legacy_loaded) == 2);
        ggml_vec_index_free(legacy_loaded);
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[32] = 0;
                bytes[33] = 0;
                bytes[34] = 0;
                bytes[35] = 0; // q8 scale must be positive and finite
            });
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[40] = 0x80; // q8 codes are restricted to [-127, 127]
            });
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                const size_t id_offset =
                    32 + 2 * sizeof(float) + 2 * kDim * sizeof(int8_t);
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    bytes[id_offset + sizeof(uint64_t) + i] = bytes[id_offset + i];
                }
            });
        std::filesystem::remove(legacy_v2_path);

        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[0] = 'X';
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[4] = 99; // unsupported version
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[6] = 1; // storage kind does not match bit_width=8
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[7] |= 0x80; // unknown flags are rejected
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[16] = 0; // qparam_type must be scale-f32 for q8
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[28] = 1; // reserved u32 must be zero
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[32] ^= 1; // q8 scale payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[40] ^= 1; // q8 code payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[48] ^= 1; // id payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 16] ^= 1; // header checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 12] ^= 1; // qparams checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 8] ^= 1; // vectors checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 4] ^= 1; // ids checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(35); // truncated q8 scales
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(43); // truncated q8 codes
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(bytes.size() - 1); // truncated ids
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.push_back(0); // trailing data is not part of the declared file
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[12] = 0xff;
            bytes[13] = 0xff;
            bytes[14] = 0xff;
            bytes[15] = 0xff; // impossible vector count for this file size
        });
        ggml_vec_index_free(q8);

        auto * q8_loaded = ggml_vec_index_load(q8_path.c_str());
        CHECK(q8_loaded != nullptr);
        CHECK(ggml_vec_index_dim(q8_loaded) == kDim);
        CHECK(ggml_vec_index_len(q8_loaded) == 2);
        CHECK(ggml_vec_index_bit_width(q8_loaded) == 8);
        CHECK(ggml_vec_index_contains(q8_loaded, q8_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(q8_loaded, q8_ids[1]) == 1);

        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search(
            q8_loaded, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);

        ggml_vec_index_free(q8_loaded);
        std::filesystem::remove(q8_path);
    }

    std::printf("test-vector-index: OK\n");
    return 0;
}
