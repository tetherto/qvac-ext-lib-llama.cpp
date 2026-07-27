// test-vector-index.cpp - standalone C-API smoke test for the POC vector
// index. Exercises lifecycle, add, search, remove, contains, write, load,
// search-after-load. No model, no llama; only the new ggml-vector-index
// public C API.

#include "ggml-vector-index.h"

#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

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

} // namespace

int main() {
    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);

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

    std::printf("test-vector-index: OK\n");
    return 0;
}
