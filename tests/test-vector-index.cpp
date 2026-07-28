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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
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

int test_pid() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

std::filesystem::path temp_path(const char * suffix) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
        ("ggml-vector-index-test-" + std::to_string(test_pid()) + "-" +
         std::to_string(counter++) + suffix);
}

struct TempFile {
    explicit TempFile(const char * suffix) : path(temp_path(suffix)) {
        std::filesystem::remove(path);
    }

    ~TempFile() {
        std::filesystem::remove(path);
    }

    std::filesystem::path path;
};

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

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

std::vector<uint8_t> snapshot_bytes(
        uint32_t dim,
        uint32_t n,
        const std::vector<float> & values,
        const std::vector<uint64_t> & ids) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), { 'T', 'V', 'P', 'I' });
    bytes.push_back(1);
    bytes.push_back(32);
    bytes.push_back(0);
    bytes.push_back(0);
    append_u32_le(bytes, dim);
    append_u32_le(bytes, n);
    for (float value : values) {
        append_f32_le(bytes, value);
    }
    for (uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    return bytes;
}

} // namespace

int main() {
    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);
    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/31) == nullptr);

    // Public APIs reject invalid arguments without mutating the index.
    {
        const std::array<float, kDim> vector = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t id = 123ULL;
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};

        CHECK(ggml_vec_index_add(nullptr, vector.data(), 1, &id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, nullptr, 1, &id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, vector.data(), 1, nullptr)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, vector.data(), -1, &id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(
            nullptr, vector.data(), 1, 1, scores.data(), out_ids.data())
            == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(
            idx, nullptr, 1, 1, scores.data(), out_ids.data())
            == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(
            idx, vector.data(), -1, 1, scores.data(), out_ids.data())
            == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(
            idx, vector.data(), 1, 0, scores.data(), out_ids.data())
            == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(
            idx, vector.data(), 1, 1, nullptr, out_ids.data())
            == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(nullptr, id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Empty indexes return only sentinel-padded search results.
    {
        const std::array<float, kDim> query = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, query.data(), 1, static_cast<int>(scores.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < scores.size(); ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
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

    // UINT64_MAX is reserved for padded search results.
    {
        const std::vector<uint64_t> reserved_ids = { UINT64_MAX };
        std::vector<float> reserved_vec(seeds[0]);
        CHECK(ggml_vec_index_add(idx, reserved_vec.data(), 1, reserved_ids.data())
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // Non-finite vectors are rejected before mutating state.
    {
        const std::vector<uint64_t> bad_ids = { 777ULL };
        std::vector<float> bad_vec(seeds[0]);
        bad_vec[2] = std::numeric_limits<float>::quiet_NaN();
        CHECK(ggml_vec_index_add(idx, bad_vec.data(), 1, bad_ids.data())
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(idx, bad_ids[0]) == 0);
        CHECK(ggml_vec_index_len(idx) == 4);
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

    // Non-finite queries are rejected.
    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        std::array<float, kDim> bad_query = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        bad_query[1] = std::numeric_limits<float>::infinity();
        CHECK(ggml_vec_index_search(
            idx, bad_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

    // Finite inputs that overflow f32 score range are clamped and still keep
    // the real result distinct from sentinel-padded ids.
    {
        auto * overflow_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(overflow_idx != nullptr);
        const std::array<float, kDim> overflow_vec = {
            -FLT_MAX, 0.0f, 0.0f, 0.0f,
        };
        const std::array<uint64_t, 1> overflow_ids = { 1234567ULL };
        CHECK(ggml_vec_index_add(
            overflow_idx, overflow_vec.data(), 1, overflow_ids.data()) == 0);
        const std::array<float, kDim> overflow_query = {
            FLT_MAX, 0.0f, 0.0f, 0.0f,
        };
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            overflow_idx, overflow_query.data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == 0);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(out_ids[0] == overflow_ids[0]);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        ggml_vec_index_free(overflow_idx);
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
    TempFile round_trip_file(".tvim");
    const std::string path = round_trip_file.path.string();
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

    // Malformed snapshots are rejected before allocating from untrusted counts.
    {
        TempFile truncated_header(".tvim");
        write_bytes(truncated_header.path, { 'T', 'V', 'P' });
        CHECK(ggml_vec_index_load(truncated_header.path.string().c_str()) == nullptr);
    }
    {
        TempFile oversized_count(".tvim");
        write_bytes(oversized_count.path, snapshot_bytes(
            /*dim=*/1,
            static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u,
            {},
            {}));
        CHECK(ggml_vec_index_load(oversized_count.path.string().c_str()) == nullptr);
    }
    {
        TempFile truncated_payload(".tvim");
        write_bytes(truncated_payload.path, snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/2,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            {}));
        CHECK(ggml_vec_index_load(truncated_payload.path.string().c_str()) == nullptr);
    }
    {
        TempFile bad_magic(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            { 123ULL });
        bytes[0] = 'X';
        write_bytes(bad_magic.path, bytes);
        CHECK(ggml_vec_index_load(bad_magic.path.string().c_str()) == nullptr);
    }
    {
        TempFile bad_version(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            { 123ULL });
        bytes[4] = 99;
        write_bytes(bad_version.path, bytes);
        CHECK(ggml_vec_index_load(bad_version.path.string().c_str()) == nullptr);
    }
    {
        TempFile bad_bit_width(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            { 123ULL });
        bytes[5] = 8;
        write_bytes(bad_bit_width.path, bytes);
        CHECK(ggml_vec_index_load(bad_bit_width.path.string().c_str()) == nullptr);
    }
    {
        TempFile reserved_header(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            { 123ULL });
        CHECK(bytes.size() >= 8);
        bytes[6] = 1;
        write_bytes(reserved_header.path, bytes);
        CHECK(ggml_vec_index_load(reserved_header.path.string().c_str()) == nullptr);
    }
    {
        TempFile duplicate_ids(".tvim");
        write_bytes(duplicate_ids.path, snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/2,
            {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
            },
            { 321ULL, 321ULL }));
        CHECK(ggml_vec_index_load(duplicate_ids.path.string().c_str()) == nullptr);
    }
    {
        TempFile reserved_id_file(".tvim");
        write_bytes(reserved_id_file.path, snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { 1.0f, 0.0f, 0.0f, 0.0f },
            { UINT64_MAX }));
        CHECK(ggml_vec_index_load(reserved_id_file.path.string().c_str()) == nullptr);
    }
    {
        TempFile non_finite_file(".tvim");
        write_bytes(non_finite_file.path, snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1,
            { std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f },
            { 321ULL }));
        CHECK(ggml_vec_index_load(non_finite_file.path.string().c_str()) == nullptr);
    }

    std::printf("test-vector-index: OK\n");
    return 0;
}
