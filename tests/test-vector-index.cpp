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
#include <cfenv>
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
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#ifndef _WIN32
#include <sys/stat.h>
#endif

uint64_t turbovec_rotation_hash_for_test(int dim);
uint64_t turbovec_query_rotation_hash_for_test(
    const float * queries,
    int n_queries,
    int dim);
uint64_t turbovec_lut_hash_for_test(
    const float * query,
    const float * tqplus_shift,
    const float * tqplus_scale,
    int bits,
    int n_queries,
    int dim,
    uint32_t * lut_scale_bits,
    uint32_t * lut_bias_bits);
uint64_t turbovec_codebook_hash_for_test(int bits, int dim);
uint64_t turbovec_blocked_hash_for_test(const ggml_vec_index_t * idx);
void turbovec_clear_blocked_for_test(ggml_vec_index_t * idx);
int turbovec_avx2_available_for_test();
int turbovec_avx2_lut_block_matches_scalar_for_test(int bits, int dim);

namespace {

constexpr int kDim = 4;

#include "turbovec-golden-q2.inc"
#include "turbovec-golden-q4.inc"

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

int round_nearest_even(float value) {
    const float lower_f = std::floor(value);
    const float upper_f = lower_f + 1.0f;
    const float lower_dist = value - lower_f;
    const float upper_dist = upper_f - value;
    if (lower_dist < upper_dist) {
        return static_cast<int>(lower_f);
    }
    if (upper_dist < lower_dist) {
        return static_cast<int>(upper_f);
    }

    const int lower = static_cast<int>(lower_f);
    return (lower % 2) == 0 ? lower : static_cast<int>(upper_f);
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
            0 : round_nearest_even(vector[i] / scale);
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
            0 : round_nearest_even(vector[i] / scale);
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

uint32_t read_u32_le_from(const uint8_t * bytes) {
    return static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
}

std::vector<uint8_t> bytes_to_vector(const uint8_t * bytes, size_t len) {
    return std::vector<uint8_t>(bytes, bytes + len);
}

bool any_score_differs(const float * lhs, const float * rhs, size_t n, float eps) {
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > eps) {
            return true;
        }
    }
    return false;
}

std::filesystem::path make_test_temp_dir() {
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(getpid());
#endif
    return std::filesystem::temp_directory_path() /
        ("ggml-vector-index-test-" + std::to_string(pid));
}

void set_test_temp_dir(const std::filesystem::path & path) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    const std::string value = path.string();
#ifdef _WIN32
    _putenv_s("TMP", value.c_str());
    _putenv_s("TEMP", value.c_str());
#else
    setenv("TMPDIR", value.c_str(), 1);
#endif
}

void check_rust_tv_shape(
        const uint8_t * bytes,
        size_t len,
        int bit_width,
        int dim,
        int n,
        int packed_bytes,
        int calib_count) {
    CHECK(len >= 14);
    CHECK(bytes[0] == 'T' && bytes[1] == 'V' && bytes[2] == 'P' && bytes[3] == 'I');
    CHECK(bytes[4] == 3);
    CHECK(bytes[5] == static_cast<uint8_t>(bit_width));
    CHECK(read_u32_le_from(bytes + 6) == static_cast<uint32_t>(dim));
    CHECK(read_u32_le_from(bytes + 10) == static_cast<uint32_t>(n));
    CHECK(14 + static_cast<size_t>(packed_bytes) + static_cast<size_t>(n) * 4 + 4 <= len);
    const size_t calib_offset = 14 + static_cast<size_t>(packed_bytes) + static_cast<size_t>(n) * 4;
    CHECK(read_u32_le_from(bytes + calib_offset) == static_cast<uint32_t>(calib_count));
}

void check_rust_persistence_parity(
        ggml_vec_index_t * idx,
        const char * suffix,
        int bit_width,
        int storage_kind,
        int n,
        const float * rust_scales,
        size_t rust_scale_count,
        const uint8_t * rust_tv,
        size_t rust_tv_len,
        const uint8_t * rust_codes,
        size_t rust_codes_len) {
    const std::string rust_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-rust-" + std::string(suffix) + ".tv")).string();
    const std::string qvac_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-qvac-" + std::string(suffix) + ".tvim")).string();
    std::filesystem::remove(rust_path);
    std::filesystem::remove(qvac_path);

    write_file_bytes(rust_path, bytes_to_vector(rust_tv, rust_tv_len));
    ggml_vec_index_t * rust_loaded = nullptr;
    CHECK(ggml_vec_index_load_ex(rust_path.c_str(), &rust_loaded) == GGML_VEC_INDEX_E_IO);
    CHECK(rust_loaded == nullptr);
    CHECK(ggml_vec_index_load(rust_path.c_str()) == nullptr);

    CHECK(ggml_vec_index_write(idx, qvac_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> qvac = read_file_bytes(qvac_path);
    CHECK(qvac.size() >= 32);
    CHECK(qvac[0] == 'T' && qvac[1] == 'V' && qvac[2] == 'P' && qvac[3] == 'I');
    CHECK(qvac[4] == 3);
    CHECK(qvac[5] == static_cast<uint8_t>(bit_width));
    CHECK(qvac[6] == static_cast<uint8_t>(storage_kind));

    const size_t qparam_bytes = read_u32_le_from(qvac.data() + 20);
    const size_t calibration_bytes = read_u32_le_from(qvac.data() + 28);
    const size_t dim = read_u32_le_from(qvac.data() + 8);
    CHECK(qparam_bytes == sizeof(float));
    CHECK(calibration_bytes == 2 * dim * sizeof(float));
    CHECK(rust_scale_count == static_cast<size_t>(n));
    for (size_t i = 0; i < rust_scale_count; ++i) {
        uint32_t expected = 0;
        std::memcpy(&expected, rust_scales + i, sizeof(expected));
        const uint32_t actual = read_u32_le_from(qvac.data() + 32 + i * sizeof(float));
        const uint32_t ulp_diff = actual > expected ? actual - expected : expected - actual;
        CHECK(ulp_diff <= 1);
    }
    const size_t vector_offset =
        32 + static_cast<size_t>(n) * qparam_bytes + calibration_bytes;
    CHECK(qvac.size() >= vector_offset + rust_codes_len);
    const bool same_codes = std::equal(
        rust_codes,
        rust_codes + rust_codes_len,
        qvac.data() + vector_offset);
    CHECK(same_codes);
    CHECK(qvac != bytes_to_vector(rust_tv, rust_tv_len));

    std::filesystem::remove(rust_path);
    std::filesystem::remove(qvac_path);
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

uint64_t read_u64_le_at(const std::vector<uint8_t> & bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[offset + static_cast<size_t>(i)]) << (8 * i);
    }
    return value;
}

void write_u32_le_at(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void append_f32_le(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32_le(bytes, bits);
}

uint32_t crc32c_update(uint32_t crc, const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0x82f63b78u & mask);
        }
    }
    return crc;
}

uint32_t crc32c_update_u32(uint32_t crc, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes, sizeof(bytes));
}

uint32_t crc32c_update_u64(uint32_t crc, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes, sizeof(bytes));
}

bool delta_log_is_v4(const std::vector<uint8_t> & bytes) {
    return bytes.size() > 4 && bytes[4] == 4;
}

size_t delta_log_header_size(const std::vector<uint8_t> & bytes) {
    return delta_log_is_v4(bytes) ? 48 : 16;
}

size_t delta_record_header_size(const std::vector<uint8_t> & bytes) {
    return delta_log_is_v4(bytes) ? 56 : 24;
}

size_t delta_record_state_offset(const std::vector<uint8_t> & bytes, size_t record_offset) {
    return record_offset + (delta_log_is_v4(bytes) ? 24 : 20);
}

size_t delta_record_payload_offset(const std::vector<uint8_t> & bytes, size_t record_offset) {
    return record_offset + delta_record_header_size(bytes);
}

void refresh_delta_record_crc(std::vector<uint8_t> & bytes, size_t record_offset) {
    const uint64_t payload_bytes = read_u64_le_at(bytes, record_offset + 8);
    uint32_t crc = crc32c_update(0xffffffffu, bytes.data() + record_offset, 16);
    if (delta_log_is_v4(bytes)) {
        crc = crc32c_update(crc, bytes.data() + record_offset + 24, 32);
    } else {
        crc = crc32c_update(crc, bytes.data() + record_offset + 20, 4);
    }
    if (payload_bytes != 0) {
        crc = crc32c_update(
            crc,
            bytes.data() + delta_record_payload_offset(bytes, record_offset),
            static_cast<size_t>(payload_bytes));
    }
    write_u32_le_at(bytes, record_offset + 16, crc ^ 0xffffffffu);
}

uint64_t rotl64(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t slot_state_hash_f32(uint64_t id, const std::vector<float> & vector) {
    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0 = crc32c_update_u64(crc0, id);
    crc1 = crc32c_update_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ull);
    for (float value : vector) {
        const uint32_t bits = float_bits(value);
        crc0 = crc32c_update_u32(crc0, bits);
        crc1 = crc32c_update_u32(crc1, bits ^ 0xa5a5a5a5u);
    }
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

void quantize_for_state_hash(
        int bit_width,
        const std::vector<float> & vector,
        std::vector<uint8_t> & codes,
        float & scale) {
    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }

    if (bit_width == 8) {
        codes.assign(vector.size(), 0);
        if (max_abs == 0.0f) {
            scale = 1.0f;
            return;
        }
        scale = max_abs / 127.0f;
        for (size_t i = 0; i < vector.size(); ++i) {
            int q = round_nearest_even(vector[i] / scale);
            q = std::max(-127, std::min(127, q));
            codes[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        }
        return;
    }

    codes.assign((vector.size() + 1) / 2, 0x88);
    if (max_abs == 0.0f) {
        scale = 1.0f;
        return;
    }
    scale = max_abs / 7.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int q = round_nearest_even(vector[i] / scale);
        q = std::max(-7, std::min(7, q));
        const uint8_t code = static_cast<uint8_t>(q + 8);
        uint8_t & byte = codes[i / 2];
        if ((i & 1) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | code);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (code << 4));
        }
    }
}

uint64_t slot_state_hash_quantized(
        int bit_width,
        uint64_t id,
        const std::vector<float> & vector) {
    std::vector<uint8_t> codes;
    float scale = 1.0f;
    quantize_for_state_hash(bit_width, vector, codes, scale);

    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0 = crc32c_update_u64(crc0, id);
    crc1 = crc32c_update_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ull);
    const uint32_t scale_bits = float_bits(scale);
    crc0 = crc32c_update_u32(crc0, scale_bits);
    crc1 = crc32c_update_u32(crc1, scale_bits ^ 0xa5a5a5a5u);
    crc0 = crc32c_update(crc0, codes.data(), codes.size());
    crc1 = crc32c_update(crc1, codes.data(), codes.size());
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

uint32_t f32_state_token(
        int dim,
        size_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    const uint32_t bit_width = 32;
    const uint32_t storage_kind = 1;
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, bit_width);
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(n_active));
    crc = crc32c_update_u64(crc, hash_xor);
    crc = crc32c_update_u64(crc, hash_sum);
    crc = crc32c_update_u64(crc, hash_sum_rot);
    return crc ^ 0xffffffffu;
}

uint32_t state_token_from_wide_log_header(const std::vector<uint8_t> & bytes, int dim) {
    CHECK(delta_log_is_v4(bytes));
    const int bit_width = static_cast<int>(bytes[5]);
    const uint32_t storage_kind =
        bit_width == 4 ? 3u : (bit_width == 8 ? 2u : 1u);
    const size_t state_offset = 16;
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 0));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 8));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 16));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 24));
    return crc ^ 0xffffffffu;
}

uint32_t state_token_from_wide_values(
        int bit_width,
        int dim,
        uint64_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    const uint32_t storage_kind =
        bit_width == 4 ? 3u : (bit_width == 8 ? 2u : 1u);
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, n_active);
    crc = crc32c_update_u64(crc, hash_xor);
    crc = crc32c_update_u64(crc, hash_sum);
    crc = crc32c_update_u64(crc, hash_sum_rot);
    return crc ^ 0xffffffffu;
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

void write_v2_turbovec_index(
        const std::string & path,
        int dim,
        int bit_width,
        uint8_t storage_kind) {
    std::vector<uint8_t> bytes = { 'T', 'V', 'P', 'I', 2,
                                   static_cast<uint8_t>(bit_width), storage_kind, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, 1);
    append_u32_le(bytes, 1);
    append_u32_le(bytes, sizeof(float));
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    append_f32_le(bytes, 1.0f);
    bytes.resize(bytes.size() + static_cast<size_t>(bit_width) * (static_cast<size_t>(dim) / 8), 0);
    append_u64_le(bytes, 12345);
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

uint64_t fnv1a_bytes(const uint8_t * values, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; ++i) {
        hash ^= values[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

float tqplus_golden_value(int row, int column) {
    const double x = static_cast<double>(row + 1);
    const double y = static_cast<double>(column + 1);
    return static_cast<float>(
        0.63 * std::sin(0.017 * x * y + 0.47) +
        0.31 * std::cos(0.041 * (x + 3.0) * (y + 1.0)) +
        0.06 * std::sin(0.097 * (x + y)));
}

void check_tqplus_rust_parity(
        int bits,
        uint64_t expected_codes_hash,
        uint64_t expected_scales_hash,
        uint64_t expected_shift_hash,
        uint64_t expected_tqscale_hash) {
    constexpr int dim = 128;
    constexpr int n = 1000;
    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    std::vector<uint64_t> ids(n);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(row + 1);
        for (int column = 0; column < dim; ++column) {
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(column)] =
                tqplus_golden_value(row, column);
        }
    }

    ggml_vec_index_t * index = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    CHECK(index != nullptr);
    CHECK(ggml_vec_index_add(index, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    const uint64_t expected_blocked_hash = bits == 2 ?
        UINT64_C(0x481ee4411871b4cd) :
        UINT64_C(0x001f1478c8b61a63);
    CHECK(turbovec_blocked_hash_for_test(index) == expected_blocked_hash);
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-tqplus-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(path);
    CHECK(ggml_vec_index_write(index, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> bytes = read_file_bytes(path);
    CHECK(bytes.size() > 32);
    CHECK(bytes[4] == 3);
    CHECK(read_u32_le_from(bytes.data() + 28) == 2 * dim * sizeof(float));

    const size_t scales_offset = 32;
    const size_t scales_bytes = n * sizeof(float);
    const size_t shift_offset = scales_offset + scales_bytes;
    const size_t calibration_bytes = dim * sizeof(float);
    const size_t tqscale_offset = shift_offset + calibration_bytes;
    const size_t codes_offset = tqscale_offset + calibration_bytes;
    const size_t codes_bytes = static_cast<size_t>(n) * bits * (dim / 8);
    CHECK(fnv1a_bytes(bytes.data() + codes_offset, codes_bytes) == expected_codes_hash);
    CHECK(fnv1a_bytes(bytes.data() + scales_offset, scales_bytes) == expected_scales_hash);
    CHECK(fnv1a_bytes(bytes.data() + shift_offset, calibration_bytes) == expected_shift_hash);
    CHECK(fnv1a_bytes(bytes.data() + tqscale_offset, calibration_bytes) == expected_tqscale_hash);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(turbovec_blocked_hash_for_test(loaded) == expected_blocked_hash);
    std::array<float, 3 * dim> queries{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < dim; ++column) {
            const double x = static_cast<double>(row * 2 + 1);
            const double y = static_cast<double>(column + 1);
            queries[static_cast<size_t>(row) * dim + static_cast<size_t>(column)] =
                static_cast<float>(
                    0.63 * std::sin(0.017 * x * y + 0.29) +
                    0.31 * std::cos(0.041 * (x + 3.0) * (y + 1.0)) +
                    0.06 * std::sin(0.097 * (x + y)));
        }
    }
    CHECK(turbovec_query_rotation_hash_for_test(queries.data(), 3, dim) ==
        UINT64_C(0x22582f085fd79768));
    std::array<float, dim> tqplus_shift{};
    std::array<float, dim> tqplus_scale{};
    std::memcpy(tqplus_shift.data(), bytes.data() + shift_offset, calibration_bytes);
    std::memcpy(tqplus_scale.data(), bytes.data() + tqscale_offset, calibration_bytes);
    uint32_t lut_scale_bits = 0;
    uint32_t lut_bias_bits = 0;
    const uint64_t lut_hash = turbovec_lut_hash_for_test(
        queries.data(),
        tqplus_shift.data(),
        tqplus_scale.data(),
        bits,
        3,
        dim,
        &lut_scale_bits,
        &lut_bias_bits);
    CHECK(lut_hash == (bits == 2 ?
        UINT64_C(0x3b105f838666dbbb) :
        UINT64_C(0x9691906f2a148805)));
    CHECK(lut_scale_bits == (bits == 2 ? 0x3ba9233c : 0x3bb920ca));
    CHECK(lut_bias_bits == (bits == 2 ? 0xc0fabc4e : 0xc1606205));
    CHECK(turbovec_codebook_hash_for_test(bits, dim) == (bits == 2 ?
        UINT64_C(0xa37c605fe8acd601) :
        UINT64_C(0xd74197c1c7f95b91)));
    static constexpr std::array<uint32_t, 9> q2_score_bits = {
        0x4223a35c, 0x420f51fc, 0x420905a1,
        0x41f4e33a, 0x41bd8451, 0x41bd7b9f,
        0x4211fc2c, 0x41cd812e, 0x41c1e660,
    };
    static constexpr std::array<uint64_t, 9> q2_ids = {
        1, 740, 370, 3, 372, 742, 5, 375, 744,
    };
    static constexpr std::array<uint32_t, 9> q4_score_bits = {
        0x422312e8, 0x42108e97, 0x42020020,
        0x41f4f68c, 0x41c97248, 0x41b70ac3,
        0x4212e92f, 0x41cb8eec, 0x41cb0d33,
    };
    static constexpr std::array<uint64_t, 9> q4_ids = {
        1, 740, 370, 3, 742, 372, 5, 375, 744,
    };
    const auto & expected_score_bits = bits == 2 ? q2_score_bits : q4_score_bits;
    const auto & expected_ids = bits == 2 ? q2_ids : q4_ids;
    std::array<float, 9> scores{};
    std::array<uint64_t, 9> results{};
    CHECK(ggml_vec_index_search(
        loaded, queries.data(), 3, 3, scores.data(), results.data()) == GGML_VEC_INDEX_OK);
    for (size_t i = 0; i < scores.size(); ++i) {
        uint32_t actual_score_bits = 0;
        std::memcpy(&actual_score_bits, &scores[i], sizeof(actual_score_bits));
        CHECK(actual_score_bits == expected_score_bits[i]);
        CHECK(results[i] == expected_ids[i]);
    }
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(index);
    std::filesystem::remove(path);
}

void check_turbovec_blocked_scalar_scores(int bits, int dim, int n, int n_queries) {
    auto * blocked = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    auto * scalar = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    CHECK(blocked != nullptr);
    CHECK(scalar != nullptr);

    const uint64_t id_base = static_cast<uint64_t>(30000 + bits * 1000 + dim + n);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = id_base + static_cast<uint64_t>(row);
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 1);
            const double y = static_cast<double>(col + 3);
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.55 * std::sin(0.013 * x * y + 0.17) +
                    0.35 * std::cos(0.019 * (x + 5.0) * (y + 1.0)) +
                    0.10 * std::sin(0.071 * (x + y)));
        }
    }

    CHECK(ggml_vec_index_add(blocked, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(scalar, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(turbovec_blocked_hash_for_test(blocked) != 0);
    turbovec_clear_blocked_for_test(scalar);
    CHECK(turbovec_blocked_hash_for_test(scalar) == 0);

    std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
    for (int row = 0; row < n_queries; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 2);
            const double y = static_cast<double>(col + 7);
            queries[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.48 * std::cos(0.023 * x * y + 0.31) +
                    0.41 * std::sin(0.037 * (x + 3.0) * (y + 2.0)) +
                    0.11 * std::cos(0.083 * (x + y)));
        }
    }

    std::vector<float> blocked_scores(static_cast<size_t>(n_queries) * n);
    std::vector<float> scalar_scores(static_cast<size_t>(n_queries) * n);
    std::vector<uint64_t> blocked_ids(static_cast<size_t>(n_queries) * n);
    std::vector<uint64_t> scalar_ids(static_cast<size_t>(n_queries) * n);
    CHECK(ggml_vec_index_search(
        blocked, queries.data(), n_queries, n, blocked_scores.data(), blocked_ids.data()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search(
        scalar, queries.data(), n_queries, n, scalar_scores.data(), scalar_ids.data()) ==
        GGML_VEC_INDEX_OK);

    const float tolerance = static_cast<float>(dim) * (bits == 2 ? 0.0045f : 0.0025f);
    std::vector<float> blocked_by_row(static_cast<size_t>(n));
    std::vector<float> scalar_by_row(static_cast<size_t>(n));
    for (int query = 0; query < n_queries; ++query) {
        std::fill(blocked_by_row.begin(), blocked_by_row.end(), std::numeric_limits<float>::quiet_NaN());
        std::fill(scalar_by_row.begin(), scalar_by_row.end(), std::numeric_limits<float>::quiet_NaN());
        for (int rank = 0; rank < n; ++rank) {
            const size_t offset = static_cast<size_t>(query) * n + static_cast<size_t>(rank);
            CHECK(blocked_ids[offset] >= id_base);
            CHECK(blocked_ids[offset] < id_base + static_cast<uint64_t>(n));
            CHECK(scalar_ids[offset] >= id_base);
            CHECK(scalar_ids[offset] < id_base + static_cast<uint64_t>(n));
            blocked_by_row[static_cast<size_t>(blocked_ids[offset] - id_base)] = blocked_scores[offset];
            scalar_by_row[static_cast<size_t>(scalar_ids[offset] - id_base)] = scalar_scores[offset];
        }
        for (int row = 0; row < n; ++row) {
            const float blocked_score = blocked_by_row[static_cast<size_t>(row)];
            const float scalar_score = scalar_by_row[static_cast<size_t>(row)];
            CHECK(std::isfinite(blocked_score));
            CHECK(std::isfinite(scalar_score));
            const float drift = std::fabs(blocked_score - scalar_score);
            if (!(drift <= tolerance)) {
                std::fprintf(
                    stderr,
                    "FAIL TurboVec q%d blocked/scalar drift: dim=%d n=%d query=%d row=%d drift=%g tolerance=%g\n",
                    bits,
                    dim,
                    n,
                    query,
                    row,
                    static_cast<double>(drift),
                    static_cast<double>(tolerance));
                std::exit(1);
            }
        }
    }

    ggml_vec_index_free(scalar);
    ggml_vec_index_free(blocked);
}

} // namespace

int main() {
    const std::filesystem::path test_temp_dir = make_test_temp_dir();
    set_test_temp_dir(test_temp_dir);

    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(-1, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/16) == nullptr);
    CHECK(ggml_vec_index_contains(nullptr, 123ULL) == 0);
    CHECK(ggml_vec_index_len(nullptr) == 0);
    CHECK(ggml_vec_index_dim(nullptr) == 0);
    CHECK(ggml_vec_index_bit_width(nullptr) == 0);
    ggml_vec_index_prepare(nullptr);
    CHECK(ggml_vec_index_create(kDim, 2) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(0) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(kDim) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(65544) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(0) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(kDim) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(65544) == nullptr);
    auto * max_dim_q2 = ggml_vec_index_create_turbovec_q2(65536);
    CHECK(max_dim_q2 != nullptr);
    ggml_vec_index_free(max_dim_q2);
    auto * max_dim_q4 = ggml_vec_index_create_turbovec_q4(65536);
    CHECK(max_dim_q4 != nullptr);
    ggml_vec_index_free(max_dim_q4);

    // The early TurboVec prototype wrote incompatible v2 snapshots. They must
    // not be silently decoded as the v3 Rust-compatible layout.
    {
        const std::string path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-v2-turbovec.tvim").string();
        for (const auto & spec : {
                 std::pair<int, uint8_t>{ 2, 5 },
                 std::pair<int, uint8_t>{ 4, 4 },
             }) {
            std::filesystem::remove(path);
            write_v2_turbovec_index(path, 128, spec.first, spec.second);
            ggml_vec_index_t * loaded = nullptr;
            CHECK(ggml_vec_index_load_ex(path.c_str(), &loaded) ==
                  GGML_VEC_INDEX_E_BAD_VERSION);
            CHECK(loaded == nullptr);
            CHECK(ggml_vec_index_load(path.c_str()) == nullptr);
        }
        std::filesystem::remove(path);
    }

    // Match Rust input validation: finite but unsafe-magnitude coordinates are
    // rejected before they can overflow TurboVec's float norm/score path.
    {
        constexpr int tv_dim = 64;
        for (const int bit_width : { 2, 4 }) {
            auto * tv = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(tv != nullptr);
            std::vector<float> safe(static_cast<size_t>(tv_dim), 0.25f);
            const uint64_t safe_id = static_cast<uint64_t>(9300 + bit_width);
            CHECK(ggml_vec_index_add(tv, safe.data(), 1, &safe_id) == GGML_VEC_INDEX_OK);
            std::vector<float> unsafe(static_cast<size_t>(tv_dim), 0.0f);
            unsafe[3] = 1e16f;
            const uint64_t unsafe_id = static_cast<uint64_t>(9350 + bit_width);
            CHECK(ggml_vec_index_add(tv, unsafe.data(), 1, &unsafe_id) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_len(tv) == 1);
            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out{};
            CHECK(ggml_vec_index_search(tv, unsafe.data(), 1, 1, scores.data(), out.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            ggml_vec_index_free(tv);
        }
    }

    // TurboVec bit-plane packing only requires dimensions to be multiples of 8.
    {
        constexpr int tv_dim = 64;
        for (const int bit_width : { 2, 4 }) {
            auto * tv = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(tv != nullptr);

            std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 2);
            for (int i = 0; i < tv_dim; ++i) {
                tv_vecs[static_cast<size_t>(i)] =
                    static_cast<float>(std::sin(0.03 * static_cast<double>(i + 1)));
                tv_vecs[static_cast<size_t>(tv_dim + i)] =
                    static_cast<float>(std::cos(0.05 * static_cast<double>(i + 3)));
            }
            const std::array<uint64_t, 2> tv_ids = {
                static_cast<uint64_t>(9400 + bit_width),
                static_cast<uint64_t>(9500 + bit_width),
            };
            CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 2, tv_ids.data()) ==
                  GGML_VEC_INDEX_OK);

            std::array<float, 2> scores{};
            std::array<uint64_t, 2> out{};
            CHECK(ggml_vec_index_search(tv, tv_vecs.data(), 1, 2, scores.data(), out.data()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(out[0] == tv_ids[0]);

            const std::string path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-turbovec-dim64-q" + std::to_string(bit_width) + ".tvim")).string();
            std::filesystem::remove(path);
            CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
            auto * loaded = ggml_vec_index_load(path.c_str());
            CHECK(loaded != nullptr);
            CHECK(ggml_vec_index_dim(loaded) == tv_dim);
            CHECK(ggml_vec_index_bit_width(loaded) == bit_width);
            CHECK(ggml_vec_index_search(
                loaded, tv_vecs.data(), 1, 2, scores.data(), out.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(out[0] == tv_ids[0]);
            ggml_vec_index_free(loaded);
            ggml_vec_index_free(tv);
            std::filesystem::remove(path);
        }
    }

    // Appending TurboVec vectors only needs to repack the dirty tail blocks.
    {
        constexpr int tv_dim = 64;
        constexpr int n_vecs = 45;
        std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * n_vecs);
        std::vector<uint64_t> tv_ids(static_cast<size_t>(n_vecs));
        for (int row = 0; row < n_vecs; ++row) {
            tv_ids[static_cast<size_t>(row)] = static_cast<uint64_t>(9600 + row);
            for (int col = 0; col < tv_dim; ++col) {
                tv_vecs[static_cast<size_t>(row) * tv_dim + static_cast<size_t>(col)] =
                    static_cast<float>(std::sin(
                        0.01 * static_cast<double>((row + 1) * (col + 3))));
            }
        }
        for (const int bit_width : { 2, 4 }) {
            auto * batch = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            auto * incremental = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(batch != nullptr);
            CHECK(incremental != nullptr);
            CHECK(ggml_vec_index_add(batch, tv_vecs.data(), n_vecs, tv_ids.data()) ==
                  GGML_VEC_INDEX_OK);
            for (int row = 0; row < n_vecs; ++row) {
                CHECK(ggml_vec_index_add(
                    incremental,
                    tv_vecs.data() + static_cast<size_t>(row) * tv_dim,
                    1,
                    tv_ids.data() + row) == GGML_VEC_INDEX_OK);
            }
            const uint64_t batch_hash = turbovec_blocked_hash_for_test(batch);
            CHECK(batch_hash != 0);
            CHECK(turbovec_blocked_hash_for_test(incremental) == batch_hash);
            ggml_vec_index_free(incremental);
            ggml_vec_index_free(batch);
        }
    }

    // Optimized blocked scoring should stay close to the scalar row-LUT path,
    // including blocks that cross 32-lane boundaries and TQ+ calibration.
    for (const int bit_width : { 2, 4 }) {
        check_turbovec_blocked_scalar_scores(bit_width, 128, 17, 3);
        check_turbovec_blocked_scalar_scores(bit_width, 128, 33, 3);
        check_turbovec_blocked_scalar_scores(bit_width, 256, 65, 2);
    }
    check_turbovec_blocked_scalar_scores(2, 128, 1000, 2);
    check_turbovec_blocked_scalar_scores(4, 128, 1000, 2);
    for (const int bit_width : { 2, 4 }) {
        CHECK(turbovec_avx2_lut_block_matches_scalar_for_test(bit_width, 128) == 1);
        CHECK(turbovec_avx2_lut_block_matches_scalar_for_test(bit_width, 256) == 1);
    }

    // TurboVec q2/q4 are distinct modes. This first milestone supports
    // add/search/filter/IVF and regular snapshots; delta logs are format-gated.
    {
        constexpr int tv_dim = 128;
        auto * tv = ggml_vec_index_create_turbovec_q2(tv_dim);
        CHECK(tv != nullptr);
        CHECK(ggml_vec_index_dim(tv) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv) == 2);

        std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 3);
        for (int i = 0; i < tv_dim; ++i) {
            const float a = static_cast<float>(std::sin(0.07 * static_cast<double>(i + 1)));
            const float b = static_cast<float>(std::cos(0.11 * static_cast<double>(i + 3)));
            tv_vecs[static_cast<size_t>(i)] = a;
            tv_vecs[static_cast<size_t>(tv_dim + i)] = -a;
            tv_vecs[static_cast<size_t>(2 * tv_dim + i)] = b;
        }
        const std::array<uint64_t, 3> tv_ids = { 9201, 9202, 9203 };
        CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 3, tv_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 3);

        std::array<float, 3> tv_scores{};
        std::array<uint64_t, 3> tv_out{};
        CHECK(ggml_vec_index_search(
            tv, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::array<uint64_t, 2> allowed = { tv_ids[1], tv_ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            tv, tv_vecs.data(), 1, 1, allowed.data(), static_cast<int>(allowed.size()),
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] != tv_ids[0]);

        CHECK(ggml_vec_index_build_ivf(tv, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            tv, tv_vecs.data(), 1, 1, /*nprobe=*/2,
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::string tv_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q2.tvim").string();
        const std::string tv_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q2.tvid").string();
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        CHECK(ggml_vec_index_write(tv, tv_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * tv_loaded = ggml_vec_index_load(tv_snapshot_path.c_str());
        CHECK(tv_loaded != nullptr);
        CHECK(ggml_vec_index_dim(tv_loaded) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv_loaded) == 2);
        CHECK(ggml_vec_index_len(tv_loaded) == 3);
        ggml_vec_index_prepare(tv_loaded);
        CHECK(ggml_vec_index_search(
            tv_loaded, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);
        CHECK(ggml_vec_index_load_mmap(tv_snapshot_path.c_str()) == nullptr);
        ggml_vec_index_free(tv_loaded);
        ggml_vec_index_t * tv_delta_loaded = nullptr;
        CHECK(ggml_vec_index_load_with_delta_ex(
            tv_snapshot_path.c_str(), tv_delta_path.c_str(), &tv_delta_loaded) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(tv_delta_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta(
            tv_snapshot_path.c_str(), tv_delta_path.c_str()) == nullptr);
        CHECK(ggml_vec_index_compact_delta(
            tv, tv_snapshot_path.c_str(), tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        CHECK(ggml_vec_index_remove(tv, tv_ids[1]) == 1);
        CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 2);
        CHECK(ggml_vec_index_contains(tv, tv_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(tv, tv_ids[1]) == 0);
        const uint64_t replacement_id = 9204;
        CHECK(ggml_vec_index_add(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            &replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            1,
            tv_scores.data(),
            tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == replacement_id);
        CHECK(ggml_vec_index_add_logged(
            tv, tv_vecs.data(), 1, &tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(tv, tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        ggml_vec_index_free(tv);
    }

    // TurboVec q4 is a distinct mode. This first milestone supports
    // add/search/filter/IVF and regular snapshots; delta logs are format-gated.
    {
        constexpr int tv_dim = 128;
        auto * tv = ggml_vec_index_create_turbovec_q4(tv_dim);
        CHECK(tv != nullptr);
        CHECK(ggml_vec_index_dim(tv) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv) == 4);

        std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 3);
        for (int i = 0; i < tv_dim; ++i) {
            const float a = static_cast<float>(std::sin(0.07 * static_cast<double>(i + 1)));
            const float b = static_cast<float>(std::cos(0.11 * static_cast<double>(i + 3)));
            tv_vecs[static_cast<size_t>(i)] = a;
            tv_vecs[static_cast<size_t>(tv_dim + i)] = -a;
            tv_vecs[static_cast<size_t>(2 * tv_dim + i)] = b;
        }
        const std::array<uint64_t, 3> tv_ids = { 9101, 9102, 9103 };
        CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 3, tv_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 3);

        std::array<float, 3> tv_scores{};
        std::array<uint64_t, 3> tv_out{};
        CHECK(ggml_vec_index_search(
            tv, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::array<uint64_t, 2> allowed = { tv_ids[1], tv_ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            tv, tv_vecs.data(), 1, 1, allowed.data(), static_cast<int>(allowed.size()),
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] != tv_ids[0]);

        CHECK(ggml_vec_index_build_ivf(tv, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            tv, tv_vecs.data(), 1, 1, /*nprobe=*/2,
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::string tv_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q4.tvim").string();
        const std::string tv_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q4.tvid").string();
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        CHECK(ggml_vec_index_write(tv, tv_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * tv_loaded = ggml_vec_index_load(tv_snapshot_path.c_str());
        CHECK(tv_loaded != nullptr);
        CHECK(ggml_vec_index_dim(tv_loaded) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv_loaded) == 4);
        CHECK(ggml_vec_index_len(tv_loaded) == 3);
        ggml_vec_index_prepare(tv_loaded);
        CHECK(ggml_vec_index_search(
            tv_loaded, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);
        CHECK(ggml_vec_index_load_mmap(tv_snapshot_path.c_str()) == nullptr);
        ggml_vec_index_free(tv_loaded);
        ggml_vec_index_t * tv_delta_loaded = nullptr;
        CHECK(ggml_vec_index_load_with_delta_ex(
            tv_snapshot_path.c_str(), tv_delta_path.c_str(), &tv_delta_loaded) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(tv_delta_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta(
            tv_snapshot_path.c_str(), tv_delta_path.c_str()) == nullptr);
        CHECK(ggml_vec_index_compact_delta(
            tv, tv_snapshot_path.c_str(), tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        CHECK(ggml_vec_index_remove(tv, tv_ids[1]) == 1);
        CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 2);
        CHECK(ggml_vec_index_contains(tv, tv_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(tv, tv_ids[1]) == 0);
        const uint64_t replacement_id = 9104;
        CHECK(ggml_vec_index_add(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            &replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            1,
            tv_scores.data(),
            tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == replacement_id);
        CHECK(ggml_vec_index_add_logged(
            tv, tv_vecs.data(), 1, &tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(tv, tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        ggml_vec_index_free(tv);
    }

    // Rust TurboVec q2 golden parity: generated by tests/turbovec-golden-gen.
    {
        auto * tv = ggml_vec_index_create_turbovec_q2(kTurboVecGoldenQ2Dim);
        CHECK(tv != nullptr);
        std::array<uint64_t, kTurboVecGoldenQ2NDb> ids{};
        for (int i = 0; i < kTurboVecGoldenQ2NDb; ++i) {
            ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
        }
        CHECK(ggml_vec_index_add(
            tv,
            kTurboVecGoldenQ2Db,
            kTurboVecGoldenQ2NDb,
            ids.data()) == GGML_VEC_INDEX_OK);
        std::array<float, kTurboVecGoldenQ2NQuery * kTurboVecGoldenQ2K> scores{};
        std::array<uint64_t, kTurboVecGoldenQ2NQuery * kTurboVecGoldenQ2K> out{};
        CHECK(ggml_vec_index_search(
            tv,
            kTurboVecGoldenQ2Queries,
            kTurboVecGoldenQ2NQuery,
            kTurboVecGoldenQ2K,
            scores.data(),
            out.data()) == GGML_VEC_INDEX_OK);
        CHECK(!any_score_differs(
            scores.data(),
            kTurboVecGoldenQ2RustScores,
            scores.size(),
            1e-5f));
        check_rust_tv_shape(
            kTurboVecGoldenQ2RustTvBytes,
            kTurboVecGoldenQ2RustTvBytesLen,
            2,
            kTurboVecGoldenQ2Dim,
            kTurboVecGoldenQ2NDb,
            kTurboVecGoldenQ2RustPackedBytes,
            kTurboVecGoldenQ2RustCalibCount);
        check_rust_persistence_parity(
            tv,
            "q2-golden",
            2,
            5,
            kTurboVecGoldenQ2NDb,
            kTurboVecGoldenQ2RustScales,
            kTurboVecGoldenQ2RustScaleCount,
            kTurboVecGoldenQ2RustTvBytes,
            kTurboVecGoldenQ2RustTvBytesLen,
            kTurboVecGoldenQ2RustPackedCodes,
            kTurboVecGoldenQ2RustPackedBytes);
        for (int i = 0; i < kTurboVecGoldenQ2NQuery * kTurboVecGoldenQ2K; ++i) {
            CHECK(out[static_cast<size_t>(i)] ==
                static_cast<uint64_t>(kTurboVecGoldenQ2TopK[i]));
        }
        ggml_vec_index_free(tv);
    }

    // Rust TurboVec q4 golden parity: generated by tests/turbovec-golden-gen.
    // The small fixture uses Rust's identity TQ+ fallback.
    {
        CHECK(turbovec_rotation_hash_for_test(kTurboVecGoldenDim) ==
            kTurboVecGoldenRustRotationHash);
        auto * tv = ggml_vec_index_create_turbovec_q4(kTurboVecGoldenDim);
        CHECK(tv != nullptr);
        std::array<uint64_t, kTurboVecGoldenNDb> ids{};
        for (int i = 0; i < kTurboVecGoldenNDb; ++i) {
            ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
        }
        CHECK(ggml_vec_index_add(
            tv,
            kTurboVecGoldenDb,
            kTurboVecGoldenNDb,
            ids.data()) == GGML_VEC_INDEX_OK);
        std::array<float, kTurboVecGoldenNQuery * kTurboVecGoldenK> scores{};
        std::array<uint64_t, kTurboVecGoldenNQuery * kTurboVecGoldenK> out{};
        CHECK(ggml_vec_index_search(
            tv,
            kTurboVecGoldenQueries,
            kTurboVecGoldenNQuery,
            kTurboVecGoldenK,
            scores.data(),
            out.data()) == GGML_VEC_INDEX_OK);
        CHECK(!any_score_differs(
            scores.data(),
            kTurboVecGoldenRustScores,
            scores.size(),
            1e-5f));
        check_rust_tv_shape(
            kTurboVecGoldenRustTvBytes,
            kTurboVecGoldenRustTvBytesLen,
            4,
            kTurboVecGoldenDim,
            kTurboVecGoldenNDb,
            kTurboVecGoldenRustPackedBytes,
            kTurboVecGoldenRustCalibCount);
        check_rust_persistence_parity(
            tv,
            "q4-golden",
            4,
            4,
            kTurboVecGoldenNDb,
            kTurboVecGoldenRustScales,
            kTurboVecGoldenRustScaleCount,
            kTurboVecGoldenRustTvBytes,
            kTurboVecGoldenRustTvBytesLen,
            kTurboVecGoldenRustPackedCodes,
            kTurboVecGoldenRustPackedBytes);
        for (int i = 0; i < kTurboVecGoldenNQuery * kTurboVecGoldenK; ++i) {
            CHECK(out[static_cast<size_t>(i)] ==
                static_cast<uint64_t>(kTurboVecGoldenTopK[i]));
        }
        ggml_vec_index_free(tv);
    }

    check_tqplus_rust_parity(
        2,
        UINT64_C(0xc4140782241d45eb),
        UINT64_C(0x07b1e4792dc7ab14),
        UINT64_C(0x2fe473aa8d8ef2b2),
        UINT64_C(0x68e2ee49c3a5d29c));
    check_tqplus_rust_parity(
        4,
        UINT64_C(0x2c4e8e9e2a991e21),
        UINT64_C(0xe820eefd4297666f),
        UINT64_C(0x2fe473aa8d8ef2b2),
        UINT64_C(0x68e2ee49c3a5d29c));

    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);
    ggml_vec_index_prepare(idx);

    // Zero-row adds are valid no-ops and must not create delta artifacts.
    {
        const std::array<float, kDim> vector = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t id = 1234ULL;
        CHECK(ggml_vec_index_add(idx, vector.data(), /*n=*/0, &id)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 0);

        const std::string zero_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-zero-add.tvid").string();
        std::filesystem::remove(zero_delta_path);
        std::filesystem::remove(zero_delta_path + ".lock");
        CHECK(ggml_vec_index_add_logged(
            idx, vector.data(), /*n=*/0, &id, zero_delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 0);
        CHECK(!std::filesystem::exists(zero_delta_path));
        std::filesystem::remove(zero_delta_path + ".lock");
    }

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

    // UINT64_MAX is reserved as the empty-result sentinel.
    {
        const std::array<float, kDim> vector = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t reserved_id = UINT64_MAX;
        CHECK(ggml_vec_index_add(idx, vector.data(), 1, &reserved_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(idx, reserved_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Padding is identified by the id sentinel, not by score alone.
    {
        const std::array<float, kDim> min_score_vector = {
            -FLT_MAX, 0.0f, 0.0f, 0.0f,
        };
        const std::array<float, kDim> min_score_query = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t min_score_id = 12345ULL;
        CHECK(ggml_vec_index_add(idx, min_score_vector.data(), 1, &min_score_id) ==
              GGML_VEC_INDEX_OK);
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, min_score_query.data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == min_score_id);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(ggml_vec_index_remove(idx, min_score_id) == 1);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Add 4 well-separated unit vectors. IDs are non-trivial uint64 to
    // catch sign-extension bugs when this codepath is called from bindings.
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

    // Zero-query searches are no-ops, while k=0 is invalid for every search mode.
    {
        std::array<float, 4> scores = { 123.0f, 456.0f, 789.0f, 101.0f };
        std::array<uint64_t, 4> out_ids = { 1, 2, 3, 4 };
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(scores[0] == 123.0f);
        CHECK(out_ids[0] == 1);
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        const std::array<uint64_t, 2> allowed = { ids[0], ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        ggml_vec_index_filter_t * filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_filter_free(filter);

        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1, /*nprobe=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0, /*nprobe=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

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

        auto * empty_list_ann = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(empty_list_ann != nullptr);
        const std::array<float, 12> empty_list_vecs = {
             3.0f, 3.0f, 0.0f, 0.0f,
            -2.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
        };
        const std::array<uint64_t, 3> empty_list_ids = { 7101, 7102, 7103 };
        const std::array<float, 4> empty_list_query = { 1.0f, -3.0f, 0.0f, 0.0f };
        CHECK(ggml_vec_index_add(
            empty_list_ann,
            empty_list_vecs.data(),
            static_cast<int>(empty_list_ids.size()),
            empty_list_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_build_ivf(empty_list_ann, /*n_lists=*/3, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            empty_list_ann,
            empty_list_query.data(),
            1,
            /*k=*/1,
            /*nprobe=*/1,
            ann_scores.data(),
            ann_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ann_ids[0] == empty_list_ids[2]);
        CHECK(std::fabs(ann_scores[0] + 3.0f) < 1e-5f);
        ggml_vec_index_free(empty_list_ann);
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

    // Mutations are serialized with readers on the same handle.
    {
        constexpr int n_rows = 16;
        std::vector<float> rows;
        std::vector<uint64_t> row_ids;
        rows.reserve(static_cast<size_t>(n_rows) * kDim);
        row_ids.reserve(n_rows);
        for (int row = 0; row < n_rows; ++row) {
            const std::vector<float> v = normalize({
                1.0f,
                static_cast<float>((row % 3) - 1),
                static_cast<float>(((row + 2) % 5) - 2),
                0.5f,
            });
            rows.insert(rows.end(), v.begin(), v.end());
            row_ids.push_back(static_cast<uint64_t>(9000 + row));
        }

        auto * concurrent_mutation = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(concurrent_mutation != nullptr);
        CHECK(ggml_vec_index_add(
            concurrent_mutation, rows.data(), n_rows, row_ids.data()) ==
            GGML_VEC_INDEX_OK);

        std::atomic<int> ready{ 0 };
        std::atomic<bool> start{ false };
        std::atomic<bool> done{ false };
        std::atomic<int> failures{ 0 };
        std::vector<std::thread> readers;
        for (int t = 0; t < 4; ++t) {
            readers.emplace_back([&, t]() {
                const float * query =
                    rows.data() + static_cast<size_t>(t % n_rows) * kDim;
                std::array<float, 3> scores{};
                std::array<uint64_t, 3> out_ids{};
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                while (!done.load()) {
                    if (ggml_vec_index_search(
                            concurrent_mutation, query, 1, /*k=*/3,
                            scores.data(), out_ids.data()) != GGML_VEC_INDEX_OK) {
                        failures.fetch_add(1);
                    }
                    if (ggml_vec_index_len(concurrent_mutation) < n_rows) {
                        failures.fetch_add(1);
                    }
                }
            });
        }

        while (ready.load() != 4) {
            std::this_thread::yield();
        }
        start.store(true);
        for (int iter = 0; iter < 100; ++iter) {
            const std::vector<float> v = normalize({
                0.25f,
                static_cast<float>((iter % 7) - 3),
                1.0f,
                -0.5f,
            });
            const uint64_t id = static_cast<uint64_t>(10000 + iter);
            CHECK(ggml_vec_index_add(concurrent_mutation, v.data(), 1, &id)
                  == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_remove(concurrent_mutation, id) == 1);
        }
        done.store(true);
        for (std::thread & reader : readers) {
            reader.join();
        }
        CHECK(failures.load() == 0);
        CHECK(ggml_vec_index_len(concurrent_mutation) == n_rows);

        ggml_vec_index_free(concurrent_mutation);
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

    // Equal scores retain the lowest ids when top-k eviction is required.
    {
        auto * tie_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(tie_idx != nullptr);
        const std::array<uint64_t, 4> tie_ids = { 50, 10, 30, 20 };
        std::vector<float> tie_vecs;
        for (size_t i = 0; i < tie_ids.size(); ++i) {
            tie_vecs.insert(tie_vecs.end(), seeds[0].begin(), seeds[0].end());
        }
        CHECK(ggml_vec_index_add(
            tie_idx, tie_vecs.data(), static_cast<int>(tie_ids.size()), tie_ids.data()) ==
            GGML_VEC_INDEX_OK);
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            tie_idx, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == 10);
        CHECK(out_ids[1] == 20);
        CHECK(scores[0] == scores[1]);
        ggml_vec_index_free(tie_idx);
    }

    // Tombstone-heavy exact search visits live slots without scanning deleted rows.
    {
        auto * tombstone_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(tombstone_idx != nullptr);
        const std::array<uint64_t, 6> tombstone_ids = { 60, 50, 40, 30, 20, 10 };
        std::vector<float> tombstone_vecs;
        for (size_t i = 0; i < tombstone_ids.size(); ++i) {
            tombstone_vecs.insert(
                tombstone_vecs.end(), seeds[0].begin(), seeds[0].end());
        }
        CHECK(ggml_vec_index_add(
            tombstone_idx,
            tombstone_vecs.data(),
            static_cast<int>(tombstone_ids.size()),
            tombstone_ids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(ggml_vec_index_remove(tombstone_idx, tombstone_ids[i]) == 1);
        }

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            tombstone_idx, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == 10);
        CHECK(out_ids[1] == 20);
        ggml_vec_index_free(tombstone_idx);
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

        auto * other_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(other_idx != nullptr);
        CHECK(ggml_vec_index_add(
            other_idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            other_idx, filter, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(other_idx);

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

    {
        ggml_vec_index_t * diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(path.c_str(), &diag_loaded) == GGML_VEC_INDEX_OK);
        CHECK(diag_loaded != nullptr);
        ggml_vec_index_free(diag_loaded);
        CHECK(ggml_vec_index_load_ex(path.c_str(), nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(std::strcmp(
            ggml_vec_index_error_to_string(GGML_VEC_INDEX_E_BAD_MAGIC),
            "unknown error") != 0);

        const std::string bad_magic_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-bad-magic.tvim").string();
        const std::string bad_version_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-bad-version.tvim").string();
        const std::string diag_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-diagnostics.tvid").string();
        std::vector<uint8_t> bytes = read_file_bytes(path);

        CHECK(bytes.size() > 4);
        bytes[0] = 'X';
        write_file_bytes(bad_magic_path, bytes);

        diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(bad_magic_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_mmap_ex(bad_magic_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta_ex(
                  bad_magic_path.c_str(), diag_delta_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);

        bytes = read_file_bytes(path);
        CHECK(bytes.size() > 4);
        bytes[4] = 99;
        write_file_bytes(bad_version_path, bytes);

        diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(bad_version_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_VERSION);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_mmap_ex(bad_version_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_VERSION);
        CHECK(diag_loaded == nullptr);

        std::filesystem::remove(bad_magic_path);
        std::filesystem::remove(bad_version_path);
        std::filesystem::remove(diag_delta_path);
    }

    const std::string reserved_id_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-reserved-id-corrupt.tvim").string();
    expect_corrupt_load_fails(path, reserved_id_path, [](std::vector<uint8_t> & bytes) {
        constexpr size_t ids_offset = 32 + 3 * kDim * sizeof(float);
        CHECK(bytes.size() >= ids_offset + sizeof(uint64_t));
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            bytes[ids_offset + i] = 0xff;
        }
    });

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

    // Tombstone removal: later adds append, searches skip deleted slots, IVF
    // rebuilds exclude them, and snapshots write only live rows.
    {
        for (int bit_width : { 32, 8, 4 }) {
            const std::string tombstone_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-tombstone-" + std::to_string(bit_width) + ".tvim")).string();
            std::filesystem::remove(tombstone_path);

            auto * tombstone_idx = ggml_vec_index_create(kDim, bit_width);
            CHECK(tombstone_idx != nullptr);
            CHECK(ggml_vec_index_add(
                tombstone_idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_remove(tombstone_idx, ids[1]) == 1);
            CHECK(ggml_vec_index_len(tombstone_idx) == 3);

            const uint64_t appended_id =
                (1ULL << 40) + 123ULL + static_cast<uint64_t>(bit_width);
            CHECK(ggml_vec_index_add(
                tombstone_idx, seeds[1].data(), 1, &appended_id) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(tombstone_idx) == 4);
            CHECK(ggml_vec_index_contains(tombstone_idx, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_idx, appended_id) == 1);

            std::array<float, 4> scores{};
            std::array<uint64_t, 4> out_ids{};
            CHECK(ggml_vec_index_search(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }
            CHECK(out_ids[0] == appended_id);

            const std::array<uint64_t, 2> allowed = { ids[1], appended_id };
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_filtered(
                tombstone_idx, seeds[1].data(), 1, /*k=*/2,
                allowed.data(), static_cast<int>(allowed.size()),
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);

            auto * filter = ggml_vec_index_filter_create(
                tombstone_idx, allowed.data(), static_cast<int>(allowed.size()));
            CHECK(filter != nullptr);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_prepared_filtered(
                tombstone_idx, filter, seeds[1].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);
            ggml_vec_index_filter_free(filter);

            CHECK(ggml_vec_index_build_ivf(tombstone_idx, /*n_lists=*/8, /*n_iter=*/2) ==
                  GGML_VEC_INDEX_OK);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4, /*nprobe=*/8,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }

            auto * stale_filter = ggml_vec_index_filter_create(
                tombstone_idx, allowed.data(), static_cast<int>(allowed.size()));
            CHECK(stale_filter != nullptr);
            CHECK(ggml_vec_index_compact(tombstone_idx) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(tombstone_idx) == 4);
            CHECK(ggml_vec_index_contains(tombstone_idx, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_idx, appended_id) == 1);
            CHECK(ggml_vec_index_search_prepared_filtered(
                tombstone_idx, stale_filter, seeds[1].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
            ggml_vec_index_filter_free(stale_filter);
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/1, /*nprobe=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_filtered(
                tombstone_idx, seeds[1].data(), 1, /*k=*/2,
                allowed.data(), static_cast<int>(allowed.size()),
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);
            CHECK(ggml_vec_index_build_ivf(tombstone_idx, /*n_lists=*/8, /*n_iter=*/2) ==
                  GGML_VEC_INDEX_OK);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4, /*nprobe=*/8,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }

            CHECK(ggml_vec_index_write(tombstone_idx, tombstone_path.c_str()) ==
                  GGML_VEC_INDEX_OK);
            ggml_vec_index_free(tombstone_idx);

            auto * tombstone_loaded = ggml_vec_index_load(tombstone_path.c_str());
            CHECK(tombstone_loaded != nullptr);
            CHECK(ggml_vec_index_bit_width(tombstone_loaded) == bit_width);
            CHECK(ggml_vec_index_len(tombstone_loaded) == 4);
            CHECK(ggml_vec_index_contains(tombstone_loaded, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_loaded, appended_id) == 1);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search(
                tombstone_loaded, seeds[1].data(), 1, /*k=*/4,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }
            ggml_vec_index_free(tombstone_loaded);
            std::filesystem::remove(tombstone_path);
        }
    }

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
        const uint64_t plain_delta_bound_id = (1ULL << 41) + 6ULL;
        CHECK(ggml_vec_index_add(
            base_only, seeds[2].data(), 1, &plain_delta_bound_id) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(base_only, ids[0]) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_compact(base_only) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(base_only);

        const uint64_t reserved_delta_id = UINT64_MAX;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &reserved_delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(
            base, reserved_delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(base) == 2);
        CHECK(!std::filesystem::exists(delta_path));

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
        const size_t first_record_offset = delta_log_header_size(corrupt_delta);
        corrupt_delta[delta_record_state_offset(corrupt_delta, first_record_offset)] ^= 1;
        write_file_bytes(corrupt_delta_path, corrupt_delta);
        auto * corrupt_delta_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(corrupt_delta_loaded == nullptr);
        ggml_vec_index_free(corrupt_delta_loaded);

        std::vector<uint8_t> forged_intermediate_delta = read_file_bytes(delta_path);
        const size_t forged_first_record_offset =
            delta_log_header_size(forged_intermediate_delta);
        forged_intermediate_delta[
            delta_record_state_offset(forged_intermediate_delta, forged_first_record_offset)] ^= 1;
        refresh_delta_record_crc(forged_intermediate_delta, forged_first_record_offset);
        write_file_bytes(corrupt_delta_path, forged_intermediate_delta);
        auto * forged_intermediate_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(forged_intermediate_loaded == nullptr);
        ggml_vec_index_free(forged_intermediate_loaded);

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

        const uint64_t missing_remove_id = (1ULL << 41) + 123ULL;
        const std::string missing_remove_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-missing-remove-delta.tvid").string();
        uint64_t hash_xor = 0;
        uint64_t hash_sum = 0;
        uint64_t hash_sum_rot = 0;
        for (size_t i = 0; i < 2; ++i) {
            const uint64_t hash = slot_state_hash_f32(ids[i], seeds[i]);
            hash_xor ^= hash;
            hash_sum += hash;
            hash_sum_rot += rotl64(hash, 17);
        }
        const uint32_t base_token =
            f32_state_token(kDim, 2, hash_xor, hash_sum, hash_sum_rot);
        // Claim an unchanged state so this only fails if replay rejects the
        // missing remove, not because final state validation catches it later.
        const uint32_t claimed_post_remove_token = base_token;
        std::vector<uint8_t> missing_remove_log = {
            'T', 'V', 'D', 'L',
            2, 32, 0, 0,
        };
        append_u32_le(missing_remove_log, kDim);
        append_u32_le(missing_remove_log, base_token);
        const size_t record_offset = missing_remove_log.size();
        missing_remove_log.push_back(2); // remove
        missing_remove_log.insert(missing_remove_log.end(), { 0, 0, 0 });
        append_u32_le(missing_remove_log, 1);
        append_u64_le(missing_remove_log, sizeof(uint64_t));
        append_u32_le(missing_remove_log, 0); // record CRC placeholder
        append_u32_le(missing_remove_log, claimed_post_remove_token);
        append_u64_le(missing_remove_log, missing_remove_id);
        uint32_t record_crc = crc32c_update(
            0xffffffffu,
            missing_remove_log.data() + record_offset,
            16);
        record_crc = crc32c_update_u32(record_crc, claimed_post_remove_token);
        record_crc = crc32c_update_u64(record_crc, missing_remove_id);
        record_crc ^= 0xffffffffu;
        for (int i = 0; i < 4; ++i) {
            missing_remove_log[record_offset + 16 + static_cast<size_t>(i)] =
                static_cast<uint8_t>(record_crc >> (8 * i));
        }
        write_file_bytes(missing_remove_delta_path, missing_remove_log);
        auto * missing_remove_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), missing_remove_delta_path.c_str());
        CHECK(missing_remove_loaded == nullptr);
        ggml_vec_index_free(missing_remove_loaded);
        std::filesystem::remove(missing_remove_delta_path);
        std::filesystem::remove(missing_remove_delta_path + ".lock");

        const std::vector<uint8_t> pre_compact_delta = read_file_bytes(delta_path);
        const std::vector<uint8_t> pre_same_path_snapshot = read_file_bytes(snapshot_path);
        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), snapshot_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(read_file_bytes(snapshot_path) == pre_same_path_snapshot);
        auto * same_path_snapshot = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(same_path_snapshot != nullptr);
        CHECK(ggml_vec_index_len(same_path_snapshot) == 2);
        CHECK(ggml_vec_index_contains(same_path_snapshot, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(same_path_snapshot, delta_id) == 0);
        ggml_vec_index_free(same_path_snapshot);

        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 48);

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

        const uint64_t post_crash_compact_id = (1ULL << 41) + 10ULL;
        CHECK(ggml_vec_index_add_logged(
            compacted_with_old_log, seeds[3].data(), 1, &post_crash_compact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_old_log_append = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_old_log_append != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_old_log_append) == 3);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, delta_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, post_crash_compact_id) == 1);
        ggml_vec_index_free(replayed_after_old_log_append);
        ggml_vec_index_free(compacted_with_old_log);

        write_file_bytes(delta_path, pre_compact_delta);
        auto * recompact_from_old_log = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(recompact_from_old_log != nullptr);
        CHECK(ggml_vec_index_compact_delta(
            recompact_from_old_log, snapshot_path.c_str(), delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 48);
        const uint64_t post_recompact_id = (1ULL << 41) + 11ULL;
        CHECK(ggml_vec_index_add_logged(
            recompact_from_old_log, seeds[3].data(), 1, &post_recompact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_recompact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_recompact != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_recompact) == 3);
        CHECK(ggml_vec_index_contains(replayed_after_recompact, post_recompact_id) == 1);
        ggml_vec_index_free(replayed_after_recompact);
        ggml_vec_index_free(recompact_from_old_log);

        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        auto * current_after_recompact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(current_after_recompact != nullptr);
        CHECK(ggml_vec_index_contains(current_after_recompact, post_recompact_id) == 1);
        const uint64_t post_compact_id = (1ULL << 41) + 8ULL;
        CHECK(ggml_vec_index_add_logged(
            current_after_recompact, seeds[3].data(), 1, &post_compact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_compact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_compact != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_compact) == 4);
        CHECK(ggml_vec_index_contains(replayed_after_compact, post_recompact_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_compact, post_compact_id) == 1);
        ggml_vec_index_free(replayed_after_compact);
        ggml_vec_index_free(current_after_recompact);

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

    // Delta replay supports tombstone delete followed by re-adding the same ID.
    {
        for (int bit_width : { 32, 8, 4 }) {
            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-delta-tombstone-" +
                  std::to_string(bit_width) + ".tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-delta-tombstone-" +
                  std::to_string(bit_width) + ".tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);

            auto * delta_tombstone = ggml_vec_index_create(kDim, bit_width);
            CHECK(delta_tombstone != nullptr);
            std::vector<float> base_vecs;
            base_vecs.insert(base_vecs.end(), seeds[0].begin(), seeds[0].end());
            base_vecs.insert(base_vecs.end(), seeds[1].begin(), seeds[1].end());
            CHECK(ggml_vec_index_add(
                delta_tombstone, base_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(
                delta_tombstone, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

            CHECK(ggml_vec_index_remove_logged(
                delta_tombstone, ids[0], delta_path.c_str()) == 1);
            CHECK(ggml_vec_index_add_logged(
                delta_tombstone, seeds[2].data(), 1, &ids[0], delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(delta_tombstone) == 2);
            CHECK(ggml_vec_index_contains(delta_tombstone, ids[0]) == 1);

            auto * replayed_tombstone = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(replayed_tombstone != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_tombstone) == bit_width);
            CHECK(ggml_vec_index_len(replayed_tombstone) == 2);
            CHECK(ggml_vec_index_contains(replayed_tombstone, ids[0]) == 1);
            CHECK(ggml_vec_index_contains(replayed_tombstone, ids[1]) == 1);

            std::array<float, 2> scores{};
            std::array<uint64_t, 2> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_tombstone, seeds[2].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == ids[0]);

            ggml_vec_index_free(replayed_tombstone);
            ggml_vec_index_free(delta_tombstone);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
        }
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
            CHECK(ggml_vec_index_load_mmap(v1_path.c_str()) == nullptr);
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

    // mmap loading maps the vector section read-only and keeps search parity.
    {
        const std::vector<uint64_t> mmap_ids = {
            (1ULL << 38) + 1ULL,
            (1ULL << 38) + 2ULL,
            (1ULL << 38) + 3ULL,
            (1ULL << 38) + 4ULL,
        };
        std::array<float, 4> normal_scores{};
        std::array<float, 4> mmap_scores{};
        std::array<uint64_t, 4> normal_ids{};
        std::array<uint64_t, 4> mmap_out_ids{};
        const std::vector<float> query = normalize({0.5f, -0.25f, 0.75f, 0.125f});

        for (int bit_width : { 32, 8, 4 }) {
            const auto mmap_tmp = std::filesystem::temp_directory_path() /
                                  ("ggml-vector-index-mmap-" + std::to_string(bit_width) + ".tvim");
            const auto mmap_copy_tmp = std::filesystem::temp_directory_path() /
                                       ("ggml-vector-index-mmap-copy-" + std::to_string(bit_width) + ".tvim");
            const auto mmap_delta_tmp = std::filesystem::temp_directory_path() /
                                        ("ggml-vector-index-mmap-delta-" + std::to_string(bit_width) + ".tvid");
            const std::string mmap_path = mmap_tmp.string();
            const std::string mmap_copy_path = mmap_copy_tmp.string();
            const std::string mmap_delta_path = mmap_delta_tmp.string();
            std::filesystem::remove(mmap_path);
            std::filesystem::remove(mmap_copy_path);
            std::filesystem::remove(mmap_delta_path);
            std::filesystem::remove(mmap_delta_path + ".lock");

            auto * source = ggml_vec_index_create(kDim, bit_width);
            CHECK(source != nullptr);
            CHECK(ggml_vec_index_add(
                source, vecs.data(), static_cast<int>(mmap_ids.size()), mmap_ids.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(source, mmap_path.c_str()) == GGML_VEC_INDEX_OK);

            auto * normal = ggml_vec_index_load(mmap_path.c_str());
            auto * mapped = ggml_vec_index_load_mmap(mmap_path.c_str());
            CHECK(normal != nullptr);
            CHECK(mapped != nullptr);
            CHECK(ggml_vec_index_bit_width(mapped) == bit_width);
            CHECK(ggml_vec_index_len(mapped) == static_cast<int>(mmap_ids.size()));

            CHECK(ggml_vec_index_search(
                normal, query.data(), 1, /*k=*/4,
                normal_scores.data(), normal_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search(
                mapped, query.data(), 1, /*k=*/4,
                mmap_scores.data(), mmap_out_ids.data()) == GGML_VEC_INDEX_OK);
            for (int i = 0; i < 4; ++i) {
                CHECK(mmap_out_ids[i] == normal_ids[i]);
                CHECK(std::fabs(mmap_scores[i] - normal_scores[i]) <= 1e-6f);
            }

            CHECK(ggml_vec_index_build_ivf(mapped, /*n_lists=*/2, /*n_iter=*/2)
                  == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search_ivf(
                mapped, query.data(), 1, /*k=*/2, /*nprobe=*/2,
                mmap_scores.data(), mmap_out_ids.data()) == GGML_VEC_INDEX_OK);

            const uint64_t new_id = (1ULL << 38) + 99ULL;
            CHECK(ggml_vec_index_add(mapped, seeds[0].data(), 1, &new_id)
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_remove(mapped, mmap_ids[0])
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_compact(mapped) == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_compact_delta(
                mapped, mmap_copy_path.c_str(), mmap_delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            auto * compacted = ggml_vec_index_load_with_delta(
                mmap_copy_path.c_str(), mmap_delta_path.c_str());
            CHECK(compacted != nullptr);
            CHECK(ggml_vec_index_len(compacted) == static_cast<int>(mmap_ids.size()));
            CHECK(ggml_vec_index_write(mapped, mmap_path.c_str())
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_write(mapped, mmap_copy_path.c_str()) == GGML_VEC_INDEX_OK);
            auto * copied = ggml_vec_index_load(mmap_copy_path.c_str());
            CHECK(copied != nullptr);
            CHECK(ggml_vec_index_len(copied) == static_cast<int>(mmap_ids.size()));

            ggml_vec_index_free(copied);
            ggml_vec_index_free(compacted);
            ggml_vec_index_free(mapped);
            ggml_vec_index_free(normal);
            ggml_vec_index_free(source);
            std::filesystem::remove(mmap_path);
            std::filesystem::remove(mmap_copy_path);
            std::filesystem::remove(mmap_delta_path);
            std::filesystem::remove(mmap_delta_path + ".lock");
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

    // SIMD safety-boundary rounding must not expose non-finite scores.
    {
        constexpr int boundary_dim = 16;
        const std::vector<float> boundary_vector(boundary_dim, 1.0f);
        const std::vector<float> boundary_query(
            boundary_dim, FLT_MAX / static_cast<float>(boundary_dim));
        const uint64_t boundary_id = 123456791ULL;

        for (int bit_width : { 8, 4 }) {
            auto * boundary_idx = ggml_vec_index_create(boundary_dim, bit_width);
            CHECK(boundary_idx != nullptr);
            CHECK(ggml_vec_index_add(
                boundary_idx, boundary_vector.data(), 1, &boundary_id) ==
                GGML_VEC_INDEX_OK);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                boundary_idx, boundary_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == boundary_id);
            CHECK(std::isfinite(scores[0]));

            ggml_vec_index_free(boundary_idx);
        }
    }

    // Large finite terms can overflow float intermediates even when the final
    // dot product is representable after cancellation.
    {
        constexpr int cancel_dim = 2;
        const std::array<float, cancel_dim> cancel_vector = { 1e30f, 1e30f };
        const std::array<float, cancel_dim> cancel_query = { 1e10f, -1e10f };
        const uint64_t cancel_id = 123456790ULL;

        for (int bit_width : { 32, 8, 4 }) {
            auto * cancel_idx = ggml_vec_index_create(cancel_dim, bit_width);
            CHECK(cancel_idx != nullptr);
            CHECK(ggml_vec_index_add(
                cancel_idx, cancel_vector.data(), 1, &cancel_id) == GGML_VEC_INDEX_OK);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                cancel_idx, cancel_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == cancel_id);
            CHECK(scores[0] == 0.0f);

            ggml_vec_index_free(cancel_idx);
        }
    }

    // q4 path: packed nibbles with one f32 scale per vector.
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

    // Quantization must not depend on the caller's active rounding mode.
    {
        const int saved_rounding_mode = std::fegetround();
        CHECK(saved_rounding_mode != -1);

        auto score_after_add_with_rounding = [&](int bit_width, int rounding_mode) {
            const std::array<float, kDim> rounding_vector = {
                1.0f, 0.02f, 0.0f, 0.0f,
            };
            const std::array<float, kDim> rounding_query = {
                0.0f, 1.0f, 0.0f, 0.0f,
            };
            const uint64_t rounding_id =
                (1ULL << 57) + static_cast<uint64_t>(bit_width) +
                static_cast<uint64_t>(rounding_mode);
            CHECK(std::fesetround(rounding_mode) == 0);
            auto * rounding_idx = ggml_vec_index_create(kDim, bit_width);
            CHECK(rounding_idx != nullptr);
            CHECK(ggml_vec_index_add(
                rounding_idx, rounding_vector.data(), 1, &rounding_id) ==
                GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(saved_rounding_mode) == 0);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                rounding_idx, rounding_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == rounding_id);
            ggml_vec_index_free(rounding_idx);
            return scores[0];
        };

        for (int bit_width : { 8, 4 }) {
            const float downward_score = score_after_add_with_rounding(bit_width, FE_DOWNWARD);
            const float upward_score = score_after_add_with_rounding(bit_width, FE_UPWARD);
            CHECK(downward_score == upward_score);

            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-rounding-" + std::to_string(bit_width) + ".tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-rounding-" + std::to_string(bit_width) + ".tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(delta_path + ".lock");

            const std::array<float, kDim> rounding_vector = {
                1.0f, 0.02f, 0.0f, 0.0f,
            };
            const std::array<float, kDim> rounding_query = {
                0.0f, 1.0f, 0.0f, 0.0f,
            };
            const uint64_t rounding_id = (1ULL << 58) + static_cast<uint64_t>(bit_width);
            auto * logged_rounding = ggml_vec_index_create(kDim, bit_width);
            CHECK(logged_rounding != nullptr);
            CHECK(ggml_vec_index_write(logged_rounding, snapshot_path.c_str()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(FE_DOWNWARD) == 0);
            CHECK(ggml_vec_index_add_logged(
                logged_rounding, rounding_vector.data(), 1, &rounding_id,
                delta_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(FE_UPWARD) == 0);
            auto * replayed_rounding = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(std::fesetround(saved_rounding_mode) == 0);
            CHECK(replayed_rounding != nullptr);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_rounding, rounding_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == rounding_id);
            CHECK(scores[0] == downward_score);

            ggml_vec_index_free(replayed_rounding);
            ggml_vec_index_free(logged_rounding);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(delta_path + ".lock");
        }
        CHECK(std::fesetround(saved_rounding_mode) == 0);
    }

    // Delta replay keeps quantized storage quantized. New v4 logs store native
    // q4/q8 rows instead of f32 vectors, while v2 f32-payload logs remain readable.
    {
        for (int bit_width : { 8, 4 }) {
            const std::string suffix = std::to_string(bit_width);
            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-base.tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-log.tvid")).string();
            const std::string corrupt_delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-corrupt.tvid")).string();
            const std::string v2_delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-v2.tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(corrupt_delta_path);
            std::filesystem::remove(v2_delta_path);

            const uint64_t base_id = (1ULL << 42) + static_cast<uint64_t>(bit_width);
            const uint64_t delta_id = (1ULL << 42) + static_cast<uint64_t>(bit_width + 100);
            auto * quant_delta = ggml_vec_index_create(kDim, bit_width);
            CHECK(quant_delta != nullptr);
            CHECK(ggml_vec_index_add(
                quant_delta, seeds[0].data(), 1, &base_id) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(
                quant_delta, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add_logged(
                quant_delta, seeds[3].data(), 1, &delta_id, delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);

            std::vector<uint8_t> delta_bytes = read_file_bytes(delta_path);
            CHECK(delta_bytes.size() >= 48 + 56);
            CHECK(delta_bytes[4] == 4);
            const size_t old_f32_payload = sizeof(uint64_t) + kDim * sizeof(uint32_t);
            const size_t expected_native_payload =
                sizeof(uint64_t) + sizeof(uint32_t) +
                (bit_width == 4 ? (kDim + 1) / 2 : kDim);
            CHECK(read_u64_le_at(delta_bytes, delta_log_header_size(delta_bytes) + 8) ==
                  expected_native_payload);
            CHECK(expected_native_payload < old_f32_payload);

            auto * replayed_quant = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(replayed_quant != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_quant) == bit_width);
            CHECK(ggml_vec_index_len(replayed_quant) == 2);
            CHECK(ggml_vec_index_contains(replayed_quant, delta_id) == 1);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_quant, seeds[3].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == delta_id);

            std::vector<uint8_t> corrupt_bytes = delta_bytes;
            const size_t record_offset = delta_log_header_size(corrupt_bytes);
            const size_t payload_offset = delta_record_payload_offset(corrupt_bytes, record_offset);
            const size_t scale_offset = payload_offset + sizeof(uint64_t);
            write_u32_le_at(corrupt_bytes, scale_offset, 0);
            refresh_delta_record_crc(corrupt_bytes, record_offset);
            write_file_bytes(corrupt_delta_path, corrupt_bytes);
            auto * corrupt_loaded = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), corrupt_delta_path.c_str());
            CHECK(corrupt_loaded == nullptr);
            ggml_vec_index_free(corrupt_loaded);

            CHECK(ggml_vec_index_compact_delta(
                quant_delta, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(std::filesystem::file_size(delta_path) == 48);
            auto * compacted_quant = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(compacted_quant != nullptr);
            CHECK(ggml_vec_index_bit_width(compacted_quant) == bit_width);
            CHECK(ggml_vec_index_contains(compacted_quant, delta_id) == 1);
            ggml_vec_index_free(compacted_quant);

            CHECK(ggml_vec_index_compact_delta(
                quant_delta, snapshot_path.c_str(), v2_delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> compacted_v4 = read_file_bytes(v2_delta_path);
            CHECK(compacted_v4.size() == 48);
            std::vector<uint8_t> empty_v2(16, 0);
            std::memcpy(empty_v2.data(), compacted_v4.data(), 12);
            empty_v2[4] = 2;
            write_u32_le_at(empty_v2, 12, state_token_from_wide_log_header(compacted_v4, kDim));
            write_file_bytes(v2_delta_path, empty_v2);
            const uint64_t v2_delta_id =
                (1ULL << 42) + static_cast<uint64_t>(bit_width + 200);
            CHECK(ggml_vec_index_add_logged(
                quant_delta, seeds[1].data(), 1, &v2_delta_id, v2_delta_path.c_str()) ==
                GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_contains(quant_delta, v2_delta_id) == 0);

            std::vector<uint8_t> v2_payload;
            append_u64_le(v2_payload, v2_delta_id);
            for (float value : seeds[1]) {
                append_f32_le(v2_payload, value);
            }
            CHECK(v2_payload.size() == old_f32_payload);

            uint64_t wide_n_active = read_u64_le_at(compacted_v4, 16);
            uint64_t wide_hash_xor = read_u64_le_at(compacted_v4, 24);
            uint64_t wide_hash_sum = read_u64_le_at(compacted_v4, 32);
            uint64_t wide_hash_sum_rot = read_u64_le_at(compacted_v4, 40);
            const uint64_t v2_hash =
                slot_state_hash_quantized(bit_width, v2_delta_id, seeds[1]);
            ++wide_n_active;
            wide_hash_xor ^= v2_hash;
            wide_hash_sum += v2_hash;
            wide_hash_sum_rot += rotl64(v2_hash, 17);
            const uint32_t v2_post_token = state_token_from_wide_values(
                bit_width,
                kDim,
                wide_n_active,
                wide_hash_xor,
                wide_hash_sum,
                wide_hash_sum_rot);

            std::vector<uint8_t> v2_log = empty_v2;
            const size_t v2_record_offset = v2_log.size();
            v2_log.push_back(1); // add
            v2_log.insert(v2_log.end(), { 0, 0, 0 });
            append_u32_le(v2_log, 1);
            append_u64_le(v2_log, v2_payload.size());
            append_u32_le(v2_log, 0); // record CRC placeholder
            append_u32_le(v2_log, v2_post_token);
            v2_log.insert(v2_log.end(), v2_payload.begin(), v2_payload.end());
            uint32_t v2_record_crc = crc32c_update(
                0xffffffffu,
                v2_log.data() + v2_record_offset,
                16);
            v2_record_crc = crc32c_update_u32(v2_record_crc, v2_post_token);
            v2_record_crc = crc32c_update(v2_record_crc, v2_payload.data(), v2_payload.size());
            write_u32_le_at(v2_log, v2_record_offset + 16, v2_record_crc ^ 0xffffffffu);
            write_file_bytes(v2_delta_path, v2_log);

            auto * replayed_v2 = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), v2_delta_path.c_str());
            CHECK(replayed_v2 != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_v2) == bit_width);
            CHECK(ggml_vec_index_contains(replayed_v2, v2_delta_id) == 1);

            ggml_vec_index_free(replayed_v2);
            ggml_vec_index_free(replayed_quant);
            ggml_vec_index_free(quant_delta);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(corrupt_delta_path);
            std::filesystem::remove(v2_delta_path);
            std::filesystem::remove(delta_path + ".lock");
            std::filesystem::remove(v2_delta_path + ".lock");
            std::filesystem::remove(corrupt_delta_path + ".lock");
        }
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

    // q8 path: stores quantized codes, searches directly against
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
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                const size_t id_offset =
                    32 + 2 * sizeof(float) + 2 * kDim * sizeof(int8_t);
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    bytes[id_offset + i] = 0xff;
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

    std::filesystem::remove_all(test_temp_dir);

    std::printf("test-vector-index: OK\n");
    return 0;
}
