// ggml-vector-index-persistence.cpp - f32 snapshot shim for early search split.

#include "ggml-vector-index-internal.h"

MappedFile::~MappedFile() = default;

void test_maybe_throw_bad_alloc() {}
bool test_consume_write_bytes(size_t) { return true; }
void test_wait_after_delta_validate() {}
void test_wait_after_load_with_delta_snapshot() {}

bool is_supported_bit_width(int bit_width) {
    return bit_width == 4 || bit_width == 8 || bit_width == 32;
}

bool is_valid_id(uint64_t id) {
    return id != UINT64_MAX;
}

bool all_finite(const float * values, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

uint32_t float_to_u32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float u32_to_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void put_u32_le(uint8_t * dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v >> 0);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

uint32_t get_u32_le(const uint8_t * src) {
    return (static_cast<uint32_t>(src[0]) << 0) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

void put_u64_le(uint8_t * dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

uint64_t get_u64_le(const uint8_t * src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return v;
}

uint32_t crc32c_update(uint32_t crc, const void * data, size_t len) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

uint32_t crc32c_update_u32(uint32_t crc, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    return crc32c_update(crc, bytes, sizeof(bytes));
}

uint32_t crc32c_update_u64(uint32_t crc, uint64_t v) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    return crc32c_update(crc, bytes, sizeof(bytes));
}

bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool expected_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected) {
    uint64_t values = 0;
    uint64_t vector_bytes = 0;
    uint64_t id_bytes = 0;
    uint64_t payload_bytes = 0;
    return checked_mul_u64(n, dim, values) &&
           checked_mul_u64(values, sizeof(float), vector_bytes) &&
           checked_mul_u64(n, sizeof(uint64_t), id_bytes) &&
           checked_add_u64(vector_bytes, id_bytes, payload_bytes) &&
           checked_add_u64(kTvimV1HeaderSize, payload_bytes, expected);
}

bool supported_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected) {
    return expected_v1_snapshot_size(n, dim, expected) && expected <= kMaxSnapshotBytes;
}

int snapshot_write_v1_preflight(size_t n, size_t dim) {
    if (n > kMaxIndexLen || n > std::numeric_limits<uint32_t>::max()) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    uint64_t expected = 0;
    if (!supported_v1_snapshot_size(static_cast<uint64_t>(n), static_cast<uint64_t>(dim), expected)) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    return GGML_VEC_INDEX_OK;
}

bool filesystem_path_from_utf8(const char * path, std::filesystem::path & out) {
    if (path == nullptr) {
        return false;
    }
    out = std::filesystem::path(path);
    return true;
}

bool filesystem_paths_equal(const char * lhs, const char * rhs) {
    std::error_code ec;
    return std::filesystem::equivalent(lhs, rhs, ec) && !ec;
}

uint64_t slot_state_hash(const ggml_vec_index & idx, size_t slot) {
    uint64_t h = idx.slot_to_id[slot] ^ (static_cast<uint64_t>(idx.bit_width) << 32);
    h ^= static_cast<uint64_t>(idx.dim) * 0x9e3779b185ebca87ull;
    return h;
}

void add_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum += hash;
    idx.state_hash_sum_rot += (hash << 17) | (hash >> 47);
}

void remove_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum -= hash;
    idx.state_hash_sum_rot -= (hash << 17) | (hash >> 47);
}

void rebuild_state_hash(ggml_vec_index & idx) {
    idx.state_hash_xor = 0;
    idx.state_hash_sum = 0;
    idx.state_hash_sum_rot = 0;
    for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
        if (slot_is_active(idx, slot)) {
            add_state_hash(idx, slot_state_hash(idx, slot));
        }
    }
}

uint32_t index_state_crc32c(const ggml_vec_index &) { return 0; }
uint32_t index_state_crc32c_after_remove(const ggml_vec_index &, uint64_t) { return 0; }
uint32_t index_state_token(const ggml_vec_index &) { return 0; }
uint32_t index_state_token_after_remove(const ggml_vec_index &, uint64_t) { return 0; }
DeltaStateWide index_state_wide(const ggml_vec_index &) { return {}; }
DeltaStateWide index_state_wide_after_remove(const ggml_vec_index &, uint64_t) { return {}; }
DeltaStateKind delta_state_kind_for_format(DeltaLogFormat) { return DeltaStateKind::legacy_crc; }
DeltaLogFormat delta_log_format_for_append(const char *) { return DeltaLogFormat::v1; }
uint32_t current_delta_state(const ggml_vec_index &, DeltaStateKind) { return 0; }
DeltaStateWide current_delta_state_wide(const ggml_vec_index &) { return {}; }
void invalidate_delta_tail_cache(ggml_vec_index &) {}
bool bind_delta_log_path(ggml_vec_index &, const char *) { return false; }
bool delta_log_matches_index_unlocked(const ggml_vec_index_t *, const char *) { return false; }
bool validate_logged_add_args(const ggml_vec_index_t *, const float *, int, const uint64_t *) { return false; }
int check_logged_add_duplicates(const ggml_vec_index_t *, int, const uint64_t *) { return GGML_VEC_INDEX_E_INVALID_ARG; }
bool build_add_delta_payload_f32(const ggml_vec_index_t *, const float *, int, const uint64_t *, std::vector<uint8_t> &) { return false; }
bool build_add_delta_payload_from_slots(const ggml_vec_index_t *, size_t, int, std::vector<uint8_t> &) { return false; }
std::vector<uint8_t> build_remove_delta_payload(uint64_t) { return {}; }
DeltaAppendResult append_delta_record_locked(ggml_vec_index &, const char *, DeltaLogFormat, uint8_t, uint32_t, uint32_t, uint32_t, const DeltaStateWide &, const DeltaStateWide &, const std::vector<uint8_t> &) { return { GGML_VEC_INDEX_E_INVALID_ARG, false }; }

static bool write_u32(std::ofstream & f, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    f.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    return static_cast<bool>(f);
}

static bool write_u64(std::ofstream & f, uint64_t v) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    f.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    return static_cast<bool>(f);
}

static bool read_u32(std::ifstream & f, uint32_t & v) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    return true;
}

static bool read_u64(std::ifstream & f, uint64_t & v) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u64_le(bytes);
    return true;
}

int ggml_vec_index_write(ggml_vec_index_t * idx, const char * path) {
    if (idx == nullptr || path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        if (idx->bit_width != 32 || idx->read_only_mmap || idx->delta_log_bound) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n = active_count(*idx);
        const int preflight_status = snapshot_write_v1_preflight(n, static_cast<size_t>(idx->dim));
        if (preflight_status != GGML_VEC_INDEX_OK) {
            return preflight_status;
        }
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            return GGML_VEC_INDEX_E_IO;
        }
        const uint32_t n_le = static_cast<uint32_t>(n);
        f.write(reinterpret_cast<const char *>(kTvimMagic), 4);
        f.put(static_cast<char>(kTvimVersionV1));
        f.put(32);
        f.put(0);
        f.put(0);
        if (!write_u32(f, static_cast<uint32_t>(idx->dim)) || !write_u32(f, n_le)) {
            return GGML_VEC_INDEX_E_IO;
        }
        const size_t dim = static_cast<size_t>(idx->dim);
        const float * data = f32_data_ptr(*idx);
        for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
            if (!slot_is_active(*idx, slot)) {
                continue;
            }
            for (size_t i = 0; i < dim; ++i) {
                if (!write_u32(f, float_to_u32(data[slot * dim + i]))) {
                    return GGML_VEC_INDEX_E_IO;
                }
            }
        }
        for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
            if (slot_is_active(*idx, slot) && !write_u64(f, idx->slot_to_id[slot])) {
                return GGML_VEC_INDEX_E_IO;
            }
        }
        return static_cast<bool>(f) ? GGML_VEC_INDEX_OK : GGML_VEC_INDEX_E_IO;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_load_ex(const char * path, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    if (path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            return GGML_VEC_INDEX_E_IO;
        }
        f.seekg(0, std::ios::end);
        if (!f) {
            return GGML_VEC_INDEX_E_IO;
        }
        const std::streamoff end_off = f.tellg();
        if (end_off < 0) {
            return GGML_VEC_INDEX_E_IO;
        }
        const uint64_t file_size = static_cast<uint64_t>(end_off);
        f.seekg(0, std::ios::beg);
        if (!f) {
            return GGML_VEC_INDEX_E_IO;
        }
        uint8_t magic[4];
        f.read(reinterpret_cast<char *>(magic), sizeof(magic));
        if (!f || std::memcmp(magic, kTvimMagic, sizeof(magic)) != 0) {
            return GGML_VEC_INDEX_E_BAD_MAGIC;
        }
        const int version = f.get();
        const int bit_width = f.get();
        const int r0 = f.get();
        const int r1 = f.get();
        if (version != kTvimVersionV1 || bit_width != 32 || r0 != 0 || r1 != 0) {
            return GGML_VEC_INDEX_E_BAD_VERSION;
        }
        uint32_t dim = 0;
        uint32_t n = 0;
        if (!read_u32(f, dim) || !read_u32(f, n) || dim == 0 || dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return GGML_VEC_INDEX_E_IO;
        }
        uint64_t expected_size = 0;
        if (!supported_v1_snapshot_size(n, dim, expected_size) || file_size != expected_size) {
            return GGML_VEC_INDEX_E_IO;
        }
        std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim));
        for (float & v : vectors) {
            uint32_t bits = 0;
            if (!read_u32(f, bits)) {
                return GGML_VEC_INDEX_E_IO;
            }
            v = u32_to_float(bits);
            if (!std::isfinite(v)) {
                return GGML_VEC_INDEX_E_IO;
            }
        }
        std::vector<uint64_t> ids(n);
        for (uint64_t & id : ids) {
            if (!read_u64(f, id) || !is_valid_id(id)) {
                return GGML_VEC_INDEX_E_IO;
            }
        }
        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(static_cast<int>(dim), 32), ggml_vec_index_free);
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_OOM;
        }
        const int rc = ggml_vec_index_add(idx.get(), vectors.data(), static_cast<int>(n), ids.data());
        if (rc != GGML_VEC_INDEX_OK) {
            return rc;
        }
        idx->delta_log_start_allowed = true;
        *out = idx.release();
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_IO;
    }
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    ggml_vec_index_t * out = nullptr;
    return ggml_vec_index_load_ex(path, &out) == GGML_VEC_INDEX_OK ? out : nullptr;
}

int ggml_vec_index_load_mmap_ex(const char *, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    return GGML_VEC_INDEX_E_BAD_VERSION;
}

ggml_vec_index_t * ggml_vec_index_load_mmap(const char *) {
    return nullptr;
}

int ggml_vec_index_load_with_delta_ex(const char *, const char *, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    return GGML_VEC_INDEX_E_INVALID_ARG;
}

ggml_vec_index_t * ggml_vec_index_load_with_delta(const char *, const char *) {
    return nullptr;
}

int ggml_vec_index_compact_delta(ggml_vec_index_t *, const char *, const char *) {
    return GGML_VEC_INDEX_E_INVALID_ARG;
}
