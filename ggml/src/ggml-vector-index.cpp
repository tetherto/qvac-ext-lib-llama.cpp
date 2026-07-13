// ggml-vector-index.cpp - CPU implementation of the fabric vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors or per-vector symmetric q8 codes. ID map uses
// std::unordered_map<uint64_t, size_t> for lookup and a parallel vector for
// the slot->id reverse map. Remove uses swap-with-last.
//
// Search: dot product across all slots + min-heap of size k. q8 search scores
// directly against stored codes and per-vector scales, with ARM NEON when
// available and a scalar fallback.

#include "ggml-vector-index.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GGML_VEC_INDEX_USE_NEON 1
#else
#define GGML_VEC_INDEX_USE_NEON 0
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" {
void ggml_vec_index_test_set_oom_countdown(int64_t countdown);
void ggml_vec_index_test_set_write_fail_after(int64_t bytes);
}
#endif

namespace {

constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
constexpr uint8_t  kTvimVersionV1  = 1;
constexpr uint8_t  kTvimVersion    = 2;
constexpr uint8_t  kStorageF32     = 1;
constexpr uint8_t  kStorageQ8      = 2;
constexpr uint8_t  kFlagCRC32C     = 1;
constexpr uint32_t kQParamNone     = 0;
constexpr uint32_t kQParamScaleF32 = 1;
constexpr size_t   kTvimV1HeaderSize = 16;
constexpr size_t   kTvimHeaderSize = 32;
constexpr size_t   kTvimChecksumSize = 16;

static_assert(sizeof(float) == sizeof(uint32_t), "ggml-vector-index requires float32");

#ifdef GGML_VEC_INDEX_TEST_HOOKS
std::atomic<int64_t> g_test_oom_countdown{ -1 };
std::atomic<int64_t> g_test_write_fail_after{ -1 };

void test_maybe_throw_bad_alloc() {
    const int64_t remaining = g_test_oom_countdown.load();
    if (remaining < 0) {
        return;
    }
    if (g_test_oom_countdown.fetch_sub(1) == 0) {
        throw std::bad_alloc();
    }
}

bool test_consume_write_bytes(size_t n) {
    const int64_t current = g_test_write_fail_after.load();
    if (current < 0) {
        return true;
    }
    if (static_cast<uint64_t>(current) < n) {
        g_test_write_fail_after.store(0);
        return false;
    }
    g_test_write_fail_after.fetch_sub(static_cast<int64_t>(n));
    return true;
}
#else
#define test_maybe_throw_bad_alloc() ((void) 0)
#endif

inline bool write_bytes(std::FILE * f, const void * data, size_t size) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    if (!test_consume_write_bytes(size)) {
        return false;
    }
#endif
    return std::fwrite(data, 1, size, f) == size;
}

void put_u32_le(uint8_t * dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v >> 0);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

void put_u64_le(uint8_t * dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

uint32_t get_u32_le(const uint8_t * src) {
    return (static_cast<uint32_t>(src[0]) << 0)  |
           (static_cast<uint32_t>(src[1]) << 8)  |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

uint64_t get_u64_le(const uint8_t * src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return v;
}

uint32_t float_to_u32(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

float u32_to_float(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

bool write_u32_le(std::FILE * f, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    return write_bytes(f, bytes, sizeof(bytes));
}

bool read_u32_le(std::ifstream & f, uint32_t & v) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    return true;
}

bool read_u64_le(std::ifstream & f, uint64_t & v) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u64_le(bytes);
    return true;
}

uint32_t crc32c_update(uint32_t crc, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

bool read_u32_le_crc(std::ifstream & f, uint32_t & v, uint32_t & crc) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

bool read_u64_le_crc(std::ifstream & f, uint64_t & v, uint32_t & crc) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u64_le(bytes);
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

bool write_u32_le_crc(std::FILE * f, uint32_t v, uint32_t & crc) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    if (!write_bytes(f, bytes, sizeof(bytes))) {
        return false;
    }
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

bool write_u64_le_crc(std::FILE * f, uint64_t v, uint32_t & crc) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    if (!write_bytes(f, bytes, sizeof(bytes))) {
        return false;
    }
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

// Top-k via min-heap of (score, id). The heap holds at most `k` candidates;
// each new score is compared against the smallest in the heap.
struct ScoreId {
    float    score;
    uint64_t id;
};

struct MinHeapCmp {
    bool operator()(const ScoreId & a, const ScoreId & b) const {
        // Min-heap by score (smallest score at the top).
        return a.score > b.score;
    }
};

bool is_supported_bit_width(int bit_width) {
    return bit_width == 8 || bit_width == 32;
}

bool all_finite(const float * values, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool expected_file_size(
        uint64_t header_size,
        uint64_t n,
        uint64_t dim,
        uint64_t qparam_bytes,
        uint64_t component_bytes,
        uint64_t & size) {
    uint64_t qparams = 0;
    uint64_t components = 0;
    uint64_t ids = 0;
    uint64_t total = header_size;
    if (!checked_mul_u64(n, qparam_bytes, qparams) ||
        !checked_mul_u64(n, dim, components) ||
        !checked_mul_u64(components, component_bytes, components) ||
        !checked_mul_u64(n, sizeof(uint64_t), ids) ||
        !checked_add_u64(total, qparams, total) ||
        !checked_add_u64(total, components, total) ||
        !checked_add_u64(total, ids, total)) {
        return false;
    }
    size = total;
    return true;
}

struct TempFile {
    std::FILE * stream = nullptr;
#ifdef _WIN32
    std::wstring path;
#else
    std::string path;
#endif

    TempFile() = default;
    ~TempFile();
    TempFile(const TempFile &) = delete;
    TempFile & operator=(const TempFile &) = delete;
};

#ifdef _WIN32

bool utf8_to_wide(const char * src, std::wstring & dst) {
    const int size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, src, -1, nullptr, 0);
    if (size <= 0) {
        return false;
    }
    std::vector<wchar_t> buffer(static_cast<size_t>(size));
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, src, -1, buffer.data(), size) != size) {
        return false;
    }
    dst.assign(buffer.data());
    return true;
}

bool open_temp_file(const char * path, TempFile & temp) {
    std::wstring destination;
    if (!utf8_to_wide(path, destination)) {
        return false;
    }
    static std::atomic<uint64_t> sequence{ 0 };
    for (int attempt = 0; attempt < 128; ++attempt) {
        wchar_t suffix[96];
        std::swprintf(
            suffix,
            sizeof(suffix) / sizeof(suffix[0]),
            L".tmp.%lu.%lu.%llu",
            static_cast<unsigned long>(GetCurrentProcessId()),
            static_cast<unsigned long>(GetCurrentThreadId()),
            static_cast<unsigned long long>(sequence.fetch_add(1)));
        temp.path = destination + suffix;

        int fd = -1;
        const errno_t error = _wsopen_s(
            &fd,
            temp.path.c_str(),
            _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY | _O_NOINHERIT,
            _SH_DENYRW,
            _S_IREAD | _S_IWRITE);
        if (error == 0) {
            temp.stream = _wfdopen(fd, L"w+b");
            if (temp.stream != nullptr) {
                return true;
            }
            _close(fd);
            _wremove(temp.path.c_str());
            return false;
        }
        if (error != EEXIST) {
            return false;
        }
    }
    return false;
}

#else

bool open_temp_file(const char * path, TempFile & temp) {
    std::string pattern = std::string(path) + ".tmp.XXXXXX";
    std::vector<char> mutable_path(pattern.begin(), pattern.end());
    mutable_path.push_back('\0');
    const int fd = ::mkstemp(mutable_path.data());
    if (fd < 0) {
        return false;
    }
    temp.path.assign(mutable_path.data());

    struct stat destination_stat;
    if (::stat(path, &destination_stat) == 0 &&
        ::fchmod(fd, destination_stat.st_mode & 0777) != 0) {
        ::close(fd);
        std::remove(temp.path.c_str());
        return false;
    }

    temp.stream = ::fdopen(fd, "w+b");
    if (temp.stream == nullptr) {
        ::close(fd);
        std::remove(temp.path.c_str());
        return false;
    }
    return true;
}

#endif

void remove_temp_file(TempFile & temp) {
    if (temp.path.empty()) {
        return;
    }
#ifdef _WIN32
    _wremove(temp.path.c_str());
#else
    std::remove(temp.path.c_str());
#endif
    temp.path.clear();
}

TempFile::~TempFile() {
    if (stream != nullptr) {
        std::fclose(stream);
    }
    remove_temp_file(*this);
}

bool flush_and_sync(std::FILE * stream) {
    if (std::fflush(stream) != 0) {
        return false;
    }
#ifdef _WIN32
    return _commit(_fileno(stream)) == 0;
#elif defined(__APPLE__)
    const int fd = ::fileno(stream);
    return ::fcntl(fd, F_FULLFSYNC) == 0 || ::fsync(fd) == 0;
#else
    return ::fsync(::fileno(stream)) == 0;
#endif
}

bool fsync_parent_dir(const char * path) {
#ifdef _WIN32
    (void) path;
    return true;
#else
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) {
        parent = ".";
    }
    const int fd = ::open(parent.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
#endif
}

bool rename_overwrite(const TempFile & temp, const char * dst) {
#ifdef _WIN32
    std::wstring destination;
    if (!utf8_to_wide(dst, destination)) {
        return false;
    }
    if (GetFileAttributesW(destination.c_str()) != INVALID_FILE_ATTRIBUTES) {
        return ReplaceFileW(
            destination.c_str(),
            temp.path.c_str(),
            nullptr,
            REPLACEFILE_WRITE_THROUGH,
            nullptr,
            nullptr) != 0;
    }
    return MoveFileExW(
        temp.path.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(temp.path.c_str(), dst) == 0;
#endif
}

} // namespace

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" {
void ggml_vec_index_test_set_oom_countdown(int64_t countdown) {
    g_test_oom_countdown.store(countdown);
}

void ggml_vec_index_test_set_write_fail_after(int64_t bytes) {
    g_test_write_fail_after.store(bytes);
}
}
#endif

// Lifetime-managed instance state. Lives behind the opaque
// `ggml_vec_index_t` typedef.
struct ggml_vec_index {
    int dim       = 0;
    int bit_width = 32;

    // Flat row-major f32 storage for bit_width=32.
    std::vector<float> data;

    // Flat row-major q8 storage for bit_width=8 plus one scale per vector.
    std::vector<int8_t> q8_data;
    std::vector<float>  q8_scale;

    // slot -> external id (parallel to logical slot index).
    std::vector<uint64_t> slot_to_id;

    // external id -> slot.
    std::unordered_map<uint64_t, size_t> id_to_slot;
};

static bool is_q8(const ggml_vec_index & idx) {
    return idx.bit_width == 8;
}

static uint8_t storage_kind(const ggml_vec_index & idx) {
    return is_q8(idx) ? kStorageQ8 : kStorageF32;
}

static void quantize_q8_row(const float * src, int8_t * dst, int dim, float & scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    if (max_abs == 0.0f) {
        scale = 1.0f;
        std::memset(dst, 0, static_cast<size_t>(dim));
        return;
    }

    scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        scale = max_abs;
    }
    for (int i = 0; i < dim; ++i) {
        const float scaled = src[i] / scale;
        int q = static_cast<int>(std::nearbyint(scaled));
        q = std::max(-127, std::min(127, q));
        dst[i] = static_cast<int8_t>(q);
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width) {
    try {
        if (dim <= 0) {
            return nullptr;
        }
        if (!is_supported_bit_width(bit_width)) {
            return nullptr;
        }
        auto * idx = new (std::nothrow) ggml_vec_index();
        if (idx == nullptr) {
            return nullptr;
        }
        idx->dim       = dim;
        idx->bit_width = bit_width;
        return idx;
    } catch (...) {
        return nullptr;
    }
}

void ggml_vec_index_free(ggml_vec_index_t * idx) {
    delete idx;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids) {

    size_t base_slot = 0;
    size_t dim_sz    = 0;
    bool resized     = false;

    auto rollback = [&]() noexcept {
        if (idx == nullptr || !resized) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            idx->id_to_slot.erase(ids[i]);
        }
        if (is_q8(*idx)) {
            idx->q8_data.resize(base_slot * dim_sz);
            idx->q8_scale.resize(base_slot);
        } else {
            idx->data.resize(base_slot * dim_sz);
        }
        idx->slot_to_id.resize(base_slot);
    };

    try {
        if (idx == nullptr || vectors == nullptr || ids == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n < 0) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }

        // Atomic add: detect duplicates first (against existing AND in-batch),
        // bail before mutating any state.
        test_maybe_throw_bad_alloc();
        std::unordered_set<uint64_t> batch_ids;
        batch_ids.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
            if (!batch_ids.insert(ids[i]).second) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
        }

        base_slot = idx->slot_to_id.size();
        dim_sz    = static_cast<size_t>(idx->dim);
        const size_t n_sz = static_cast<size_t>(n);
        if (n_sz > std::numeric_limits<size_t>::max() - base_slot) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t new_slots = base_slot + n_sz;
        if (dim_sz != 0 && new_slots > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(vectors, n_sz * dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        resized = true;
        if (is_q8(*idx)) {
            idx->q8_data.resize(new_slots * dim_sz);
            idx->q8_scale.resize(new_slots);
        } else {
            idx->data.resize(new_slots * dim_sz);
        }
        idx->slot_to_id.resize(new_slots);
        test_maybe_throw_bad_alloc();
        idx->id_to_slot.reserve(new_slots);

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            const float * src = vectors + static_cast<size_t>(i) * dim_sz;
            if (is_q8(*idx)) {
                quantize_q8_row(
                    src,
                    idx->q8_data.data() + slot * dim_sz,
                    idx->dim,
                    idx->q8_scale[slot]);
            } else {
                std::memcpy(
                    idx->data.data() + slot * dim_sz,
                    src,
                    dim_sz * sizeof(float));
            }
            idx->slot_to_id[slot] = ids[i];
            test_maybe_throw_bad_alloc();
            idx->id_to_slot.emplace(ids[i], slot);
        }
    } catch (const std::bad_alloc &) {
        rollback();
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        rollback();
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id) {
    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        auto it = idx->id_to_slot.find(id);
        if (it == idx->id_to_slot.end()) {
            return 0;
        }
        const size_t slot     = it->second;
        const size_t last     = idx->slot_to_id.size() - 1;
        const size_t dim_sz   = static_cast<size_t>(idx->dim);

        if (slot != last) {
            // Move last vector into the freed slot and update its id mapping.
            if (is_q8(*idx)) {
                std::memcpy(
                    idx->q8_data.data() + slot * dim_sz,
                    idx->q8_data.data() + last * dim_sz,
                    dim_sz * sizeof(int8_t));
                idx->q8_scale[slot] = idx->q8_scale[last];
            } else {
                std::memcpy(
                    idx->data.data() + slot * dim_sz,
                    idx->data.data() + last * dim_sz,
                    dim_sz * sizeof(float));
            }
            const uint64_t moved_id = idx->slot_to_id[last];
            idx->slot_to_id[slot]   = moved_id;
            idx->id_to_slot[moved_id] = slot;
        }

        idx->slot_to_id.pop_back();
        if (is_q8(*idx)) {
            idx->q8_data.resize(last * dim_sz);
            idx->q8_scale.resize(last);
        } else {
            idx->data.resize(last * dim_sz);
        }
        idx->id_to_slot.erase(it);
        return 1;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return 0;
    }
    return idx->id_to_slot.count(id) != 0 ? 1 : 0;
}

void ggml_vec_index_prepare(ggml_vec_index_t * /*idx*/) {
    // Future: warm caches, materialize codebooks, etc.
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

namespace {

// Scalar dot product of two `dim`-length f32 vectors.
inline float dot(const float * a, const float * b, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

inline float dot_q8_scalar(const float * query, const int8_t * codes, float scale, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float value = static_cast<float>(codes[i]) * scale;
        acc += query[i] * value;
    }
    return acc;
}

#if GGML_VEC_INDEX_USE_NEON

inline float horizontal_sum(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    const float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
#endif
}

inline float dot_q8_neon(const float * query, const int8_t * codes, float scale, int dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const int16x8_t q16 = vmovl_s8(vld1_s8(codes + i));
        const float32x4_t q0 =
            vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))), scale);
        const float32x4_t q1 =
            vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16))), scale);
        acc0 = vmlaq_f32(acc0, vld1q_f32(query + i), q0);
        acc1 = vmlaq_f32(acc1, vld1q_f32(query + i + 4), q1);
    }

    float acc = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; i < dim; ++i) {
        const float value = static_cast<float>(codes[i]) * scale;
        acc += query[i] * value;
    }
    return acc;
}

#endif

inline float dot_q8(const float * query, const int8_t * codes, float scale, int dim) {
#if GGML_VEC_INDEX_USE_NEON
    return dot_q8_neon(query, codes, scale, dim);
#else
    return dot_q8_scalar(query, codes, scale, dim);
#endif
}

// Run a single query against all slots, write top-k into out_scores/out_ids.
// If the index holds fewer than k entries, pad with sentinels.
void search_one(
    const ggml_vec_index_t & idx,
    const float            * query,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {

    const int    dim     = idx.dim;
    const size_t n_slots = idx.slot_to_id.size();

    test_maybe_throw_bad_alloc();
    std::priority_queue<ScoreId, std::vector<ScoreId>, MinHeapCmp> heap;

    for (size_t slot = 0; slot < n_slots; ++slot) {
        const float s = is_q8(idx) ?
            dot_q8(
                query,
                idx.q8_data.data() + slot * static_cast<size_t>(dim),
                idx.q8_scale[slot],
                dim) :
            dot(
                query,
                idx.data.data() + slot * static_cast<size_t>(dim),
                dim);
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push({ s, idx.slot_to_id[slot] });
        } else if (s > heap.top().score) {
            heap.pop();
            heap.push({ s, idx.slot_to_id[slot] });
        }
    }

    // Drain the heap into a temporary descending list.
    std::vector<ScoreId> drained;
    drained.reserve(heap.size());
    while (!heap.empty()) {
        drained.push_back(heap.top());
        heap.pop();
    }
    std::reverse(drained.begin(), drained.end()); // now descending by score

    for (int i = 0; i < k; ++i) {
        if (static_cast<size_t>(i) < drained.size()) {
            out_scores[i] = drained[i].score;
            out_ids[i]    = drained[i].id;
        } else {
            out_scores[i] = -FLT_MAX;
            out_ids[i]    = UINT64_MAX;
        }
    }
}

} // namespace

int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr || queries == nullptr ||
        out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }

    try {
        const int dim = idx->dim;
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz   = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        if ((dim_sz != 0 && n_q_sz > std::numeric_limits<size_t>::max() / dim_sz) ||
            n_q_sz > std::numeric_limits<size_t>::max() / k_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(queries, n_q_sz * dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        for (int q = 0; q < n_q; ++q) {
            search_one(
                *idx,
                queries + static_cast<size_t>(q) * static_cast<size_t>(dim),
                k,
                out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
                out_ids    + static_cast<size_t>(q) * static_cast<size_t>(k));
        }
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

int ggml_vec_index_write(ggml_vec_index_t * idx, const char * path) {
    try {
        if (idx == nullptr || path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->slot_to_id.size() > std::numeric_limits<uint32_t>::max()) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n      = idx->slot_to_id.size();
        const size_t dim_sz = static_cast<size_t>(idx->dim);
        if (dim_sz != 0 && n > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }
        if (is_q8(*idx)) {
            if (idx->q8_data.size() != n * dim_sz || idx->q8_scale.size() != n) {
                return GGML_VEC_INDEX_E_INTERNAL;
            }
        } else if (idx->data.size() != n * dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }

        test_maybe_throw_bad_alloc();
        TempFile temp;
        if (!open_temp_file(path, temp)) {
            return GGML_VEC_INDEX_E_IO;
        }
        test_maybe_throw_bad_alloc();
        auto fail_io = [&]() {
            if (temp.stream != nullptr) {
                std::fclose(temp.stream);
                temp.stream = nullptr;
            }
            remove_temp_file(temp);
            return GGML_VEC_INDEX_E_IO;
        };
        std::FILE * f = temp.stream;

        // Header: 32 bytes. Layout matches the comment block in the header file.
        uint8_t header[kTvimHeaderSize] = {};
        std::memcpy(header, kTvimMagic, 4);
        header[4] = kTvimVersion;
        header[5] = static_cast<uint8_t>(idx->bit_width);
        header[6] = storage_kind(*idx);
        header[7] = kFlagCRC32C;
        const uint32_t dim_le = static_cast<uint32_t>(idx->dim);
        const uint32_t n_le   = static_cast<uint32_t>(idx->slot_to_id.size());
        put_u32_le(header + 8, dim_le);
        put_u32_le(header + 12, n_le);
        put_u32_le(header + 16, is_q8(*idx) ? kQParamScaleF32 : kQParamNone);
        put_u32_le(header + 20, is_q8(*idx) ? 4u : 0u);
        put_u32_le(header + 24, is_q8(*idx) ? 1u : 4u);
        put_u32_le(header + 28, 0);

        if (!write_bytes(f, header, sizeof(header))) {
            return fail_io();
        }

        uint32_t header_crc  = crc32c_update(0xffffffffu, header, sizeof(header));
        uint32_t qparams_crc = 0xffffffffu;
        uint32_t vectors_crc = 0xffffffffu;
        uint32_t ids_crc     = 0xffffffffu;
        if (is_q8(*idx)) {
            for (float scale : idx->q8_scale) {
                if (!write_u32_le_crc(f, float_to_u32(scale), qparams_crc)) {
                    return fail_io();
                }
            }

            if (!idx->q8_data.empty()) {
                if (!write_bytes(f, idx->q8_data.data(), idx->q8_data.size())) {
                    return fail_io();
                }
                vectors_crc =
                    crc32c_update(vectors_crc, idx->q8_data.data(), idx->q8_data.size());
            }
        } else {
            for (float v : idx->data) {
                if (!write_u32_le_crc(f, float_to_u32(v), vectors_crc)) {
                    return fail_io();
                }
            }
        }

        for (uint64_t id : idx->slot_to_id) {
            if (!write_u64_le_crc(f, id, ids_crc)) {
                return fail_io();
            }
        }

        if (!write_u32_le(f, header_crc ^ 0xffffffffu) ||
            !write_u32_le(f, qparams_crc ^ 0xffffffffu) ||
            !write_u32_le(f, vectors_crc ^ 0xffffffffu) ||
            !write_u32_le(f, ids_crc ^ 0xffffffffu)) {
            return fail_io();
        }
        if (!flush_and_sync(f)) {
            return fail_io();
        }
        const int close_result = std::fclose(temp.stream);
        temp.stream = nullptr;
        if (close_result != 0) {
            remove_temp_file(temp);
            return GGML_VEC_INDEX_E_IO;
        }
        if (!rename_overwrite(temp, path)) {
            return fail_io();
        }
        temp.path.clear();
        if (!fsync_parent_dir(path)) {
            return GGML_VEC_INDEX_E_IO;
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    try {
        if (path == nullptr) {
            return nullptr;
        }
        std::ifstream f;
#ifdef _WIN32
        std::wstring wide_path;
        if (!utf8_to_wide(path, wide_path)) {
            return nullptr;
        }
        f.open(std::filesystem::path(wide_path), std::ios::binary);
#else
        f.open(path, std::ios::binary);
#endif
        if (!f.is_open()) {
            return nullptr;
        }

        uint8_t header[kTvimHeaderSize] = {};
        f.read(reinterpret_cast<char *>(header), kTvimV1HeaderSize);
        if (!f || f.gcount() != static_cast<std::streamsize>(kTvimV1HeaderSize)) {
            return nullptr;
        }
        if (std::memcmp(header, kTvimMagic, 4) != 0) {
            return nullptr;
        }

        const uint8_t version = header[4];
        if (version != kTvimVersionV1 && version != kTvimVersion) {
            return nullptr;
        }
        if (version == kTvimVersion) {
            f.read(
                reinterpret_cast<char *>(header + kTvimV1HeaderSize),
                kTvimHeaderSize - kTvimV1HeaderSize);
            if (!f ||
                f.gcount() != static_cast<std::streamsize>(
                    kTvimHeaderSize - kTvimV1HeaderSize)) {
                return nullptr;
            }
        }

        const uint8_t flags = version == kTvimVersion ? header[7] : 0;
        if ((version == kTvimVersionV1 && (header[6] != 0 || header[7] != 0)) ||
            (version == kTvimVersion &&
             ((flags & ~kFlagCRC32C) != 0 || get_u32_le(header + 28) != 0))) {
            return nullptr;
        }

        const int serialized_bit_width = static_cast<int>(header[5]);
        if ((version == kTvimVersionV1 &&
             (serialized_bit_width <= 0 || serialized_bit_width > 32)) ||
            (version == kTvimVersion && !is_supported_bit_width(serialized_bit_width))) {
            return nullptr;
        }
        const int bit_width =
            version == kTvimVersionV1 && serialized_bit_width != 8 ?
                32 : serialized_bit_width;
        const uint8_t kind = header[6];
        const uint32_t dim_le = get_u32_le(header + 8);
        const uint32_t n_le   = get_u32_le(header + 12);
        const uint32_t qparam_type =
            version == kTvimVersion ? get_u32_le(header + 16) : kQParamNone;
        const uint32_t qparam_bytes =
            version == kTvimVersion ? get_u32_le(header + 20) : 0;
        const uint32_t comp_bytes =
            version == kTvimVersion ? get_u32_le(header + 24) : 4;
        if (dim_le == 0 || dim_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return nullptr;
        }
        if (version == kTvimVersion &&
            ((bit_width == 8 && (kind != kStorageQ8 || qparam_type != kQParamScaleF32 ||
                                  qparam_bytes != 4 || comp_bytes != 1)) ||
             (bit_width == 32 && (kind != kStorageF32 || qparam_type != kQParamNone ||
                                   qparam_bytes != 0 || comp_bytes != 4)))) {
            return nullptr;
        }

        uint64_t expected_size = 0;
        if (!expected_file_size(
                version == kTvimVersion ? kTvimHeaderSize : kTvimV1HeaderSize,
                n_le,
                dim_le,
                qparam_bytes,
                comp_bytes,
                expected_size)) {
            return nullptr;
        }
        if ((flags & kFlagCRC32C) != 0 &&
            !checked_add_u64(expected_size, kTvimChecksumSize, expected_size)) {
            return nullptr;
        }
        if (expected_size >
            static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            return nullptr;
        }
        f.seekg(0, std::ios::end);
        const std::streamoff actual_size = f.tellg();
        if (!f || actual_size != static_cast<std::streamoff>(expected_size)) {
            return nullptr;
        }
        f.seekg(
            static_cast<std::streamoff>(
                version == kTvimVersion ? kTvimHeaderSize : kTvimV1HeaderSize),
            std::ios::beg);
        if (!f) {
            return nullptr;
        }

        const int dim = static_cast<int>(dim_le);

        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(dim, bit_width),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return nullptr;
        }
        const size_t dim_sz = static_cast<size_t>(dim);
        const size_t n      = static_cast<size_t>(n_le);
        if (n != 0 && dim_sz > std::numeric_limits<size_t>::max() / n) {
            return nullptr;
        }

        test_maybe_throw_bad_alloc();
        if (is_q8(*idx)) {
            idx->q8_data.resize(n * dim_sz);
            idx->q8_scale.resize(n);
        } else {
            idx->data.resize(n * dim_sz);
        }
        idx->slot_to_id.resize(n);
        idx->id_to_slot.reserve(n);

        const bool checksummed = (flags & kFlagCRC32C) != 0;
        uint32_t header_crc =
            crc32c_update(0xffffffffu, header, kTvimHeaderSize);
        uint32_t qparams_crc = 0xffffffffu;
        uint32_t vectors_crc = 0xffffffffu;
        uint32_t ids_crc = 0xffffffffu;

        if (version == kTvimVersionV1 && is_q8(*idx)) {
            std::vector<float> row(dim_sz);
            for (size_t slot = 0; slot < n; ++slot) {
                for (float & v : row) {
                    uint32_t bits = 0;
                    if (!read_u32_le(f, bits)) {
                        return nullptr;
                    }
                    v = u32_to_float(bits);
                    if (!std::isfinite(v)) {
                        return nullptr;
                    }
                }
                quantize_q8_row(
                    row.data(),
                    idx->q8_data.data() + slot * dim_sz,
                    dim,
                    idx->q8_scale[slot]);
            }
        } else if (is_q8(*idx)) {
            for (float & scale : idx->q8_scale) {
                uint32_t bits = 0;
                const bool read_ok = checksummed ?
                    read_u32_le_crc(f, bits, qparams_crc) :
                    read_u32_le(f, bits);
                if (!read_ok) {
                    return nullptr;
                }
                scale = u32_to_float(bits);
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    return nullptr;
                }
            }

            if (!idx->q8_data.empty()) {
                if (idx->q8_data.size() >
                    static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                    return nullptr;
                }
                f.read(
                    reinterpret_cast<char *>(idx->q8_data.data()),
                    static_cast<std::streamsize>(idx->q8_data.size() * sizeof(int8_t)));
                if (!f) {
                    return nullptr;
                }
                if (checksummed) {
                    vectors_crc = crc32c_update(
                        vectors_crc, idx->q8_data.data(), idx->q8_data.size());
                }
            }
            for (size_t slot = 0; slot < n; ++slot) {
                const float scale = idx->q8_scale[slot];
                const int8_t * row = idx->q8_data.data() + slot * dim_sz;
                for (size_t i = 0; i < dim_sz; ++i) {
                    if (row[i] == std::numeric_limits<int8_t>::min() ||
                        !std::isfinite(static_cast<float>(row[i]) * scale)) {
                        return nullptr;
                    }
                }
            }
        } else {
            for (float & v : idx->data) {
                uint32_t bits = 0;
                const bool read_ok = checksummed ?
                    read_u32_le_crc(f, bits, vectors_crc) :
                    read_u32_le(f, bits);
                if (!read_ok) {
                    return nullptr;
                }
                v = u32_to_float(bits);
                if (!std::isfinite(v)) {
                    return nullptr;
                }
            }
        }

        for (uint64_t & id : idx->slot_to_id) {
            const bool read_ok = checksummed ?
                read_u64_le_crc(f, id, ids_crc) :
                read_u64_le(f, id);
            if (!read_ok) {
                return nullptr;
            }
        }

        if (checksummed) {
            uint32_t expected_header_crc = 0;
            uint32_t expected_qparams_crc = 0;
            uint32_t expected_vectors_crc = 0;
            uint32_t expected_ids_crc = 0;
            if (!read_u32_le(f, expected_header_crc) ||
                !read_u32_le(f, expected_qparams_crc) ||
                !read_u32_le(f, expected_vectors_crc) ||
                !read_u32_le(f, expected_ids_crc) ||
                (header_crc ^ 0xffffffffu) != expected_header_crc ||
                (qparams_crc ^ 0xffffffffu) != expected_qparams_crc ||
                (vectors_crc ^ 0xffffffffu) != expected_vectors_crc ||
                (ids_crc ^ 0xffffffffu) != expected_ids_crc) {
                return nullptr;
            }
        }

        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = idx->slot_to_id[slot];
            const bool inserted =
                idx->id_to_slot.emplace(id, slot).second;
            if (!inserted) {
                // Duplicate id in persisted file: corrupted.
                return nullptr;
            }
        }

        return idx.release();
    } catch (...) {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int ggml_vec_index_len(const ggml_vec_index_t * idx) {
    return idx ? static_cast<int>(idx->slot_to_id.size()) : 0;
}

int ggml_vec_index_dim(const ggml_vec_index_t * idx) {
    return idx ? idx->dim : 0;
}

int ggml_vec_index_bit_width(const ggml_vec_index_t * idx) {
    return idx ? idx->bit_width : 0;
}
