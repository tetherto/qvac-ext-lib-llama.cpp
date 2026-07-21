// ggml-vector-index.cpp - CPU implementation of the vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors or per-vector symmetric q8 codes. ID map uses
// std::unordered_map<uint64_t, size_t> for lookup and a parallel vector for
// the slot->id reverse map. Remove marks slots deleted; snapshots compact live rows.
//
// Search: dot product across all slots + min-heap of size k. q8/q4 search scores
// directly against stored codes and per-vector scales, with ARM NEON or x86
// AVX2 when available and a scalar fallback.

#include "ggml-vector-index.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GGML_VEC_INDEX_USE_NEON 1
#else
#define GGML_VEC_INDEX_USE_NEON 0
#endif

#if !GGML_VEC_INDEX_USE_NEON && (defined(__AVX2__) || ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))))
#include <immintrin.h>
#define GGML_VEC_INDEX_USE_AVX2 1
#if defined(__AVX2__)
#define GGML_VEC_INDEX_AVX2_ATTR
#else
#define GGML_VEC_INDEX_AVX2_ATTR __attribute__((target("avx2")))
#endif
#else
#define GGML_VEC_INDEX_USE_AVX2 0
#define GGML_VEC_INDEX_AVX2_ATTR
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
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
void ggml_vec_index_test_set_truncate_fail(int fail);
void ggml_vec_index_test_set_parent_fsync_fail(int fail);
void ggml_vec_index_test_set_delta_append_wait_target(int target);
void ggml_vec_index_test_reset_delta_tail_scan_count(void);
int64_t ggml_vec_index_test_get_delta_tail_scan_count(void);
}
#endif

namespace {

constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
constexpr uint8_t  kTvimVersionV1  = 1;
constexpr uint8_t  kTvimVersion    = 2;
constexpr uint8_t  kStorageF32     = 1;
constexpr uint8_t  kStorageQ8      = 2;
constexpr uint8_t  kStorageQ4      = 3;
constexpr uint8_t  kFlagCRC32C     = 1;
constexpr uint32_t kQParamNone     = 0;
constexpr uint32_t kQParamScaleF32 = 1;
constexpr size_t   kTvimV1HeaderSize = 16;
constexpr size_t   kTvimHeaderSize = 32;
constexpr size_t   kTvimChecksumSize = 16;
constexpr uint8_t  kTvidMagic[4]   = { 'T', 'V', 'D', 'L' };
constexpr uint8_t  kTvidVersion    = 1;
constexpr uint8_t  kTvidOpAdd      = 1;
constexpr uint8_t  kTvidOpRemove   = 2;
constexpr size_t   kTvidHeaderSize = 16;
constexpr size_t   kTvidRecordHeaderSize = 24;
constexpr size_t   kMaxIndexLen    = static_cast<size_t>(std::numeric_limits<int>::max());

static_assert(sizeof(float) == sizeof(uint32_t), "ggml-vector-index requires float32");

struct MappedFile {
    void * data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif

    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
};

void close_mapped_file(MappedFile & mapped) {
#ifdef _WIN32
    if (mapped.data != nullptr) {
        UnmapViewOfFile(mapped.data);
        mapped.data = nullptr;
    }
    if (mapped.mapping != nullptr) {
        CloseHandle(mapped.mapping);
        mapped.mapping = nullptr;
    }
    if (mapped.file != INVALID_HANDLE_VALUE) {
        CloseHandle(mapped.file);
        mapped.file = INVALID_HANDLE_VALUE;
    }
#else
    if (mapped.data != nullptr) {
        munmap(mapped.data, mapped.size);
        mapped.data = nullptr;
    }
    if (mapped.fd >= 0) {
        close(mapped.fd);
        mapped.fd = -1;
    }
#endif
    mapped.size = 0;
}

MappedFile::~MappedFile() {
    close_mapped_file(*this);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
std::atomic<int64_t> g_test_oom_countdown{ -1 };
std::atomic<int64_t> g_test_write_fail_after{ -1 };
std::atomic<bool> g_test_truncate_fail{ false };
std::atomic<bool> g_test_parent_fsync_fail{ false };
std::atomic<int> g_test_delta_append_wait_target{ 0 };
std::atomic<int> g_test_delta_append_waiters{ 0 };
std::atomic<int64_t> g_test_delta_tail_scan_count{ 0 };

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

void test_wait_after_delta_validate() {
    const int target = g_test_delta_append_wait_target.load();
    if (target <= 0) {
        return;
    }

    g_test_delta_append_waiters.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (g_test_delta_append_waiters.load() < target &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
#else
#define test_maybe_throw_bad_alloc() ((void) 0)
#define test_wait_after_delta_validate() ((void) 0)
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

bool host_is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
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
    if (component_bytes == 0) {
        uint64_t row_bytes = 0;
        if (!checked_add_u64(dim, 1, row_bytes)) {
            return false;
        }
        row_bytes /= 2;
        if (!checked_mul_u64(n, row_bytes, components)) {
            return false;
        }
    } else if (!checked_mul_u64(n, dim, components) ||
               !checked_mul_u64(components, component_bytes, components)) {
        return false;
    }
    if (!checked_mul_u64(n, qparam_bytes, qparams) ||
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

bool map_file_readonly(const char * path, MappedFile & mapped) {
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) {
        return false;
    }
    mapped.file = CreateFileW(
        wide.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (mapped.file == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(mapped.file, &size) || size.QuadPart <= 0 ||
        static_cast<uint64_t>(size.QuadPart) > std::numeric_limits<size_t>::max()) {
        close_mapped_file(mapped);
        return false;
    }
    mapped.mapping = CreateFileMappingW(mapped.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapped.mapping == nullptr) {
        close_mapped_file(mapped);
        return false;
    }
    mapped.data = MapViewOfFile(mapped.mapping, FILE_MAP_READ, 0, 0, 0);
    if (mapped.data == nullptr) {
        close_mapped_file(mapped);
        return false;
    }
    mapped.size = static_cast<size_t>(size.QuadPart);
    return true;
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

bool map_file_readonly(const char * path, MappedFile & mapped) {
    mapped.fd = ::open(path, O_RDONLY);
    if (mapped.fd < 0) {
        return false;
    }
    struct stat st;
    if (::fstat(mapped.fd, &st) != 0 || st.st_size <= 0 ||
        static_cast<uint64_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
        close_mapped_file(mapped);
        return false;
    }
    mapped.size = static_cast<size_t>(st.st_size);
    mapped.data = mmap(nullptr, mapped.size, PROT_READ, MAP_PRIVATE, mapped.fd, 0);
    if (mapped.data == MAP_FAILED) {
        mapped.data = nullptr;
        close_mapped_file(mapped);
        return false;
    }
    ::close(mapped.fd);
    mapped.fd = -1;
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
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    if (g_test_parent_fsync_fail.load()) {
        return false;
    }
#endif
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

void ggml_vec_index_test_set_truncate_fail(int fail) {
    g_test_truncate_fail.store(fail != 0);
}

void ggml_vec_index_test_set_parent_fsync_fail(int fail) {
    g_test_parent_fsync_fail.store(fail != 0);
}

void ggml_vec_index_test_set_delta_append_wait_target(int target) {
    g_test_delta_append_waiters.store(0);
    g_test_delta_append_wait_target.store(target);
}

void ggml_vec_index_test_reset_delta_tail_scan_count(void) {
    g_test_delta_tail_scan_count.store(0);
}

int64_t ggml_vec_index_test_get_delta_tail_scan_count(void) {
    return g_test_delta_tail_scan_count.load();
}
}
#endif

static std::atomic<uint64_t> g_next_filter_cookie{ 1 };

// Lifetime-managed instance state. Lives behind the opaque
// `ggml_vec_index_t` typedef.
struct ggml_vec_index {
    mutable std::shared_mutex mutex;

    int dim       = 0;
    int bit_width = 32;
    uint64_t generation = 0;
    uint64_t filter_cookie = 0;
    bool read_only_mmap = false;
    bool delta_log_rebase_pending = false;
    uint32_t delta_log_rebase_crc = 0;
    bool delta_tail_cache_valid = false;
    std::string delta_tail_cache_path;
    uint64_t delta_tail_cache_size = 0;
    uint32_t delta_tail_cache_crc = 0;

    std::unique_ptr<MappedFile> mapped_file;
    std::string mapped_source_path;
    size_t mapped_vector_bytes = 0;
    const float   * mapped_data = nullptr;
    const int8_t  * mapped_q8_data = nullptr;
    const uint8_t * mapped_q4_data = nullptr;

    // Flat row-major f32 storage for bit_width=32.
    std::vector<float> data;

    // Flat row-major q8 storage for bit_width=8 plus one scale per vector.
    std::vector<int8_t> q8_data;
    std::vector<float>  q8_scale;

    // Packed row-major q4 storage for bit_width=4 plus one scale per vector.
    std::vector<uint8_t> q4_data;
    std::vector<float>   q4_scale;

    // slot -> external id (parallel to logical slot index).
    std::vector<uint64_t> slot_to_id;
    std::vector<uint8_t>  slot_active;
    size_t n_active = 0;

    // external id -> slot.
    std::unordered_map<uint64_t, size_t> id_to_slot;

    // In-memory IVF-flat ANN structure. Rebuilt explicitly after mutations.
    uint64_t ivf_generation = std::numeric_limits<uint64_t>::max();
    int ivf_n_lists = 0;
    std::vector<float> ivf_centroids;
    std::vector<std::vector<size_t>> ivf_lists;
};

struct ggml_vec_index_filter {
    const ggml_vec_index_t * owner = nullptr;
    uint64_t owner_cookie = 0;
    int dim = 0;
    int bit_width = 32;
    uint64_t generation = 0;
    std::vector<size_t> slots;
};

static void invalidate_ivf(ggml_vec_index & idx) {
    idx.ivf_generation = std::numeric_limits<uint64_t>::max();
    idx.ivf_n_lists = 0;
    idx.ivf_centroids.clear();
    idx.ivf_lists.clear();
}

static bool is_q8(const ggml_vec_index & idx) {
    return idx.bit_width == 8;
}

static bool is_q4(const ggml_vec_index & idx) {
    return idx.bit_width == 4;
}

static bool is_quantized(const ggml_vec_index & idx) {
    return is_q4(idx) || is_q8(idx);
}

static uint8_t storage_kind(const ggml_vec_index & idx) {
    return is_q4(idx) ? kStorageQ4 : (is_q8(idx) ? kStorageQ8 : kStorageF32);
}

static size_t q4_row_bytes(size_t dim) {
    return (dim + 1) / 2;
}

static size_t vector_bytes(const ggml_vec_index & idx) {
    const size_t n = idx.slot_to_id.size();
    const size_t dim_sz = static_cast<size_t>(idx.dim);
    if (is_q4(idx)) {
        return n * q4_row_bytes(dim_sz);
    }
    if (is_q8(idx)) {
        return n * dim_sz * sizeof(int8_t);
    }
    return n * dim_sz * sizeof(float);
}

static bool slot_is_active(const ggml_vec_index & idx, size_t slot) {
    return slot < idx.slot_active.size() && idx.slot_active[slot] != 0;
}

static size_t active_count(const ggml_vec_index & idx) {
    return idx.n_active;
}

static const float * f32_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_data != nullptr ? idx.mapped_data : idx.data.data();
}

static const int8_t * q8_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_q8_data != nullptr ? idx.mapped_q8_data : idx.q8_data.data();
}

static const uint8_t * q4_data_ptr(const ggml_vec_index & idx) {
    return idx.mapped_q4_data != nullptr ? idx.mapped_q4_data : idx.q4_data.data();
}

static bool has_vector_storage(const ggml_vec_index & idx) {
    const size_t bytes = vector_bytes(idx);
    if (idx.read_only_mmap) {
        return idx.mapped_vector_bytes == bytes &&
            (bytes == 0 ||
             idx.mapped_data != nullptr ||
             idx.mapped_q8_data != nullptr ||
             idx.mapped_q4_data != nullptr);
    }
    if (is_q4(idx)) {
        return idx.q4_data.size() == bytes;
    }
    if (is_q8(idx)) {
        return idx.q8_data.size() == bytes;
    }
    return idx.data.size() == bytes / sizeof(float);
}

static uint8_t q4_encode(int q) {
    return static_cast<uint8_t>(q + 8);
}

static int q4_decode(uint8_t nibble) {
    return static_cast<int>(nibble) - 8;
}

static int round_nearest_even(float value) {
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
        int q = round_nearest_even(scaled);
        q = std::max(-127, std::min(127, q));
        dst[i] = static_cast<int8_t>(q);
    }
}

static void quantize_q4_row(const float * src, uint8_t * dst, int dim, float & scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    std::memset(dst, 0x88, q4_row_bytes(static_cast<size_t>(dim)));
    if (max_abs == 0.0f) {
        scale = 1.0f;
        return;
    }

    scale = max_abs / 7.0f;
    if (scale == 0.0f) {
        scale = max_abs;
    }
    for (int i = 0; i < dim; ++i) {
        const float scaled = src[i] / scale;
        int q = round_nearest_even(scaled);
        q = std::max(-7, std::min(7, q));
        const uint8_t code = q4_encode(q);
        uint8_t & byte = dst[static_cast<size_t>(i) / 2];
        if ((i & 1) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | code);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (code << 4));
        }
    }
}

namespace {

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

uint32_t index_state_crc32c(const ggml_vec_index & idx) {
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.bit_width));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(storage_kind(idx)));
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(active_count(idx)));
    if (is_q4(idx)) {
        const size_t row_bytes = q4_row_bytes(static_cast<size_t>(idx.dim));
        const uint8_t * data = q4_data_ptr(idx);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot)) {
                crc = crc32c_update_u32(crc, float_to_u32(idx.q4_scale[slot]));
            }
        }
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot)) {
                crc = crc32c_update(crc, data + slot * row_bytes, row_bytes);
            }
        }
    } else if (is_q8(idx)) {
        const size_t dim_sz = static_cast<size_t>(idx.dim);
        const int8_t * data = q8_data_ptr(idx);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot)) {
                crc = crc32c_update_u32(crc, float_to_u32(idx.q8_scale[slot]));
            }
        }
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot)) {
                crc = crc32c_update(crc, data + slot * dim_sz, dim_sz * sizeof(int8_t));
            }
        }
    } else {
        const float * data = f32_data_ptr(idx);
        const size_t dim_sz = static_cast<size_t>(idx.dim);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (!slot_is_active(idx, slot)) {
                continue;
            }
            for (size_t i = 0; i < dim_sz; ++i) {
                crc = crc32c_update_u32(crc, float_to_u32(data[slot * dim_sz + i]));
            }
        }
    }
    for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
        if (slot_is_active(idx, slot)) {
            crc = crc32c_update_u64(crc, idx.slot_to_id[slot]);
        }
    }
    return crc ^ 0xffffffffu;
}

uint32_t index_state_crc32c_after_remove(const ggml_vec_index & idx, uint64_t id) {
    const auto it = idx.id_to_slot.find(id);
    if (it == idx.id_to_slot.end()) {
        return index_state_crc32c(idx);
    }
    const size_t removed = it->second;
    const size_t dim_sz = static_cast<size_t>(idx.dim);

    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.bit_width));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(storage_kind(idx)));
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(active_count(idx) - 1));

    if (is_q4(idx)) {
        const size_t row_bytes = q4_row_bytes(dim_sz);
        const uint8_t * data = q4_data_ptr(idx);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot) && slot != removed) {
                crc = crc32c_update_u32(crc, float_to_u32(idx.q4_scale[slot]));
            }
        }
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot) && slot != removed) {
                crc = crc32c_update(crc, data + slot * row_bytes, row_bytes);
            }
        }
    } else if (is_q8(idx)) {
        const int8_t * data = q8_data_ptr(idx);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot) && slot != removed) {
                crc = crc32c_update_u32(crc, float_to_u32(idx.q8_scale[slot]));
            }
        }
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (slot_is_active(idx, slot) && slot != removed) {
                crc = crc32c_update(crc, data + slot * dim_sz, dim_sz * sizeof(int8_t));
            }
        }
    } else {
        const float * data = f32_data_ptr(idx);
        for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
            if (!slot_is_active(idx, slot) || slot == removed) {
                continue;
            }
            for (size_t i = 0; i < dim_sz; ++i) {
                crc = crc32c_update_u32(crc, float_to_u32(data[slot * dim_sz + i]));
            }
        }
    }
    for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
        if (slot_is_active(idx, slot) && slot != removed) {
            crc = crc32c_update_u64(crc, idx.slot_to_id[slot]);
        }
    }
    return crc ^ 0xffffffffu;
}

bool filesystem_path_from_utf8(const char * path, std::filesystem::path & out) {
#ifdef _WIN32
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) {
        return false;
    }
    out = std::filesystem::path(wide);
#else
    out = std::filesystem::path(path);
#endif
    return true;
}

bool filesystem_paths_equal(const char * lhs, const char * rhs) {
    if (std::strcmp(lhs, rhs) == 0) {
        return true;
    }

    std::filesystem::path lhs_path;
    std::filesystem::path rhs_path;
    if (!filesystem_path_from_utf8(lhs, lhs_path) ||
        !filesystem_path_from_utf8(rhs, rhs_path)) {
        return false;
    }

    std::error_code lhs_ec;
    std::error_code rhs_ec;
    std::filesystem::path lhs_resolved = std::filesystem::weakly_canonical(lhs_path, lhs_ec);
    std::filesystem::path rhs_resolved = std::filesystem::weakly_canonical(rhs_path, rhs_ec);
    if (lhs_ec || rhs_ec) {
        lhs_ec.clear();
        rhs_ec.clear();
        lhs_resolved = std::filesystem::absolute(lhs_path, lhs_ec);
        rhs_resolved = std::filesystem::absolute(rhs_path, rhs_ec);
        if (lhs_ec || rhs_ec) {
            return false;
        }
    }
    return lhs_resolved.lexically_normal() == rhs_resolved.lexically_normal();
}

void invalidate_delta_tail_cache(ggml_vec_index & idx) noexcept {
    idx.delta_tail_cache_valid = false;
    idx.delta_tail_cache_size = 0;
    idx.delta_tail_cache_crc = 0;
}

void set_delta_tail_cache(
        ggml_vec_index & idx,
        const char * path,
        uint64_t size,
        uint32_t tail_crc) noexcept {
    try {
        idx.delta_tail_cache_path = path;
        idx.delta_tail_cache_size = size;
        idx.delta_tail_cache_crc = tail_crc;
        idx.delta_tail_cache_valid = true;
    } catch (...) {
        invalidate_delta_tail_cache(idx);
    }
}

bool get_delta_tail_cache(
        ggml_vec_index & idx,
        const char * path,
        uint64_t file_size,
        uint32_t & tail_crc,
        uint64_t & complete_size) {
    if (!idx.delta_tail_cache_valid ||
        idx.delta_tail_cache_size != file_size ||
        idx.delta_tail_cache_path.empty()) {
        return false;
    }
    const bool same_path =
        idx.delta_tail_cache_path == path ||
        filesystem_paths_equal(idx.delta_tail_cache_path.c_str(), path);
    if (!same_path) {
        return false;
    }
    tail_crc = idx.delta_tail_cache_crc;
    complete_size = idx.delta_tail_cache_size;
    return true;
}

bool delta_lock_path(const char * path, std::filesystem::path & out) {
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }

    std::error_code ec;
    out = std::filesystem::weakly_canonical(fs_path, ec);
    if (ec) {
        ec.clear();
        out = std::filesystem::absolute(fs_path, ec);
        if (ec) {
            return false;
        }
    }
    out = out.lexically_normal();
    out += ".lock";
    return true;
}

std::shared_ptr<std::mutex> delta_log_process_mutex_for(const std::filesystem::path & lock_path) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<std::mutex>> registry;

    // POSIX advisory locks are process-owned, so same-process threads need an
    // in-memory companion lock. Key it by path to avoid serializing every log.
    const std::string key = lock_path.u8string();
    std::lock_guard<std::mutex> guard(registry_mutex);
    const auto found = registry.find(key);
    if (found != registry.end()) {
        std::shared_ptr<std::mutex> mutex = found->second.lock();
        if (mutex != nullptr) {
            return mutex;
        }
        registry.erase(found);
    }

    for (auto it = registry.begin(); it != registry.end();) {
        if (it->second.expired()) {
            it = registry.erase(it);
        } else {
            ++it;
        }
    }

    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    registry.emplace(key, mutex);
    return mutex;
}

class DeltaLogLock {
public:
    explicit DeltaLogLock(const char * path) {
        std::filesystem::path lock_path;
        if (!delta_lock_path(path, lock_path)) {
            return;
        }
        process_mutex = delta_log_process_mutex_for(lock_path);
        process_lock = std::unique_lock<std::mutex>(*process_mutex);
#ifdef _WIN32
        file = CreateFileW(
            lock_path.wstring().c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return;
        }
        OVERLAPPED overlapped = {};
        if (LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped) == 0) {
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
            return;
        }
#else
        fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            return;
        }
        struct flock lock = {};
        lock.l_type = F_WRLCK;
        lock.l_whence = SEEK_SET;
        while (::fcntl(fd, F_SETLKW, &lock) != 0) {
            if (errno == EINTR) {
                continue;
            }
            ::close(fd);
            fd = -1;
            return;
        }
#endif
        locked = true;
    }

    ~DeltaLogLock() {
        if (!locked) {
            return;
        }
#ifdef _WIN32
        OVERLAPPED overlapped = {};
        UnlockFileEx(file, 0, MAXDWORD, MAXDWORD, &overlapped);
        CloseHandle(file);
        file = INVALID_HANDLE_VALUE;
#else
        struct flock lock = {};
        lock.l_type = F_UNLCK;
        lock.l_whence = SEEK_SET;
        (void) ::fcntl(fd, F_SETLK, &lock);
        ::close(fd);
        fd = -1;
#endif
    }

    bool ok() const {
        return locked;
    }

private:
    std::shared_ptr<std::mutex> process_mutex;
    std::unique_lock<std::mutex> process_lock;
    bool locked = false;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

bool open_append_file(const char * path, std::FILE ** out) {
#ifdef _WIN32
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) {
        return false;
    }
    *out = _wfopen(wide.c_str(), L"a+b");
#else
    *out = std::fopen(path, "a+b");
#endif
    return *out != nullptr;
}

void fill_delta_header(const ggml_vec_index & idx, uint32_t base_crc, uint8_t * header) {
    std::memset(header, 0, kTvidHeaderSize);
    std::memcpy(header, kTvidMagic, 4);
    header[4] = kTvidVersion;
    header[5] = static_cast<uint8_t>(idx.bit_width);
    put_u32_le(header + 8, static_cast<uint32_t>(idx.dim));
    put_u32_le(header + 12, base_crc);
}

bool validate_delta_header(
        const char * path,
        const ggml_vec_index & idx,
        uint64_t & size,
        uint32_t & base_crc) {
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }
    if (!std::filesystem::exists(fs_path)) {
        size = 0;
        base_crc = index_state_crc32c(idx);
        return true;
    }
    size = static_cast<uint64_t>(std::filesystem::file_size(fs_path));
    if (size == 0) {
        base_crc = index_state_crc32c(idx);
        return true;
    }
    if (size < kTvidHeaderSize) {
        return false;
    }

    std::ifstream f(fs_path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    uint8_t header[kTvidHeaderSize] = {};
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!f) {
        return false;
    }
    if (std::memcmp(header, kTvidMagic, 4) != 0 ||
        header[4] != kTvidVersion ||
        header[5] != static_cast<uint8_t>(idx.bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx.dim)) {
        return false;
    }
    base_crc = get_u32_le(header + 12);
    return true;
}

bool truncate_file_to(const char * path, uint64_t size) {
    try {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        if (g_test_truncate_fail.load()) {
            return false;
        }
#endif
        std::filesystem::path fs_path;
        if (!filesystem_path_from_utf8(path, fs_path)) {
            return false;
        }
        std::filesystem::resize_file(fs_path, size);
        return true;
    } catch (...) {
        return false;
    }
}

bool inspect_delta_log_tail(
        const char * path,
        const ggml_vec_index & idx,
        uint32_t & last_state_crc,
        uint64_t & complete_size) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    g_test_delta_tail_scan_count.fetch_add(1);
#endif
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }
    const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(fs_path));
    if (file_size < kTvidHeaderSize) {
        return false;
    }

    std::ifstream f(fs_path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    uint8_t header[kTvidHeaderSize] = {};
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!f ||
        std::memcmp(header, kTvidMagic, 4) != 0 ||
        header[4] != kTvidVersion ||
        header[5] != static_cast<uint8_t>(idx.bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx.dim)) {
        return false;
    }

    last_state_crc = get_u32_le(header + 12);
    complete_size = kTvidHeaderSize;
    uint64_t offset = kTvidHeaderSize;
    while (offset < file_size) {
        uint8_t record[kTvidRecordHeaderSize] = {};
        f.read(reinterpret_cast<char *>(record), sizeof(record));
        if (f.gcount() != static_cast<std::streamsize>(sizeof(record))) {
            return true;
        }
        offset += kTvidRecordHeaderSize;

        const uint8_t op = record[0];
        const uint64_t payload_bytes = get_u64_le(record + 8);
        const uint32_t expected_crc = get_u32_le(record + 16);
        const uint32_t state_crc = get_u32_le(record + 20);
        if (record[1] != 0 || record[2] != 0 || record[3] != 0 ||
            (op != kTvidOpAdd && op != kTvidOpRemove)) {
            return false;
        }
        if (payload_bytes > file_size - offset) {
            return true;
        }

        if (payload_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            payload_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            return false;
        }
        std::vector<uint8_t> payload(static_cast<size_t>(payload_bytes));
        if (!payload.empty()) {
            f.read(
                reinterpret_cast<char *>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
        }
        if (!f) {
            return true;
        }
        offset += payload_bytes;

        uint32_t crc = crc32c_update(0xffffffffu, record, 16);
        crc = crc32c_update(crc, record + 20, 4);
        if (!payload.empty()) {
            crc = crc32c_update(crc, payload.data(), payload.size());
        }
        if ((crc ^ 0xffffffffu) != expected_crc) {
            return false;
        }
        last_state_crc = state_crc;
        complete_size = offset;
    }
    return true;
}

bool delta_log_ends_at_state(
        const char * path,
        const ggml_vec_index & idx,
        uint32_t state_crc) {
    try {
        std::filesystem::path fs_path;
        if (!filesystem_path_from_utf8(path, fs_path)) {
            return false;
        }
        uint32_t tail_crc = 0;
        uint64_t complete_size = 0;
        return inspect_delta_log_tail(path, idx, tail_crc, complete_size) &&
            complete_size == static_cast<uint64_t>(std::filesystem::file_size(fs_path)) &&
            tail_crc == state_crc;
    } catch (...) {
        return false;
    }
}

struct DeltaAppendResult {
    int status = GGML_VEC_INDEX_OK;
    bool record_complete = false;
};

DeltaAppendResult append_delta_record(
        ggml_vec_index & idx,
        const char * delta_path,
        uint8_t op,
        uint32_t n,
        uint32_t base_crc_for_new_log,
        uint32_t state_crc,
        const std::vector<uint8_t> & payload) {
    if (delta_path == nullptr) {
        return { GGML_VEC_INDEX_E_INVALID_ARG, false };
    }
    DeltaLogLock delta_lock(delta_path);
    if (!delta_lock.ok()) {
        return { GGML_VEC_INDEX_E_IO, false };
    }

    uint64_t old_size = 0;
    uint32_t existing_base_crc = 0;
    if (!validate_delta_header(delta_path, idx, old_size, existing_base_crc)) {
        return { GGML_VEC_INDEX_E_IO, false };
    }
    if (old_size != 0) {
        uint32_t tail_crc = 0;
        uint64_t complete_size = 0;
        if (!get_delta_tail_cache(idx, delta_path, old_size, tail_crc, complete_size) &&
            !inspect_delta_log_tail(delta_path, idx, tail_crc, complete_size)) {
            return { GGML_VEC_INDEX_E_IO, false };
        }
        if (tail_crc != base_crc_for_new_log) {
            invalidate_delta_tail_cache(idx);
            return { GGML_VEC_INDEX_E_IO, false };
        }
        if (complete_size != old_size) {
            if (!truncate_file_to(delta_path, complete_size)) {
                invalidate_delta_tail_cache(idx);
                return { GGML_VEC_INDEX_E_INTERNAL, false };
            }
            old_size = complete_size;
        }
        if (idx.delta_log_rebase_pending &&
            idx.delta_log_rebase_crc == base_crc_for_new_log &&
            existing_base_crc != base_crc_for_new_log) {
            // The snapshot already includes this log's records (e.g. crash after
            // compacting the snapshot but before replacing the old delta log).
            if (!truncate_file_to(delta_path, 0)) {
                invalidate_delta_tail_cache(idx);
                return { GGML_VEC_INDEX_E_INTERNAL, false };
            }
            old_size = 0;
            idx.delta_log_rebase_pending = false;
            idx.delta_log_rebase_crc = 0;
            invalidate_delta_tail_cache(idx);
        }
    }
    test_wait_after_delta_validate();

    std::FILE * f = nullptr;
    if (!open_append_file(delta_path, &f)) {
        return { GGML_VEC_INDEX_E_IO, false };
    }
    auto close_file = [&]() {
        if (f != nullptr) {
            std::fclose(f);
            f = nullptr;
        }
    };
    auto fail_io = [&]() -> DeltaAppendResult {
        close_file();
        const bool truncated = truncate_file_to(delta_path, old_size);
        const bool record_complete =
            !truncated && delta_log_ends_at_state(delta_path, idx, state_crc);
        invalidate_delta_tail_cache(idx);
        return {
            truncated ? GGML_VEC_INDEX_E_IO : GGML_VEC_INDEX_E_INTERNAL,
            record_complete,
        };
    };

    if (old_size == 0) {
        uint8_t header[kTvidHeaderSize] = {};
        fill_delta_header(idx, base_crc_for_new_log, header);
        if (!write_bytes(f, header, sizeof(header))) {
            return fail_io();
        }
    }

    uint8_t record[kTvidRecordHeaderSize] = {};
    record[0] = op;
    put_u32_le(record + 4, n);
    put_u64_le(record + 8, static_cast<uint64_t>(payload.size()));
    put_u32_le(record + 20, state_crc);
    uint32_t crc = crc32c_update(0xffffffffu, record, 16);
    crc = crc32c_update(crc, record + 20, 4);
    if (!payload.empty()) {
        crc = crc32c_update(crc, payload.data(), payload.size());
    }
    put_u32_le(record + 16, crc ^ 0xffffffffu);

    if (!write_bytes(f, record, sizeof(record)) ||
        (!payload.empty() && !write_bytes(f, payload.data(), payload.size())) ||
        !flush_and_sync(f)) {
        return fail_io();
    }
    const int close_result = std::fclose(f);
    f = nullptr;
    if (close_result != 0 || !fsync_parent_dir(delta_path)) {
        const bool truncated = truncate_file_to(delta_path, old_size);
        const bool record_complete =
            !truncated && delta_log_ends_at_state(delta_path, idx, state_crc);
        invalidate_delta_tail_cache(idx);
        return {
            truncated ? GGML_VEC_INDEX_E_IO : GGML_VEC_INDEX_E_INTERNAL,
            record_complete,
        };
    }
    const uint64_t written_size =
        old_size +
        (old_size == 0 ? kTvidHeaderSize : 0) +
        kTvidRecordHeaderSize +
        static_cast<uint64_t>(payload.size());
    set_delta_tail_cache(idx, delta_path, written_size, state_crc);
    return { GGML_VEC_INDEX_OK, true };
}

int write_empty_delta_log_unlocked(const ggml_vec_index & idx, const char * delta_path) {
    if (delta_path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        TempFile temp;
        if (!open_temp_file(delta_path, temp)) {
            return GGML_VEC_INDEX_E_IO;
        }
        auto fail_io = [&]() {
            if (temp.stream != nullptr) {
                std::fclose(temp.stream);
                temp.stream = nullptr;
            }
            remove_temp_file(temp);
            return GGML_VEC_INDEX_E_IO;
        };

        uint8_t header[kTvidHeaderSize] = {};
        fill_delta_header(idx, index_state_crc32c(idx), header);
        if (!write_bytes(temp.stream, header, sizeof(header)) ||
            !flush_and_sync(temp.stream)) {
            return fail_io();
        }
        const int close_result = std::fclose(temp.stream);
        temp.stream = nullptr;
        if (close_result != 0) {
            remove_temp_file(temp);
            return GGML_VEC_INDEX_E_IO;
        }
        if (!rename_overwrite(temp, delta_path)) {
            return fail_io();
        }
        temp.path.clear();
        if (!fsync_parent_dir(delta_path)) {
            return GGML_VEC_INDEX_E_IO;
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int write_empty_delta_log(const ggml_vec_index & idx, const char * delta_path) {
    if (delta_path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    DeltaLogLock delta_lock(delta_path);
    if (!delta_lock.ok()) {
        return GGML_VEC_INDEX_E_IO;
    }
    return write_empty_delta_log_unlocked(idx, delta_path);
}

bool validate_logged_add_args(
        const ggml_vec_index_t * idx,
        const float * vectors,
        int n,
        const uint64_t * ids) {
    if (idx == nullptr || vectors == nullptr || ids == nullptr || n < 0) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    const size_t n_sz = static_cast<size_t>(n);
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    if (dim_sz != 0 && n_sz > std::numeric_limits<size_t>::max() / dim_sz) {
        return false;
    }
    return all_finite(vectors, n_sz * dim_sz);
}

int check_logged_add_duplicates(
        const ggml_vec_index_t * idx,
        int n,
        const uint64_t * ids) {
    std::unordered_set<uint64_t> batch_ids;
    batch_ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!is_valid_id(ids[i])) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
            return GGML_VEC_INDEX_E_DUPLICATE;
        }
        if (!batch_ids.insert(ids[i]).second) {
            return GGML_VEC_INDEX_E_DUPLICATE;
        }
    }
    return GGML_VEC_INDEX_OK;
}

bool build_add_delta_payload(
        const ggml_vec_index_t * idx,
        const float * vectors,
        int n,
        const uint64_t * ids,
        std::vector<uint8_t> & payload) {
    const size_t n_sz = static_cast<size_t>(n);
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    const size_t id_bytes = n_sz * sizeof(uint64_t);
    const size_t vector_count = n_sz * dim_sz;
    if (vector_count > (std::numeric_limits<size_t>::max() - id_bytes) / sizeof(uint32_t)) {
        return false;
    }
    payload.clear();
    payload.reserve(id_bytes + vector_count * sizeof(uint32_t));
    for (int i = 0; i < n; ++i) {
        uint8_t bytes[8];
        put_u64_le(bytes, ids[i]);
        payload.insert(payload.end(), bytes, bytes + sizeof(bytes));
    }
    for (size_t i = 0; i < vector_count; ++i) {
        uint8_t bytes[4];
        put_u32_le(bytes, float_to_u32(vectors[i]));
        payload.insert(payload.end(), bytes, bytes + sizeof(bytes));
    }
    return true;
}

std::vector<uint8_t> build_remove_delta_payload(uint64_t id) {
    std::vector<uint8_t> payload(sizeof(uint64_t));
    put_u64_le(payload.data(), id);
    return payload;
}

} // namespace

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
        idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        if (idx->filter_cookie == 0) {
            idx->filter_cookie = g_next_filter_cookie.fetch_add(1, std::memory_order_relaxed);
        }
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

static void rollback_appended_slots_unlocked(
        ggml_vec_index_t * idx,
        size_t base_slot,
        const uint64_t * ids,
        int n) noexcept {
    if (idx == nullptr) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        idx->id_to_slot.erase(ids[i]);
    }
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    if (is_q4(*idx)) {
        idx->q4_data.resize(base_slot * q4_row_bytes(dim_sz));
        idx->q4_scale.resize(base_slot);
    } else if (is_q8(*idx)) {
        idx->q8_data.resize(base_slot * dim_sz);
        idx->q8_scale.resize(base_slot);
    } else {
        idx->data.resize(base_slot * dim_sz);
    }
    idx->slot_to_id.resize(base_slot);
    idx->slot_active.resize(base_slot);
    idx->n_active = idx->id_to_slot.size();
}

static int ggml_vec_index_add_unlocked(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    bool               finalize) {

    size_t base_slot = 0;
    size_t dim_sz    = 0;
    bool resized     = false;

    auto rollback = [&]() noexcept {
        if (idx == nullptr || !resized) {
            return;
        }
        rollback_appended_slots_unlocked(idx, base_slot, ids, n);
    };

    try {
        if (idx == nullptr || vectors == nullptr || ids == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
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
            if (!is_valid_id(ids[i])) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
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
        if (n_sz > kMaxIndexLen || active_count(*idx) > kMaxIndexLen - n_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
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
        if (is_q4(*idx)) {
            idx->q4_data.resize(new_slots * q4_row_bytes(dim_sz));
            idx->q4_scale.resize(new_slots);
        } else if (is_q8(*idx)) {
            idx->q8_data.resize(new_slots * dim_sz);
            idx->q8_scale.resize(new_slots);
        } else {
            idx->data.resize(new_slots * dim_sz);
        }
        idx->slot_to_id.resize(new_slots);
        idx->slot_active.resize(new_slots, 0);
        test_maybe_throw_bad_alloc();
        idx->id_to_slot.reserve(new_slots);

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            const float * src = vectors + static_cast<size_t>(i) * dim_sz;
            if (is_q4(*idx)) {
                quantize_q4_row(
                    src,
                    idx->q4_data.data() + slot * q4_row_bytes(dim_sz),
                    idx->dim,
                    idx->q4_scale[slot]);
            } else if (is_q8(*idx)) {
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
            idx->slot_active[slot] = 1;
            test_maybe_throw_bad_alloc();
            idx->id_to_slot.emplace(ids[i], slot);
        }
        idx->n_active += n_sz;
        if (finalize) {
            ++idx->generation;
            invalidate_ivf(*idx);
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

int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_add_unlocked(idx, vectors, n, ids, true);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static int ggml_vec_index_remove_unlocked(ggml_vec_index_t * idx, uint64_t id) {
    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!is_valid_id(id)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        auto it = idx->id_to_slot.find(id);
        if (it == idx->id_to_slot.end()) {
            return 0;
        }
        const size_t slot = it->second;
        if (!slot_is_active(*idx, slot)) {
            idx->id_to_slot.erase(it);
            return 0;
        }
        idx->slot_active[slot] = 0;
        --idx->n_active;
        idx->id_to_slot.erase(it);
        ++idx->generation;
        invalidate_ivf(*idx);
        return 1;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_remove_unlocked(idx, id);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static int ggml_vec_index_compact_unlocked(ggml_vec_index_t * idx) {
    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const size_t n_slots = idx->slot_to_id.size();
        const size_t n_live = active_count(*idx);
        if (n_live == n_slots) {
            return GGML_VEC_INDEX_OK;
        }

        const size_t dim_sz = static_cast<size_t>(idx->dim);
        if (dim_sz != 0 && n_live > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }

        test_maybe_throw_bad_alloc();
        std::vector<uint64_t> new_slot_to_id;
        std::vector<uint8_t> new_slot_active;
        std::unordered_map<uint64_t, size_t> new_id_to_slot;
        new_slot_to_id.reserve(n_live);
        new_slot_active.assign(n_live, 1);
        new_id_to_slot.reserve(n_live);

        if (is_q4(*idx)) {
            const size_t row_bytes = q4_row_bytes(dim_sz);
            std::vector<uint8_t> new_q4_data;
            std::vector<float> new_q4_scale;
            new_q4_data.resize(n_live * row_bytes);
            new_q4_scale.reserve(n_live);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_q4_data.data() + out_slot * row_bytes,
                    idx->q4_data.data() + slot * row_bytes,
                    row_bytes * sizeof(uint8_t));
                new_q4_scale.push_back(idx->q4_scale[slot]);
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->q4_data.swap(new_q4_data);
            idx->q4_scale.swap(new_q4_scale);
        } else if (is_q8(*idx)) {
            std::vector<int8_t> new_q8_data;
            std::vector<float> new_q8_scale;
            new_q8_data.resize(n_live * dim_sz);
            new_q8_scale.reserve(n_live);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_q8_data.data() + out_slot * dim_sz,
                    idx->q8_data.data() + slot * dim_sz,
                    dim_sz * sizeof(int8_t));
                new_q8_scale.push_back(idx->q8_scale[slot]);
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->q8_data.swap(new_q8_data);
            idx->q8_scale.swap(new_q8_scale);
        } else {
            std::vector<float> new_data;
            new_data.resize(n_live * dim_sz);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const size_t out_slot = new_slot_to_id.size();
                std::memcpy(
                    new_data.data() + out_slot * dim_sz,
                    idx->data.data() + slot * dim_sz,
                    dim_sz * sizeof(float));
                new_slot_to_id.push_back(idx->slot_to_id[slot]);
                new_id_to_slot.emplace(idx->slot_to_id[slot], out_slot);
            }
            idx->data.swap(new_data);
        }

        idx->slot_to_id.swap(new_slot_to_id);
        idx->slot_active.swap(new_slot_active);
        idx->id_to_slot.swap(new_id_to_slot);
        idx->n_active = idx->slot_to_id.size();
        ++idx->generation;
        invalidate_ivf(*idx);
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_compact(ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_compact_unlocked(idx);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path) {
    bool added = false;
    size_t base_slot = 0;
    std::unique_lock<std::shared_mutex> lock;
    try {
        if (idx == nullptr || delta_path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        lock = std::unique_lock<std::shared_mutex>(idx->mutex);
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!validate_logged_add_args(idx, vectors, n, ids)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }

        const int duplicate_status = check_logged_add_duplicates(idx, n, ids);
        if (duplicate_status != GGML_VEC_INDEX_OK) {
            return duplicate_status;
        }

        std::vector<uint8_t> payload;
        if (!build_add_delta_payload(idx, vectors, n, ids, payload)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const uint32_t base_crc = index_state_crc32c(*idx);
        base_slot = idx->slot_to_id.size();
        const int add_status = ggml_vec_index_add_unlocked(idx, vectors, n, ids, false);
        if (add_status != GGML_VEC_INDEX_OK) {
            return add_status;
        }
        added = true;

        const uint32_t added_state_crc = index_state_crc32c(*idx);
        const DeltaAppendResult append_result = append_delta_record(
            *idx,
            delta_path,
            kTvidOpAdd,
            static_cast<uint32_t>(n),
            base_crc,
            added_state_crc,
            payload);
        if (append_result.status != GGML_VEC_INDEX_OK) {
            if (append_result.record_complete) {
                ++idx->generation;
                invalidate_ivf(*idx);
                added = false;
                return GGML_VEC_INDEX_OK;
            } else {
                rollback_appended_slots_unlocked(idx, base_slot, ids, n);
            }
            added = false;
            return append_result.status;
        }
        ++idx->generation;
        invalidate_ivf(*idx);
        added = false;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        if (added) {
            rollback_appended_slots_unlocked(idx, base_slot, ids, n);
        }
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        if (added) {
            rollback_appended_slots_unlocked(idx, base_slot, ids, n);
        }
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path) {
    try {
        if (idx == nullptr || delta_path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        if (!is_valid_id(id)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->id_to_slot.count(id) == 0) {
            return 0;
        }
        const std::vector<uint8_t> payload = build_remove_delta_payload(id);
        const uint32_t base_crc = index_state_crc32c(*idx);
        const uint32_t post_remove_crc = index_state_crc32c_after_remove(*idx, id);
        const DeltaAppendResult append_result = append_delta_record(
            *idx,
            delta_path,
            kTvidOpRemove,
            1,
            base_crc,
            post_remove_crc,
            payload);
        if (append_result.status != GGML_VEC_INDEX_OK) {
            if (append_result.record_complete) {
                return ggml_vec_index_remove_unlocked(idx, id);
            }
            return append_result.status;
        }
        return ggml_vec_index_remove_unlocked(idx, id);
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        return idx->id_to_slot.count(id) != 0 ? 1 : 0;
    } catch (...) {
        return 0;
    }
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

inline float dot_q4_scalar(const float * query, const uint8_t * codes, float scale, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const float value = static_cast<float>(q4_decode(nibble)) * scale;
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

inline void dot_q4_neon_accum8(
        const float * query,
        uint8x8_t codes,
        float scale,
        float32x4_t & acc0,
        float32x4_t & acc1) {
    const int16x8_t q16 = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(codes)),
        vdupq_n_s16(8));
    const float32x4_t q0 =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))), scale);
    const float32x4_t q1 =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16))), scale);
    acc0 = vmlaq_f32(acc0, vld1q_f32(query), q0);
    acc1 = vmlaq_f32(acc1, vld1q_f32(query + 4), q1);
}

inline float dot_q4_neon(const float * query, const uint8_t * codes, float scale, int dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        const uint8x8_t packed = vld1_u8(codes + static_cast<size_t>(i) / 2);
        const uint8x8_t low = vand_u8(packed, vdup_n_u8(0x0f));
        const uint8x8_t high = vshr_n_u8(packed, 4);
        const uint8x8x2_t zipped = vzip_u8(low, high);

        dot_q4_neon_accum8(query + i, zipped.val[0], scale, acc0, acc1);
        dot_q4_neon_accum8(query + i + 8, zipped.val[1], scale, acc2, acc3);
    }

    float acc =
        horizontal_sum(acc0) + horizontal_sum(acc1) +
        horizontal_sum(acc2) + horizontal_sum(acc3);
    for (; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const float value = static_cast<float>(q4_decode(nibble)) * scale;
        acc += query[i] * value;
    }
    return acc;
}

#endif

#if GGML_VEC_INDEX_USE_AVX2

GGML_VEC_INDEX_AVX2_ATTR inline float horizontal_sum_avx2(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

GGML_VEC_INDEX_AVX2_ATTR inline float dot_q8_avx2(
        const float * query,
        const int8_t * codes,
        float scale,
        int dim) {
    const __m256 scale_v = _mm256_set1_ps(scale);
    __m256 acc_v = _mm256_setzero_ps();

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + i));
        const __m256 q = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)),
            scale_v);
        acc_v = _mm256_add_ps(acc_v, _mm256_mul_ps(_mm256_loadu_ps(query + i), q));
    }

    float acc = horizontal_sum_avx2(acc_v);
    for (; i < dim; ++i) {
        const float value = static_cast<float>(codes[i]) * scale;
        acc += query[i] * value;
    }
    return acc;
}

GGML_VEC_INDEX_AVX2_ATTR inline float dot_q4_avx2(
        const float * query,
        const uint8_t * codes,
        float scale,
        int dim) {
    const __m128i low_mask = _mm_set1_epi8(0x0f);
    const __m128i zero_point = _mm_set1_epi8(8);
    const __m256 scale_v = _mm256_set1_ps(scale);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + static_cast<size_t>(i) / 2));
        const __m128i low = _mm_and_si128(packed, low_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        const __m128i nibbles = _mm_unpacklo_epi8(low, high);
        const __m128i qbytes = _mm_sub_epi8(nibbles, zero_point);

        const __m256 q0 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qbytes)),
            scale_v);
        const __m256 q1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qbytes, 8))),
            scale_v);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(query + i), q0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(query + i + 8), q1));
    }

    float acc = horizontal_sum_avx2(acc0) + horizontal_sum_avx2(acc1);
    for (; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const float value = static_cast<float>(q4_decode(nibble)) * scale;
        acc += query[i] * value;
    }
    return acc;
}

bool cpu_has_avx2() {
#if defined(__AVX2__)
    return true;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

#endif

inline float dot_q8(const float * query, const int8_t * codes, float scale, int dim) {
#if GGML_VEC_INDEX_USE_NEON
    return dot_q8_neon(query, codes, scale, dim);
#elif GGML_VEC_INDEX_USE_AVX2
    static const bool has_avx2 = cpu_has_avx2();
    if (has_avx2) {
        return dot_q8_avx2(query, codes, scale, dim);
    }
    return dot_q8_scalar(query, codes, scale, dim);
#else
    return dot_q8_scalar(query, codes, scale, dim);
#endif
}

inline float dot_q4(const float * query, const uint8_t * codes, float scale, int dim) {
#if GGML_VEC_INDEX_USE_NEON
    return dot_q4_neon(query, codes, scale, dim);
#elif GGML_VEC_INDEX_USE_AVX2
    static const bool has_avx2 = cpu_has_avx2();
    if (has_avx2) {
        return dot_q4_avx2(query, codes, scale, dim);
    }
    return dot_q4_scalar(query, codes, scale, dim);
#else
    return dot_q4_scalar(query, codes, scale, dim);
#endif
}

inline float score_slot(const ggml_vec_index_t & idx, const float * query, size_t slot) {
    const int dim = idx.dim;
    return is_q4(idx) ?
        dot_q4(
            query,
            q4_data_ptr(idx) + slot * q4_row_bytes(static_cast<size_t>(dim)),
            idx.q4_scale[slot],
            dim) :
        is_q8(idx) ?
        dot_q8(
            query,
            q8_data_ptr(idx) + slot * static_cast<size_t>(dim),
            idx.q8_scale[slot],
            dim) :
        dot(
            query,
            f32_data_ptr(idx) + slot * static_cast<size_t>(dim),
            dim);
}

void decode_slot_to_f32(const ggml_vec_index_t & idx, size_t slot, float * dst) {
    const int dim = idx.dim;
    if (is_q4(idx)) {
        const uint8_t * codes =
            q4_data_ptr(idx) + slot * q4_row_bytes(static_cast<size_t>(dim));
        const float scale = idx.q4_scale[slot];
        for (int i = 0; i < dim; ++i) {
            const uint8_t byte = codes[static_cast<size_t>(i) / 2];
            const uint8_t nibble = (i & 1) == 0 ?
                static_cast<uint8_t>(byte & 0x0f) :
                static_cast<uint8_t>(byte >> 4);
            dst[i] = static_cast<float>(q4_decode(nibble)) * scale;
        }
    } else if (is_q8(idx)) {
        const int8_t * codes = q8_data_ptr(idx) + slot * static_cast<size_t>(dim);
        const float scale = idx.q8_scale[slot];
        for (int i = 0; i < dim; ++i) {
            dst[i] = static_cast<float>(codes[i]) * scale;
        }
    } else {
        std::memcpy(
            dst,
            f32_data_ptr(idx) + slot * static_cast<size_t>(dim),
            static_cast<size_t>(dim) * sizeof(float));
    }
}

size_t best_centroid(const float * query, const std::vector<float> & centroids, int n_lists, int dim) {
    size_t best = 0;
    float best_score = -FLT_MAX;
    for (int list = 0; list < n_lists; ++list) {
        const float s = dot(query, centroids.data() + static_cast<size_t>(list) * dim, dim);
        if (s > best_score) {
            best_score = s;
            best = static_cast<size_t>(list);
        }
    }
    return best;
}

// Run a single query against all slots, write top-k into out_scores/out_ids.
// If the index holds fewer than k entries, pad with sentinels.
void search_one(
    const ggml_vec_index_t & idx,
    const float            * query,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids,
    const std::vector<size_t> * allowed_slots = nullptr) {

    const size_t n_slots = idx.slot_to_id.size();

    test_maybe_throw_bad_alloc();
    std::priority_queue<ScoreId, std::vector<ScoreId>, MinHeapCmp> heap;

    auto visit_slot = [&](size_t slot) {
        if (!slot_is_active(idx, slot)) {
            return;
        }
        const float s = score_slot(idx, query, slot);
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push({ s, idx.slot_to_id[slot] });
        } else if (s > heap.top().score) {
            heap.pop();
            heap.push({ s, idx.slot_to_id[slot] });
        }
    };

    if (allowed_slots != nullptr) {
        for (size_t slot : *allowed_slots) {
            if (slot < n_slots) {
                visit_slot(slot);
            }
        }
    } else {
        for (size_t slot = 0; slot < n_slots; ++slot) {
            visit_slot(slot);
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

std::vector<size_t> allowed_slots_for_ids(
    const ggml_vec_index_t & idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed) {
    std::vector<size_t> slots;
    slots.reserve(static_cast<size_t>(n_allowed));
    for (int i = 0; i < n_allowed; ++i) {
        const auto it = idx.id_to_slot.find(allowed_ids[i]);
        if (it != idx.id_to_slot.end() && slot_is_active(idx, it->second)) {
            slots.push_back(it->second);
        }
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    return slots;
}

} // namespace

static int ggml_vec_index_build_ivf_unlocked(ggml_vec_index_t * idx, int n_lists, int n_iter) {
    try {
        if (idx == nullptr || n_lists <= 0 || n_iter < 0) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const size_t n_slots = idx->slot_to_id.size();
        const size_t n_live = active_count(*idx);
        const int dim = idx->dim;
        if (n_live == 0) {
            invalidate_ivf(*idx);
            idx->ivf_generation = idx->generation;
            return GGML_VEC_INDEX_OK;
        }

        const int actual_lists = static_cast<int>(
            std::min(static_cast<size_t>(n_lists), n_live));
        const size_t dim_sz = static_cast<size_t>(dim);
        test_maybe_throw_bad_alloc();

        std::vector<float> centroids(static_cast<size_t>(actual_lists) * dim_sz);
        std::vector<float> next_centroids(centroids.size());
        std::vector<int> counts(static_cast<size_t>(actual_lists));
        std::vector<float> row(dim_sz);
        std::vector<std::vector<size_t>> lists(static_cast<size_t>(actual_lists));
        std::vector<size_t> active_slots;
        active_slots.reserve(n_live);
        for (size_t slot = 0; slot < n_slots; ++slot) {
            if (slot_is_active(*idx, slot)) {
                active_slots.push_back(slot);
            }
        }

        for (int list = 0; list < actual_lists; ++list) {
            const size_t slot = active_slots[static_cast<size_t>(list) * active_slots.size() /
                static_cast<size_t>(actual_lists)];
            float * centroid = centroids.data() + static_cast<size_t>(list) * dim_sz;
            decode_slot_to_f32(*idx, slot, centroid);
        }

        for (int iter = 0; iter < n_iter; ++iter) {
            std::fill(next_centroids.begin(), next_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t slot : active_slots) {
                decode_slot_to_f32(*idx, slot, row.data());
                const size_t list = best_centroid(row.data(), centroids, actual_lists, dim);
                float * dst = next_centroids.data() + list * dim_sz;
                for (int i = 0; i < dim; ++i) {
                    dst[i] += row[static_cast<size_t>(i)];
                }
                ++counts[list];
            }

            for (int list = 0; list < actual_lists; ++list) {
                float * centroid = centroids.data() + static_cast<size_t>(list) * dim_sz;
                if (counts[static_cast<size_t>(list)] == 0) {
                    continue;
                }
                const float inv_count = 1.0f /
                    static_cast<float>(counts[static_cast<size_t>(list)]);
                const float * src =
                    next_centroids.data() + static_cast<size_t>(list) * dim_sz;
                for (int i = 0; i < dim; ++i) {
                    centroid[i] = src[static_cast<size_t>(i)] * inv_count;
                }
            }
        }

        for (size_t slot : active_slots) {
            decode_slot_to_f32(*idx, slot, row.data());
            const size_t list = best_centroid(row.data(), centroids, actual_lists, dim);
            lists[list].push_back(slot);
        }

        idx->ivf_centroids = std::move(centroids);
        idx->ivf_lists = std::move(lists);
        idx->ivf_n_lists = actual_lists;
        idx->ivf_generation = idx->generation;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_build_ivf(ggml_vec_index_t * idx, int n_lists, int n_iter) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_build_ivf_unlocked(idx, n_lists, n_iter);
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

static int ggml_vec_index_search_impl(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    bool                     filtered,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    const ggml_vec_index_filter_t * prepared_filter,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr || queries == nullptr ||
        out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0 ||
        (filtered && prepared_filter == nullptr &&
         (n_allowed < 0 || (n_allowed > 0 && allowed_ids == nullptr)))) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }

    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
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

        std::vector<size_t> allowed_slots;
        const std::vector<size_t> * allowed_ptr = nullptr;
        if (prepared_filter != nullptr) {
            if (prepared_filter->owner != idx ||
                prepared_filter->owner_cookie != idx->filter_cookie ||
                prepared_filter->dim != idx->dim ||
                prepared_filter->bit_width != idx->bit_width ||
                prepared_filter->generation != idx->generation) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            allowed_ptr = &prepared_filter->slots;
        } else if (filtered) {
            allowed_slots = allowed_slots_for_ids(*idx, allowed_ids, n_allowed);
            allowed_ptr = &allowed_slots;
        }

        for (int q = 0; q < n_q; ++q) {
            search_one(
                *idx,
                queries + static_cast<size_t>(q) * static_cast<size_t>(dim),
                k,
                out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
                out_ids    + static_cast<size_t>(q) * static_cast<size_t>(k),
                allowed_ptr);
        }
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids) {
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, false, nullptr, 0, nullptr, out_scores, out_ids);
}

int ggml_vec_index_search_filtered(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    float                  * out_scores,
    uint64_t               * out_ids) {
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, true, allowed_ids, n_allowed, nullptr, out_scores, out_ids);
}

ggml_vec_index_filter_t * ggml_vec_index_filter_create(
    const ggml_vec_index_t * idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed) {
    try {
        if (idx == nullptr || n_allowed < 0 ||
            (n_allowed > 0 && allowed_ids == nullptr)) {
            return nullptr;
        }
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        auto * filter = new (std::nothrow) ggml_vec_index_filter();
        if (filter == nullptr) {
            return nullptr;
        }
        std::unique_ptr<ggml_vec_index_filter> owned(filter);
        owned->owner = idx;
        owned->owner_cookie = idx->filter_cookie;
        owned->dim = idx->dim;
        owned->bit_width = idx->bit_width;
        owned->generation = idx->generation;
        owned->slots = allowed_slots_for_ids(*idx, allowed_ids, n_allowed);
        return owned.release();
    } catch (...) {
        return nullptr;
    }
}

void ggml_vec_index_filter_free(ggml_vec_index_filter_t * filter) {
    delete filter;
}

int ggml_vec_index_search_prepared_filtered(
    const ggml_vec_index_t        * idx,
    const ggml_vec_index_filter_t * filter,
    const float                   * queries,
    int                             n_q,
    int                             k,
    float                         * out_scores,
    uint64_t                      * out_ids) {
    if (filter == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    return ggml_vec_index_search_impl(
        idx, queries, n_q, k, true, nullptr, 0, filter, out_scores, out_ids);
}

int ggml_vec_index_search_ivf(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    int                      nprobe,
    float                  * out_scores,
    uint64_t               * out_ids) {

    if (idx == nullptr || queries == nullptr ||
        out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0 || nprobe <= 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }

    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const int dim = idx->dim;
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        if ((dim_sz != 0 && n_q_sz > std::numeric_limits<size_t>::max() / dim_sz) ||
            n_q_sz > std::numeric_limits<size_t>::max() / k_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(queries, n_q_sz * dim_sz)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->ivf_generation != idx->generation ||
            idx->ivf_n_lists < 0 ||
            static_cast<size_t>(idx->ivf_n_lists) != idx->ivf_lists.size() ||
            idx->ivf_centroids.size() != static_cast<size_t>(idx->ivf_n_lists) * dim_sz) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        const int probe_count = std::min(nprobe, idx->ivf_n_lists);
        for (int q = 0; q < n_q; ++q) {
            const float * query = queries + static_cast<size_t>(q) * dim_sz;
            float * scores = out_scores + static_cast<size_t>(q) * k_sz;
            uint64_t * ids = out_ids + static_cast<size_t>(q) * k_sz;

            if (idx->ivf_n_lists == 0) {
                const std::vector<size_t> empty_slots;
                search_one(*idx, query, k, scores, ids, &empty_slots);
                continue;
            }

            std::vector<ScoreId> centroid_scores;
            centroid_scores.reserve(static_cast<size_t>(idx->ivf_n_lists));
            for (int list = 0; list < idx->ivf_n_lists; ++list) {
                const float score = dot(
                    query,
                    idx->ivf_centroids.data() + static_cast<size_t>(list) * dim_sz,
                    dim);
                centroid_scores.push_back({ score, static_cast<uint64_t>(list) });
            }
            std::sort(
                centroid_scores.begin(),
                centroid_scores.end(),
                [](const ScoreId & a, const ScoreId & b) {
                    return a.score > b.score;
                });

            std::vector<size_t> selected_lists;
            selected_lists.reserve(static_cast<size_t>(probe_count));
            size_t candidate_count = 0;
            for (const ScoreId & centroid : centroid_scores) {
                const size_t list_id = static_cast<size_t>(centroid.id);
                const auto & list = idx->ivf_lists[list_id];
                if (list.empty()) {
                    continue;
                }
                selected_lists.push_back(list_id);
                candidate_count += list.size();
                if (selected_lists.size() == static_cast<size_t>(probe_count)) {
                    break;
                }
            }
            std::vector<size_t> candidate_slots;
            candidate_slots.reserve(candidate_count);
            for (size_t list_id : selected_lists) {
                const auto & list = idx->ivf_lists[list_id];
                candidate_slots.insert(candidate_slots.end(), list.begin(), list.end());
            }
            search_one(*idx, query, k, scores, ids, &candidate_slots);
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

static int ggml_vec_index_write_unlocked(ggml_vec_index_t * idx, const char * path) {
    try {
        if (idx == nullptr || path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap &&
            !idx->mapped_source_path.empty() &&
            filesystem_paths_equal(idx->mapped_source_path.c_str(), path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (active_count(*idx) > std::numeric_limits<uint32_t>::max()) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n      = active_count(*idx);
        const size_t n_slots = idx->slot_to_id.size();
        const size_t dim_sz = static_cast<size_t>(idx->dim);
        if (dim_sz != 0 && n_slots > std::numeric_limits<size_t>::max() / dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }
        if (!has_vector_storage(*idx) ||
            (is_q4(*idx) && idx->q4_scale.size() != n_slots) ||
            (is_q8(*idx) && idx->q8_scale.size() != n_slots)) {
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
        const uint32_t n_le   = static_cast<uint32_t>(n);
        put_u32_le(header + 8, dim_le);
        put_u32_le(header + 12, n_le);
        put_u32_le(header + 16, is_quantized(*idx) ? kQParamScaleF32 : kQParamNone);
        put_u32_le(header + 20, is_quantized(*idx) ? 4u : 0u);
        put_u32_le(header + 24, is_q4(*idx) ? 0u : (is_q8(*idx) ? 1u : 4u));
        put_u32_le(header + 28, 0);

        if (!write_bytes(f, header, sizeof(header))) {
            return fail_io();
        }

        uint32_t header_crc  = crc32c_update(0xffffffffu, header, sizeof(header));
        uint32_t qparams_crc = 0xffffffffu;
        uint32_t vectors_crc = 0xffffffffu;
        uint32_t ids_crc     = 0xffffffffu;
        if (is_quantized(*idx)) {
            const std::vector<float> & scales = is_q4(*idx) ? idx->q4_scale : idx->q8_scale;
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                const float scale = scales[slot];
                if (!write_u32_le_crc(f, float_to_u32(scale), qparams_crc)) {
                    return fail_io();
                }
            }

            if (is_q4(*idx)) {
                const size_t row_bytes = q4_row_bytes(dim_sz);
                const uint8_t * data = q4_data_ptr(*idx);
                for (size_t slot = 0; slot < n_slots; ++slot) {
                    if (!slot_is_active(*idx, slot)) {
                        continue;
                    }
                    const uint8_t * row = data + slot * row_bytes;
                    if (!write_bytes(f, row, row_bytes)) {
                        return fail_io();
                    }
                    vectors_crc = crc32c_update(vectors_crc, row, row_bytes);
                }
            } else {
                const size_t row_bytes = dim_sz * sizeof(int8_t);
                const int8_t * data = q8_data_ptr(*idx);
                for (size_t slot = 0; slot < n_slots; ++slot) {
                    if (!slot_is_active(*idx, slot)) {
                        continue;
                    }
                    const int8_t * row = data + slot * dim_sz;
                    if (!write_bytes(f, row, row_bytes)) {
                        return fail_io();
                    }
                    vectors_crc = crc32c_update(vectors_crc, row, row_bytes);
                }
            }
        } else {
            const float * data = f32_data_ptr(*idx);
            for (size_t slot = 0; slot < n_slots; ++slot) {
                if (!slot_is_active(*idx, slot)) {
                    continue;
                }
                for (size_t i = 0; i < dim_sz; ++i) {
                    const float v = data[slot * dim_sz + i];
                    if (!write_u32_le_crc(f, float_to_u32(v), vectors_crc)) {
                        return fail_io();
                    }
                }
            }
        }

        for (size_t slot = 0; slot < n_slots; ++slot) {
            if (!slot_is_active(*idx, slot)) {
                continue;
            }
            if (!write_u64_le_crc(f, idx->slot_to_id[slot], ids_crc)) {
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

int ggml_vec_index_write(ggml_vec_index_t * idx, const char * path) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        return ggml_vec_index_write_unlocked(idx, path);
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
            ((bit_width == 4 && (kind != kStorageQ4 || qparam_type != kQParamScaleF32 ||
                                  qparam_bytes != 4 || comp_bytes != 0)) ||
             (bit_width == 8 && (kind != kStorageQ8 || qparam_type != kQParamScaleF32 ||
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
        if (n > kMaxIndexLen) {
            return nullptr;
        }
        if (n != 0 && dim_sz > std::numeric_limits<size_t>::max() / n) {
            return nullptr;
        }

        test_maybe_throw_bad_alloc();
        if (is_q4(*idx)) {
            idx->q4_data.resize(n * q4_row_bytes(dim_sz));
            idx->q4_scale.resize(n);
        } else if (is_q8(*idx)) {
            idx->q8_data.resize(n * dim_sz);
            idx->q8_scale.resize(n);
        } else {
            idx->data.resize(n * dim_sz);
        }
        idx->slot_to_id.resize(n);
        idx->slot_active.assign(n, 1);
        idx->n_active = n;
        idx->id_to_slot.reserve(n);

        const bool checksummed = (flags & kFlagCRC32C) != 0;
        uint32_t header_crc =
            crc32c_update(0xffffffffu, header, kTvimHeaderSize);
        uint32_t qparams_crc = 0xffffffffu;
        uint32_t vectors_crc = 0xffffffffu;
        uint32_t ids_crc = 0xffffffffu;

        if (version == kTvimVersionV1 && is_quantized(*idx)) {
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
                if (is_q4(*idx)) {
                    quantize_q4_row(
                        row.data(),
                        idx->q4_data.data() + slot * q4_row_bytes(dim_sz),
                        dim,
                        idx->q4_scale[slot]);
                } else {
                    quantize_q8_row(
                        row.data(),
                        idx->q8_data.data() + slot * dim_sz,
                        dim,
                        idx->q8_scale[slot]);
                }
            }
        } else if (is_quantized(*idx)) {
            std::vector<float> & scales = is_q4(*idx) ? idx->q4_scale : idx->q8_scale;
            for (float & scale : scales) {
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

            std::vector<uint8_t> * q4_data = is_q4(*idx) ? &idx->q4_data : nullptr;
            if (is_q4(*idx)) {
                if (!q4_data->empty()) {
                    if (q4_data->size() >
                        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                        return nullptr;
                    }
                    f.read(
                        reinterpret_cast<char *>(q4_data->data()),
                        static_cast<std::streamsize>(q4_data->size() * sizeof(uint8_t)));
                    if (!f) {
                        return nullptr;
                    }
                    if (checksummed) {
                        vectors_crc = crc32c_update(vectors_crc, q4_data->data(), q4_data->size());
                    }
                }
                const size_t row_bytes = q4_row_bytes(dim_sz);
                for (size_t slot = 0; slot < n; ++slot) {
                    const float scale = idx->q4_scale[slot];
                    const uint8_t * row = idx->q4_data.data() + slot * row_bytes;
                    for (size_t i = 0; i < dim_sz; ++i) {
                        const uint8_t byte = row[i / 2];
                        const uint8_t nibble = (i & 1) == 0 ?
                            static_cast<uint8_t>(byte & 0x0f) :
                            static_cast<uint8_t>(byte >> 4);
                        if (nibble == 0 ||
                            !std::isfinite(static_cast<float>(q4_decode(nibble)) * scale)) {
                            return nullptr;
                        }
                    }
                    if ((dim_sz & 1) != 0 && (row[row_bytes - 1] >> 4) != 8) {
                        return nullptr;
                    }
                }
            } else {
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
            if (!is_valid_id(id)) {
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

ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path) {
    try {
        if (path == nullptr || !host_is_little_endian()) {
            return nullptr;
        }

        auto mapped = std::make_unique<MappedFile>();
        if (!map_file_readonly(path, *mapped) || mapped->size < kTvimHeaderSize) {
            return nullptr;
        }

        const auto * bytes = static_cast<const uint8_t *>(mapped->data);
        if (std::memcmp(bytes, kTvimMagic, 4) != 0 || bytes[4] != kTvimVersion) {
            return nullptr;
        }

        const uint8_t flags = bytes[7];
        if ((flags & ~kFlagCRC32C) != 0 || get_u32_le(bytes + 28) != 0) {
            return nullptr;
        }

        const int bit_width = static_cast<int>(bytes[5]);
        if (!is_supported_bit_width(bit_width)) {
            return nullptr;
        }
        const uint8_t kind = bytes[6];
        const uint32_t dim_le = get_u32_le(bytes + 8);
        const uint32_t n_le = get_u32_le(bytes + 12);
        const uint32_t qparam_type = get_u32_le(bytes + 16);
        const uint32_t qparam_bytes = get_u32_le(bytes + 20);
        const uint32_t comp_bytes = get_u32_le(bytes + 24);
        if (dim_le == 0 || dim_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return nullptr;
        }
        if ((bit_width == 4 && (kind != kStorageQ4 || qparam_type != kQParamScaleF32 ||
                                qparam_bytes != 4 || comp_bytes != 0)) ||
            (bit_width == 8 && (kind != kStorageQ8 || qparam_type != kQParamScaleF32 ||
                                qparam_bytes != 4 || comp_bytes != 1)) ||
            (bit_width == 32 && (kind != kStorageF32 || qparam_type != kQParamNone ||
                                 qparam_bytes != 0 || comp_bytes != 4))) {
            return nullptr;
        }

        uint64_t expected_size = 0;
        if (!expected_file_size(
                kTvimHeaderSize,
                n_le,
                dim_le,
                qparam_bytes,
                comp_bytes,
                expected_size)) {
            return nullptr;
        }
        const bool checksummed = (flags & kFlagCRC32C) != 0;
        if (checksummed && !checked_add_u64(expected_size, kTvimChecksumSize, expected_size)) {
            return nullptr;
        }
        if (expected_size != static_cast<uint64_t>(mapped->size)) {
            return nullptr;
        }

        uint64_t qparams_bytes_u64 = 0;
        uint64_t vectors_bytes_u64 = 0;
        uint64_t ids_bytes_u64 = 0;
        uint64_t component_count_u64 = 0;
        if (!checked_mul_u64(n_le, qparam_bytes, qparams_bytes_u64) ||
            !checked_mul_u64(n_le, sizeof(uint64_t), ids_bytes_u64)) {
            return nullptr;
        }
        if (comp_bytes == 0) {
            uint64_t row_bytes = 0;
            if (!checked_add_u64(dim_le, 1, row_bytes)) {
                return nullptr;
            }
            row_bytes /= 2;
            if (!checked_mul_u64(n_le, row_bytes, vectors_bytes_u64)) {
                return nullptr;
            }
        } else if (!checked_mul_u64(n_le, dim_le, component_count_u64) ||
                   !checked_mul_u64(component_count_u64, comp_bytes, vectors_bytes_u64)) {
            return nullptr;
        }
        const uint64_t qparams_offset = kTvimHeaderSize;
        const uint64_t vectors_offset = qparams_offset + qparams_bytes_u64;
        const uint64_t ids_offset = vectors_offset + vectors_bytes_u64;
        const uint64_t checksums_offset = ids_offset + ids_bytes_u64;

        if (checksummed) {
            const uint32_t header_crc =
                crc32c_update(0xffffffffu, bytes, kTvimHeaderSize) ^ 0xffffffffu;
            const uint32_t qparams_crc =
                crc32c_update(
                    0xffffffffu,
                    bytes + static_cast<size_t>(qparams_offset),
                    static_cast<size_t>(qparams_bytes_u64)) ^ 0xffffffffu;
            const uint32_t vectors_crc =
                crc32c_update(
                    0xffffffffu,
                    bytes + static_cast<size_t>(vectors_offset),
                    static_cast<size_t>(vectors_bytes_u64)) ^ 0xffffffffu;
            const uint32_t ids_crc =
                crc32c_update(
                    0xffffffffu,
                    bytes + static_cast<size_t>(ids_offset),
                    static_cast<size_t>(ids_bytes_u64)) ^ 0xffffffffu;
            if (header_crc != get_u32_le(bytes + static_cast<size_t>(checksums_offset)) ||
                qparams_crc != get_u32_le(bytes + static_cast<size_t>(checksums_offset + 4)) ||
                vectors_crc != get_u32_le(bytes + static_cast<size_t>(checksums_offset + 8)) ||
                ids_crc != get_u32_le(bytes + static_cast<size_t>(checksums_offset + 12))) {
                return nullptr;
            }
        }

        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(static_cast<int>(dim_le), bit_width),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return nullptr;
        }

        const size_t n = static_cast<size_t>(n_le);
        if (n > kMaxIndexLen) {
            return nullptr;
        }
        const size_t dim_sz = static_cast<size_t>(dim_le);
        idx->slot_to_id.resize(n);
        idx->slot_active.assign(n, 1);
        idx->n_active = n;
        idx->id_to_slot.reserve(n);
        if (is_quantized(*idx)) {
            std::vector<float> & scales = is_q4(*idx) ? idx->q4_scale : idx->q8_scale;
            scales.resize(n);
            for (size_t slot = 0; slot < n; ++slot) {
                const uint32_t bits = get_u32_le(
                    bytes + static_cast<size_t>(qparams_offset) + slot * sizeof(uint32_t));
                scales[slot] = u32_to_float(bits);
                if (!std::isfinite(scales[slot]) || scales[slot] <= 0.0f) {
                    return nullptr;
                }
            }
        }

        if (bit_width == 4) {
            const size_t row_bytes = q4_row_bytes(dim_sz);
            const auto * q4 = bytes + static_cast<size_t>(vectors_offset);
            for (size_t slot = 0; slot < n; ++slot) {
                const float scale = idx->q4_scale[slot];
                const uint8_t * row = q4 + slot * row_bytes;
                for (size_t i = 0; i < dim_sz; ++i) {
                    const uint8_t byte = row[i / 2];
                    const uint8_t nibble = (i & 1) == 0 ?
                        static_cast<uint8_t>(byte & 0x0f) :
                        static_cast<uint8_t>(byte >> 4);
                    if (nibble == 0 ||
                        !std::isfinite(static_cast<float>(q4_decode(nibble)) * scale)) {
                        return nullptr;
                    }
                }
                if ((dim_sz & 1) != 0 && (row[row_bytes - 1] >> 4) != 8) {
                    return nullptr;
                }
            }
            idx->mapped_q4_data = q4;
        } else if (bit_width == 8) {
            const auto * q8 = reinterpret_cast<const int8_t *>(
                bytes + static_cast<size_t>(vectors_offset));
            for (size_t slot = 0; slot < n; ++slot) {
                const float scale = idx->q8_scale[slot];
                const int8_t * row = q8 + slot * dim_sz;
                for (size_t i = 0; i < dim_sz; ++i) {
                    if (row[i] == std::numeric_limits<int8_t>::min() ||
                        !std::isfinite(static_cast<float>(row[i]) * scale)) {
                        return nullptr;
                    }
                }
            }
            idx->mapped_q8_data = q8;
        } else {
            const auto * f32 = reinterpret_cast<const float *>(
                bytes + static_cast<size_t>(vectors_offset));
            const size_t count = n * dim_sz;
            if (!all_finite(f32, count)) {
                return nullptr;
            }
            idx->mapped_data = f32;
        }

        const uint8_t * ids = bytes + static_cast<size_t>(ids_offset);
        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = get_u64_le(ids + slot * sizeof(uint64_t));
            if (!is_valid_id(id)) {
                return nullptr;
            }
            idx->slot_to_id[slot] = id;
            if (!idx->id_to_slot.emplace(id, slot).second) {
                return nullptr;
            }
        }

        idx->read_only_mmap = true;
        idx->mapped_source_path = path;
        idx->mapped_vector_bytes = static_cast<size_t>(vectors_bytes_u64);
        idx->mapped_file = std::move(mapped);
        return idx.release();
    } catch (...) {
        return nullptr;
    }
}

namespace {

bool expected_add_delta_payload_size(uint64_t n, uint64_t dim, uint64_t & size) {
    uint64_t id_bytes = 0;
    uint64_t values = 0;
    uint64_t vector_bytes = 0;
    if (!checked_mul_u64(n, sizeof(uint64_t), id_bytes) ||
        !checked_mul_u64(n, dim, values) ||
        !checked_mul_u64(values, sizeof(uint32_t), vector_bytes) ||
        !checked_add_u64(id_bytes, vector_bytes, size)) {
        return false;
    }
    return true;
}

bool read_delta_payload(std::ifstream & f, uint64_t payload_bytes, std::vector<uint8_t> & payload) {
    if (payload_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        payload_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    payload.resize(static_cast<size_t>(payload_bytes));
    if (!payload.empty()) {
        f.read(
            reinterpret_cast<char *>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
        if (!f) {
            return false;
        }
    }
    return true;
}

bool replay_add_delta(
        ggml_vec_index_t * idx,
        uint32_t n,
        const std::vector<uint8_t> & payload) {
    if (n == 0 || n > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    uint64_t expected_payload = 0;
    if (!expected_add_delta_payload_size(n, static_cast<uint64_t>(idx->dim), expected_payload) ||
        expected_payload != payload.size()) {
        return false;
    }

    const size_t n_sz = static_cast<size_t>(n);
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    std::vector<uint64_t> ids(n_sz);
    std::vector<float> vectors(n_sz * dim_sz);

    const uint8_t * ptr = payload.data();
    for (size_t i = 0; i < n_sz; ++i) {
        ids[i] = get_u64_le(ptr);
        ptr += sizeof(uint64_t);
    }
    for (float & v : vectors) {
        v = u32_to_float(get_u32_le(ptr));
        if (!std::isfinite(v)) {
            return false;
        }
        ptr += sizeof(uint32_t);
    }

    const int status = ggml_vec_index_add(
        idx,
        vectors.data(),
        static_cast<int>(n),
        ids.data());
    return status == GGML_VEC_INDEX_OK;
}

bool replay_remove_delta(ggml_vec_index_t * idx, uint32_t n, const std::vector<uint8_t> & payload) {
    if (n != 1 || payload.size() != sizeof(uint64_t)) {
        return false;
    }
    const uint64_t id = get_u64_le(payload.data());
    if (!is_valid_id(id)) {
        return false;
    }
    return ggml_vec_index_remove(idx, id) >= 0;
}

bool replay_delta_log(ggml_vec_index_t * idx, const char * delta_path) {
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(delta_path, fs_path)) {
        return false;
    }
    if (!std::filesystem::exists(fs_path) || std::filesystem::file_size(fs_path) == 0) {
        return true;
    }
    const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(fs_path));
    if (file_size < kTvidHeaderSize) {
        return false;
    }

    std::ifstream f(fs_path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    uint8_t header[kTvidHeaderSize] = {};
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!f ||
        std::memcmp(header, kTvidMagic, 4) != 0 ||
        header[4] != kTvidVersion ||
        header[5] != static_cast<uint8_t>(idx->bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx->dim)) {
        return false;
    }

    const uint32_t base_crc = get_u32_le(header + 12);
    const uint32_t snapshot_crc = index_state_crc32c(*idx);
    const bool apply_records = snapshot_crc == base_crc;
    uint32_t last_state_crc = base_crc;

    uint64_t offset = kTvidHeaderSize;
    while (offset < file_size) {
        uint8_t record[kTvidRecordHeaderSize] = {};
        f.read(reinterpret_cast<char *>(record), sizeof(record));
        if (f.gcount() == 0 && f.eof()) {
            break;
        }
        if (f.gcount() != static_cast<std::streamsize>(sizeof(record))) {
            break; // torn trailing record header
        }
        offset += kTvidRecordHeaderSize;

        const uint8_t op = record[0];
        const uint32_t n = get_u32_le(record + 4);
        const uint64_t payload_bytes = get_u64_le(record + 8);
        const uint32_t expected_crc = get_u32_le(record + 16);
        const uint32_t state_crc = get_u32_le(record + 20);
        if (record[1] != 0 || record[2] != 0 || record[3] != 0 ||
            (op != kTvidOpAdd && op != kTvidOpRemove)) {
            return false;
        }
        if (payload_bytes > file_size - offset) {
            break; // torn trailing record payload
        }

        std::vector<uint8_t> payload;
        if (!read_delta_payload(f, payload_bytes, payload)) {
            break;
        }
        offset += payload_bytes;

        uint32_t crc = crc32c_update(0xffffffffu, record, 16);
        crc = crc32c_update(crc, record + 20, 4);
        if (!payload.empty()) {
            crc = crc32c_update(crc, payload.data(), payload.size());
        }
        if ((crc ^ 0xffffffffu) != expected_crc) {
            return false;
        }

        if (apply_records) {
            if (op == kTvidOpAdd) {
                if (!replay_add_delta(idx, n, payload)) {
                    return false;
                }
            } else {
                if (!replay_remove_delta(idx, n, payload)) {
                    return false;
                }
            }
        }
        last_state_crc = state_crc;
    }

    if (apply_records) {
        return index_state_crc32c(*idx) == last_state_crc;
    }
    if (snapshot_crc != last_state_crc) {
        return false;
    }
    idx->delta_log_rebase_pending = base_crc != snapshot_crc;
    idx->delta_log_rebase_crc = snapshot_crc;
    return true;
}

} // namespace

ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path) {
    try {
        if (snapshot_path == nullptr || delta_path == nullptr) {
            return nullptr;
        }
        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_load(snapshot_path),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return nullptr;
        }
        if (!replay_delta_log(idx.get(), delta_path)) {
            return nullptr;
        }
        return idx.release();
    } catch (...) {
        return nullptr;
    }
}

int ggml_vec_index_compact_delta(
    ggml_vec_index_t * idx,
    const char       * snapshot_path,
    const char       * delta_path) {
    try {
        if (idx == nullptr || snapshot_path == nullptr || delta_path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (filesystem_paths_equal(snapshot_path, delta_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        DeltaLogLock delta_lock(delta_path);
        if (!delta_lock.ok()) {
            return GGML_VEC_INDEX_E_IO;
        }
        const int write_status = ggml_vec_index_write_unlocked(idx, snapshot_path);
        if (write_status != GGML_VEC_INDEX_OK) {
            return write_status;
        }
        const int delta_status = write_empty_delta_log_unlocked(*idx, delta_path);
        if (delta_status != GGML_VEC_INDEX_OK) {
            return delta_status;
        }
        idx->delta_log_rebase_pending = false;
        idx->delta_log_rebase_crc = 0;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int ggml_vec_index_len(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        const size_t n = active_count(*idx);
        return n > kMaxIndexLen ? std::numeric_limits<int>::max() : static_cast<int>(n);
    } catch (...) {
        return 0;
    }
}

int ggml_vec_index_dim(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        return idx->dim;
    } catch (...) {
        return 0;
    }
}

int ggml_vec_index_bit_width(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        return idx->bit_width;
    } catch (...) {
        return 0;
    }
}
