// ggml-vector-index-persistence.cpp - snapshot, mmap, and delta persistence.

#include "ggml-vector-index-internal.h"

static void close_mapped_file(MappedFile & mapped) {
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
std::atomic<int> g_test_load_with_delta_pause_ms{ 0 };
std::atomic<int> g_test_load_with_delta_waiters{ 0 };
std::atomic<int64_t> g_test_delta_tail_scan_count{ 0 };
std::atomic<int64_t> g_test_state_crc_scan_count{ 0 };

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

void test_wait_after_load_with_delta_snapshot() {
    const int pause_ms = g_test_load_with_delta_pause_ms.load();
    if (pause_ms <= 0) {
        return;
    }
    g_test_load_with_delta_waiters.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
}
#else
void test_maybe_throw_bad_alloc() {}
bool test_consume_write_bytes(size_t) {
    return true;
}
void test_wait_after_delta_validate() {}
void test_wait_after_load_with_delta_snapshot() {}
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

static bool host_is_little_endian() {
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

static bool write_u32_le(std::FILE * f, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    return write_bytes(f, bytes, sizeof(bytes));
}

static bool read_u32_le(std::ifstream & f, uint32_t & v) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    return true;
}

static bool read_u64_le(std::ifstream & f, uint64_t & v) {
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

static bool read_u32_le_crc(std::ifstream & f, uint32_t & v, uint32_t & crc) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

static bool read_u64_le_crc(std::ifstream & f, uint64_t & v, uint32_t & crc) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u64_le(bytes);
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

static bool write_u32_le_crc(std::FILE * f, uint32_t v, uint32_t & crc) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    if (!write_bytes(f, bytes, sizeof(bytes))) {
        return false;
    }
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

static bool write_u64_le_crc(std::FILE * f, uint64_t v, uint32_t & crc) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    if (!write_bytes(f, bytes, sizeof(bytes))) {
        return false;
    }
    crc = crc32c_update(crc, bytes, sizeof(bytes));
    return true;
}

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

static bool expected_file_size(
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

static bool open_temp_file(const char * path, TempFile & temp) {
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

static bool map_file_readonly(const char * path, MappedFile & mapped) {
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) {
        return false;
    }
    mapped.file = CreateFileW(
        wide.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
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

static bool open_temp_file(const char * path, TempFile & temp) {
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

static bool map_file_readonly(const char * path, MappedFile & mapped) {
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

static void remove_temp_file(TempFile & temp) {
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

static bool flush_and_sync(std::FILE * stream) {
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

static bool fsync_parent_dir(const char * path) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    if (g_test_parent_fsync_fail.load()) {
        return false;
    }
#endif
#ifdef _WIN32
    // Windows has no supported directory fsync; rename_overwrite uses write-through flags.
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

static bool rename_overwrite(const TempFile & temp, const char * dst) {
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

void ggml_vec_index_test_set_load_with_delta_pause_ms(int pause_ms) {
    g_test_load_with_delta_waiters.store(0);
    g_test_load_with_delta_pause_ms.store(pause_ms);
}

void ggml_vec_index_test_reset_delta_tail_scan_count(void) {
    g_test_delta_tail_scan_count.store(0);
}

int64_t ggml_vec_index_test_get_delta_tail_scan_count(void) {
    return g_test_delta_tail_scan_count.load();
}

void ggml_vec_index_test_reset_state_crc_scan_count(void) {
    g_test_state_crc_scan_count.store(0);
}

int64_t ggml_vec_index_test_get_state_crc_scan_count(void) {
    return g_test_state_crc_scan_count.load();
}

int ggml_vec_index_test_get_load_with_delta_waiters(void) {
    return g_test_load_with_delta_waiters.load();
}
}
#endif

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
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    g_test_state_crc_scan_count.fetch_add(1);
#endif
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

static uint64_t rotl64(uint64_t v, int shift) {
    return (v << shift) | (v >> (64 - shift));
}

uint64_t slot_state_hash(const ggml_vec_index & idx, size_t slot) {
    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    auto update_u32 = [&](uint32_t v) {
        crc0 = crc32c_update_u32(crc0, v);
        crc1 = crc32c_update_u32(crc1, v ^ 0xa5a5a5a5u);
    };
    auto update_u64 = [&](uint64_t v) {
        crc0 = crc32c_update_u64(crc0, v);
        crc1 = crc32c_update_u64(crc1, v ^ 0xa5a5a5a5a5a5a5a5ull);
    };
    auto update_bytes = [&](const void * data, size_t size) {
        crc0 = crc32c_update(crc0, data, size);
        crc1 = crc32c_update(crc1, data, size);
    };

    const size_t dim_sz = static_cast<size_t>(idx.dim);
    update_u64(idx.slot_to_id[slot]);
    if (is_q4(idx)) {
        update_u32(float_to_u32(idx.q4_scale[slot]));
        const size_t row_bytes = q4_row_bytes(dim_sz);
        update_bytes(q4_data_ptr(idx) + slot * row_bytes, row_bytes);
    } else if (is_q8(idx)) {
        update_u32(float_to_u32(idx.q8_scale[slot]));
        update_bytes(q8_data_ptr(idx) + slot * dim_sz, dim_sz * sizeof(int8_t));
    } else {
        const float * data = f32_data_ptr(idx) + slot * dim_sz;
        for (size_t i = 0; i < dim_sz; ++i) {
            update_u32(float_to_u32(data[i]));
        }
    }
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

void add_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum += hash;
    idx.state_hash_sum_rot += rotl64(hash, 17);
}

void remove_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum -= hash;
    idx.state_hash_sum_rot -= rotl64(hash, 17);
}

static uint32_t index_state_token_from(
        const ggml_vec_index & idx,
        size_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(idx.bit_width));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(storage_kind(idx)));
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(n_active));
    crc = crc32c_update_u64(crc, hash_xor);
    crc = crc32c_update_u64(crc, hash_sum);
    crc = crc32c_update_u64(crc, hash_sum_rot);
    return crc ^ 0xffffffffu;
}

static DeltaStateWide index_state_wide_from(
        size_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    DeltaStateWide state;
    state.n_active = static_cast<uint64_t>(n_active);
    state.hash_xor = hash_xor;
    state.hash_sum = hash_sum;
    state.hash_sum_rot = hash_sum_rot;
    return state;
}

uint32_t index_state_token(const ggml_vec_index & idx) {
    return index_state_token_from(
        idx,
        active_count(idx),
        idx.state_hash_xor,
        idx.state_hash_sum,
        idx.state_hash_sum_rot);
}

uint32_t index_state_token_after_remove(const ggml_vec_index & idx, uint64_t id) {
    const auto it = idx.id_to_slot.find(id);
    if (it == idx.id_to_slot.end()) {
        return index_state_token(idx);
    }
    const uint64_t hash = slot_state_hash(idx, it->second);
    return index_state_token_from(
        idx,
        active_count(idx) - 1,
        idx.state_hash_xor ^ hash,
        idx.state_hash_sum - hash,
        idx.state_hash_sum_rot - rotl64(hash, 17));
}

DeltaStateWide index_state_wide(const ggml_vec_index & idx) {
    return index_state_wide_from(
        active_count(idx),
        idx.state_hash_xor,
        idx.state_hash_sum,
        idx.state_hash_sum_rot);
}

DeltaStateWide index_state_wide_after_remove(const ggml_vec_index & idx, uint64_t id) {
    const auto it = idx.id_to_slot.find(id);
    if (it == idx.id_to_slot.end()) {
        return index_state_wide(idx);
    }
    const uint64_t hash = slot_state_hash(idx, it->second);
    return index_state_wide_from(
        active_count(idx) - 1,
        idx.state_hash_xor ^ hash,
        idx.state_hash_sum - hash,
        idx.state_hash_sum_rot - rotl64(hash, 17));
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

static bool delta_lock_path(const char * path, std::filesystem::path & out) {
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

static bool delta_log_path_key(const char * path, std::string & out) {
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }

    std::error_code ec;
    std::filesystem::path resolved = std::filesystem::weakly_canonical(fs_path, ec);
    if (ec) {
        ec.clear();
        resolved = std::filesystem::absolute(fs_path, ec);
        if (ec) {
            return false;
        }
    }
    out = resolved.lexically_normal().u8string();
    return true;
}

static bool delta_file_stamp(const char * path, DeltaFileStamp & stamp) {
    stamp = {};
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }
    std::error_code ec;
    const auto write_time = std::filesystem::last_write_time(fs_path, ec);
    if (ec) {
        return false;
    }
#ifdef _WIN32
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) {
        return false;
    }
    HANDLE file = CreateFileW(
        wide.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    BY_HANDLE_FILE_INFORMATION info = {};
    const bool ok = GetFileInformationByHandle(file, &info) != 0;
    CloseHandle(file);
    if (!ok) {
        return false;
    }
    stamp.size =
        (static_cast<uint64_t>(info.nFileSizeHigh) << 32) |
        static_cast<uint64_t>(info.nFileSizeLow);
    stamp.volume_serial = info.dwVolumeSerialNumber;
    stamp.file_index =
        (static_cast<uint64_t>(info.nFileIndexHigh) << 32) |
        static_cast<uint64_t>(info.nFileIndexLow);
#else
    struct stat st;
    if (::stat(fs_path.c_str(), &st) != 0) {
        return false;
    }
    if (st.st_size < 0) {
        return false;
    }
    stamp.size = static_cast<uint64_t>(st.st_size);
    stamp.device = static_cast<uint64_t>(st.st_dev);
    stamp.inode = static_cast<uint64_t>(st.st_ino);
#endif
    stamp.write_time = write_time.time_since_epoch().count();
    stamp.valid = true;
    return true;
}

static bool delta_file_stamp_equal(const DeltaFileStamp & a, const DeltaFileStamp & b) {
    if (!a.valid || !b.valid ||
        a.size != b.size ||
        a.write_time != b.write_time) {
        return false;
    }
#ifdef _WIN32
    return a.volume_serial == b.volume_serial &&
        a.file_index == b.file_index;
#else
    return a.device == b.device &&
        a.inode == b.inode;
#endif
}

static std::shared_ptr<std::mutex> delta_log_process_mutex_for(const std::filesystem::path & lock_path) {
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

static bool open_append_file(const char * path, std::FILE ** out) {
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

DeltaStateKind delta_state_kind_for_format(DeltaLogFormat format) {
    if (format == DeltaLogFormat::v1) {
        return DeltaStateKind::legacy_crc;
    }
    if (format == DeltaLogFormat::v4) {
        return DeltaStateKind::wide_state;
    }
    return DeltaStateKind::state_token;
}

static uint8_t delta_log_version_for_format(DeltaLogFormat format) {
    switch (format) {
        case DeltaLogFormat::v1:
            return kTvidVersionV1;
        case DeltaLogFormat::v2:
            return kTvidVersion;
        case DeltaLogFormat::v3:
            return kTvidVersionV3;
        case DeltaLogFormat::v4:
            return kTvidVersionV4;
    }
    return kTvidVersionV4;
}

static bool delta_log_format_from_version(uint8_t version, DeltaLogFormat & format) {
    if (version == kTvidVersionV1) {
        format = DeltaLogFormat::v1;
        return true;
    }
    if (version == kTvidVersion) {
        format = DeltaLogFormat::v2;
        return true;
    }
    if (version == kTvidVersionV3) {
        format = DeltaLogFormat::v3;
        return true;
    }
    if (version == kTvidVersionV4) {
        format = DeltaLogFormat::v4;
        return true;
    }
    return false;
}

static bool delta_state_kind_from_version(uint8_t version, DeltaStateKind & state_kind) {
    DeltaLogFormat format = DeltaLogFormat::v3;
    if (!delta_log_format_from_version(version, format)) {
        return false;
    }
    state_kind = delta_state_kind_for_format(format);
    return true;
}

uint32_t current_delta_state(const ggml_vec_index & idx, DeltaStateKind state_kind) {
    return state_kind == DeltaStateKind::legacy_crc ?
        index_state_crc32c(idx) :
        index_state_token(idx);
}

DeltaStateWide current_delta_state_wide(const ggml_vec_index & idx) {
    return index_state_wide(idx);
}

static bool delta_state_wide_equal(const DeltaStateWide & a, const DeltaStateWide & b) {
    return a.n_active == b.n_active &&
        a.hash_xor == b.hash_xor &&
        a.hash_sum == b.hash_sum &&
        a.hash_sum_rot == b.hash_sum_rot;
}

static bool delta_state_matches(
        DeltaStateKind state_kind,
        uint32_t lhs_crc,
        const DeltaStateWide & lhs_wide,
        uint32_t rhs_crc,
        const DeltaStateWide & rhs_wide) {
    if (state_kind == DeltaStateKind::wide_state) {
        return delta_state_wide_equal(lhs_wide, rhs_wide);
    }
    return lhs_crc == rhs_crc;
}

static void put_delta_state_wide(uint8_t * dst, const DeltaStateWide & state) {
    put_u64_le(dst + 0, state.n_active);
    put_u64_le(dst + 8, state.hash_xor);
    put_u64_le(dst + 16, state.hash_sum);
    put_u64_le(dst + 24, state.hash_sum_rot);
}

static DeltaStateWide get_delta_state_wide(const uint8_t * src) {
    DeltaStateWide state;
    state.n_active = get_u64_le(src + 0);
    state.hash_xor = get_u64_le(src + 8);
    state.hash_sum = get_u64_le(src + 16);
    state.hash_sum_rot = get_u64_le(src + 24);
    return state;
}

static size_t delta_header_size_for_format(DeltaLogFormat format) {
    return format == DeltaLogFormat::v4 ? kTvidHeaderSizeV4 : kTvidHeaderSize;
}

static size_t delta_record_header_size_for_format(DeltaLogFormat format) {
    return format == DeltaLogFormat::v4 ? kTvidRecordHeaderSizeV4 : kTvidRecordHeaderSize;
}

static int delta_state_kind_cache_value(DeltaStateKind state_kind) {
    if (state_kind == DeltaStateKind::legacy_crc) {
        return 1;
    }
    if (state_kind == DeltaStateKind::state_token) {
        return 2;
    }
    return 3;
}

void invalidate_delta_tail_cache(ggml_vec_index & idx) {
    idx.delta_tail_cache = {};
}

static bool get_cached_delta_tail(
        const ggml_vec_index & idx,
        const char * path,
        DeltaStateKind state_kind,
        uint64_t expected_size,
        uint32_t & tail_crc,
        DeltaStateWide & tail_wide,
        uint64_t & complete_size) {
    if (!idx.delta_tail_cache.valid ||
        idx.delta_tail_cache.state_kind != delta_state_kind_cache_value(state_kind) ||
        idx.delta_tail_cache.complete_size != expected_size) {
        return false;
    }

    std::string path_key;
    DeltaFileStamp stamp;
    if (!delta_log_path_key(path, path_key) ||
        !delta_file_stamp(path, stamp) ||
        path_key != idx.delta_tail_cache.path_key ||
        !delta_file_stamp_equal(stamp, idx.delta_tail_cache.stamp)) {
        return false;
    }

    tail_crc = idx.delta_tail_cache.tail_crc;
    tail_wide = idx.delta_tail_cache.tail_wide;
    complete_size = idx.delta_tail_cache.complete_size;
    return true;
}

static void update_delta_tail_cache(
        ggml_vec_index & idx,
        const char * path,
        DeltaStateKind state_kind,
        uint32_t tail_crc,
        const DeltaStateWide & tail_wide) {
    std::string path_key;
    DeltaFileStamp stamp;
    if (!delta_log_path_key(path, path_key) ||
        !delta_file_stamp(path, stamp)) {
        invalidate_delta_tail_cache(idx);
        return;
    }

    idx.delta_tail_cache.valid = true;
    idx.delta_tail_cache.path_key = path_key;
    idx.delta_tail_cache.state_kind = delta_state_kind_cache_value(state_kind);
    idx.delta_tail_cache.tail_crc = tail_crc;
    idx.delta_tail_cache.tail_wide = tail_wide;
    idx.delta_tail_cache.complete_size = stamp.size;
    idx.delta_tail_cache.stamp = stamp;
}

DeltaLogFormat delta_log_format_for_append(const char * path) {
    try {
        std::filesystem::path fs_path;
        if (!filesystem_path_from_utf8(path, fs_path) ||
            !std::filesystem::exists(fs_path) ||
            std::filesystem::file_size(fs_path) == 0) {
            return DeltaLogFormat::v4;
        }
        std::ifstream f(fs_path, std::ios::binary);
        if (!f.is_open()) {
            return DeltaLogFormat::v4;
        }
        uint8_t header[kTvidHeaderSize] = {};
        f.read(reinterpret_cast<char *>(header), sizeof(header));
        DeltaLogFormat format = DeltaLogFormat::v4;
        if (f &&
            std::memcmp(header, kTvidMagic, 4) == 0 &&
            delta_log_format_from_version(header[4], format)) {
            return format;
        }
    } catch (...) {
    }
    return DeltaLogFormat::v4;
}

static void fill_delta_header(
        const ggml_vec_index & idx,
        DeltaLogFormat format,
        uint32_t base_crc,
        const DeltaStateWide & base_wide,
        uint8_t * header) {
    std::memset(header, 0, delta_header_size_for_format(format));
    std::memcpy(header, kTvidMagic, 4);
    header[4] = delta_log_version_for_format(format);
    header[5] = static_cast<uint8_t>(idx.bit_width);
    put_u32_le(header + 8, static_cast<uint32_t>(idx.dim));
    if (format == DeltaLogFormat::v4) {
        put_delta_state_wide(header + 16, base_wide);
    } else {
        put_u32_le(header + 12, base_crc);
    }
}

static bool validate_delta_header(
        const char * path,
        const ggml_vec_index & idx,
        uint64_t & size,
        DeltaLogFormat & format,
        DeltaStateKind & state_kind,
        uint32_t & base_crc,
        DeltaStateWide & base_wide) {
    std::filesystem::path fs_path;
    if (!filesystem_path_from_utf8(path, fs_path)) {
        return false;
    }
    if (!std::filesystem::exists(fs_path)) {
        size = 0;
        format = DeltaLogFormat::v4;
        state_kind = DeltaStateKind::wide_state;
        base_crc = current_delta_state(idx, state_kind);
        base_wide = current_delta_state_wide(idx);
        return true;
    }
    size = static_cast<uint64_t>(std::filesystem::file_size(fs_path));
    if (size == 0) {
        format = DeltaLogFormat::v4;
        state_kind = DeltaStateKind::wide_state;
        base_crc = current_delta_state(idx, state_kind);
        base_wide = current_delta_state_wide(idx);
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
        !delta_log_format_from_version(header[4], format) ||
        header[5] != static_cast<uint8_t>(idx.bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx.dim)) {
        return false;
    }
    state_kind = delta_state_kind_for_format(format);
    if (size < delta_header_size_for_format(format)) {
        return false;
    }
    if (format == DeltaLogFormat::v4) {
        uint8_t wide_state[kTvidWideStateSize] = {};
        f.read(reinterpret_cast<char *>(wide_state), sizeof(wide_state));
        if (!f) {
            return false;
        }
        base_crc = 0;
        base_wide = get_delta_state_wide(wide_state);
        if (get_u32_le(header + 12) != 0) {
            return false;
        }
    } else {
        base_crc = get_u32_le(header + 12);
        base_wide = {};
    }
    return true;
}

static bool truncate_file_to(const char * path, uint64_t size) {
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

static bool inspect_delta_log_tail(
        const char * path,
        const ggml_vec_index & idx,
        uint32_t & last_state_crc,
        DeltaStateWide & last_state_wide,
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
    DeltaLogFormat format = DeltaLogFormat::v4;
    DeltaStateKind state_kind = DeltaStateKind::state_token;
    if (!f ||
        std::memcmp(header, kTvidMagic, 4) != 0 ||
        !delta_log_format_from_version(header[4], format) ||
        header[5] != static_cast<uint8_t>(idx.bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx.dim)) {
        return false;
    }
    state_kind = delta_state_kind_for_format(format);
    const size_t header_size = delta_header_size_for_format(format);
    if (file_size < header_size) {
        return false;
    }

    if (format == DeltaLogFormat::v4) {
        uint8_t wide_state[kTvidWideStateSize] = {};
        f.read(reinterpret_cast<char *>(wide_state), sizeof(wide_state));
        if (!f || get_u32_le(header + 12) != 0) {
            return false;
        }
        last_state_crc = 0;
        last_state_wide = get_delta_state_wide(wide_state);
    } else {
        last_state_crc = get_u32_le(header + 12);
        last_state_wide = {};
    }
    complete_size = header_size;
    uint64_t offset = header_size;
    while (offset < file_size) {
        uint8_t record[kTvidRecordHeaderSizeV4] = {};
        const size_t record_size = delta_record_header_size_for_format(format);
        f.read(reinterpret_cast<char *>(record), static_cast<std::streamsize>(record_size));
        if (f.gcount() != static_cast<std::streamsize>(record_size)) {
            return true;
        }
        offset += record_size;

        const uint8_t op = record[0];
        const uint64_t payload_bytes = get_u64_le(record + 8);
        const uint32_t expected_crc = get_u32_le(record + 16);
        const uint32_t state_crc = format == DeltaLogFormat::v4 ? 0 : get_u32_le(record + 20);
        const DeltaStateWide state_wide =
            format == DeltaLogFormat::v4 ? get_delta_state_wide(record + 24) : DeltaStateWide{};
        if (record[1] != 0 || record[2] != 0 || record[3] != 0 ||
            (op != kTvidOpAdd && op != kTvidOpRemove) ||
            (format == DeltaLogFormat::v4 && get_u32_le(record + 20) != 0)) {
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
        if (format == DeltaLogFormat::v4) {
            crc = crc32c_update(crc, record + 24, kTvidWideStateSize);
        } else {
            crc = crc32c_update(crc, record + 20, 4);
        }
        if (!payload.empty()) {
            crc = crc32c_update(crc, payload.data(), payload.size());
        }
        if ((crc ^ 0xffffffffu) != expected_crc) {
            return false;
        }
        last_state_crc = state_crc;
        last_state_wide = state_wide;
        complete_size = offset;
    }
    return true;
}

static bool delta_log_ends_at_state(
        const char * path,
        const ggml_vec_index & idx,
        DeltaStateKind state_kind,
        uint32_t state_crc,
        const DeltaStateWide & state_wide) {
    try {
        std::filesystem::path fs_path;
        if (!filesystem_path_from_utf8(path, fs_path)) {
            return false;
        }
        uint32_t tail_crc = 0;
        DeltaStateWide tail_wide;
        uint64_t complete_size = 0;
        return inspect_delta_log_tail(path, idx, tail_crc, tail_wide, complete_size) &&
            complete_size == static_cast<uint64_t>(std::filesystem::file_size(fs_path)) &&
            delta_state_matches(state_kind, tail_crc, tail_wide, state_crc, state_wide);
    } catch (...) {
        return false;
    }
}

static bool delta_log_matches_index_state(const char * path, const ggml_vec_index & idx) {
    uint64_t size = 0;
    uint32_t base_crc = 0;
    DeltaStateWide base_wide;
    DeltaLogFormat format = DeltaLogFormat::v4;
    DeltaStateKind state_kind = DeltaStateKind::wide_state;
    if (!validate_delta_header(path, idx, size, format, state_kind, base_crc, base_wide)) {
        return false;
    }

    const uint32_t current_crc = current_delta_state(idx, state_kind);
    const DeltaStateWide current_wide = current_delta_state_wide(idx);
    if (size == 0) {
        return delta_state_matches(state_kind, base_crc, base_wide, current_crc, current_wide);
    }

    uint32_t tail_crc = 0;
    DeltaStateWide tail_wide;
    uint64_t complete_size = 0;
    if (!get_cached_delta_tail(idx, path, state_kind, size, tail_crc, tail_wide, complete_size) &&
        !inspect_delta_log_tail(path, idx, tail_crc, tail_wide, complete_size)) {
        return false;
    }
    return complete_size == size &&
        delta_state_matches(state_kind, tail_crc, tail_wide, current_crc, current_wide);
}

DeltaAppendResult append_delta_record(
        ggml_vec_index & idx,
        const char * delta_path,
        DeltaLogFormat format,
        uint8_t op,
        uint32_t n,
        uint32_t base_crc_for_new_log,
        uint32_t state_crc,
        const DeltaStateWide & base_wide_for_new_log,
        const DeltaStateWide & state_wide,
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
    DeltaStateWide existing_base_wide;
    DeltaLogFormat existing_format = format;
    DeltaStateKind existing_state_kind = delta_state_kind_for_format(format);
    if (!validate_delta_header(
            delta_path,
            idx,
            old_size,
            existing_format,
            existing_state_kind,
            existing_base_crc,
            existing_base_wide)) {
        return { GGML_VEC_INDEX_E_IO, false };
    }
    if (old_size != 0) {
        format = existing_format;
    }
    const DeltaStateKind state_kind = delta_state_kind_for_format(format);
    if (old_size != 0) {
        uint32_t tail_crc = 0;
        DeltaStateWide tail_wide;
        uint64_t complete_size = 0;
        if (!get_cached_delta_tail(
                idx, delta_path, state_kind, old_size, tail_crc, tail_wide, complete_size) &&
            !inspect_delta_log_tail(delta_path, idx, tail_crc, tail_wide, complete_size)) {
            return { GGML_VEC_INDEX_E_IO, false };
        }
        if (!delta_state_matches(
                state_kind,
                tail_crc,
                tail_wide,
                base_crc_for_new_log,
                base_wide_for_new_log)) {
            return { GGML_VEC_INDEX_E_IO, false };
        }
        if (complete_size != old_size) {
            if (!truncate_file_to(delta_path, complete_size)) {
                return { GGML_VEC_INDEX_E_INTERNAL, false };
            }
            old_size = complete_size;
            invalidate_delta_tail_cache(idx);
        }
        if (idx.delta_log_rebase_pending &&
            idx.delta_log_rebase_state_kind == delta_state_kind_cache_value(state_kind) &&
            delta_state_matches(
                state_kind,
                idx.delta_log_rebase_crc,
                idx.delta_log_rebase_wide,
                base_crc_for_new_log,
                base_wide_for_new_log) &&
            !delta_state_matches(
                state_kind,
                existing_base_crc,
                existing_base_wide,
                base_crc_for_new_log,
                base_wide_for_new_log)) {
            // The snapshot already includes this log's records (e.g. crash after
            // compacting the snapshot but before replacing the old delta log).
            if (!truncate_file_to(delta_path, 0)) {
                return { GGML_VEC_INDEX_E_INTERNAL, false };
            }
            old_size = 0;
            invalidate_delta_tail_cache(idx);
            idx.delta_log_rebase_pending = false;
            idx.delta_log_rebase_crc = 0;
            idx.delta_log_rebase_wide = {};
            idx.delta_log_rebase_state_kind = 0;
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
            !truncated && delta_log_ends_at_state(
                delta_path, idx, state_kind, state_crc, state_wide);
        if (record_complete) {
            update_delta_tail_cache(idx, delta_path, state_kind, state_crc, state_wide);
        } else if (truncated) {
            invalidate_delta_tail_cache(idx);
        }
        return {
            truncated ? GGML_VEC_INDEX_E_IO : GGML_VEC_INDEX_E_INTERNAL,
            record_complete,
        };
    };

    if (old_size == 0) {
        uint8_t header[kTvidHeaderSizeV4] = {};
        const size_t header_size = delta_header_size_for_format(format);
        fill_delta_header(idx, format, base_crc_for_new_log, base_wide_for_new_log, header);
        if (!write_bytes(f, header, header_size)) {
            return fail_io();
        }
    }

    uint8_t record[kTvidRecordHeaderSizeV4] = {};
    const size_t record_size = delta_record_header_size_for_format(format);
    record[0] = op;
    put_u32_le(record + 4, n);
    put_u64_le(record + 8, static_cast<uint64_t>(payload.size()));
    if (format == DeltaLogFormat::v4) {
        put_delta_state_wide(record + 24, state_wide);
    } else {
        put_u32_le(record + 20, state_crc);
    }
    uint32_t crc = crc32c_update(0xffffffffu, record, 16);
    if (format == DeltaLogFormat::v4) {
        crc = crc32c_update(crc, record + 24, kTvidWideStateSize);
    } else {
        crc = crc32c_update(crc, record + 20, 4);
    }
    if (!payload.empty()) {
        crc = crc32c_update(crc, payload.data(), payload.size());
    }
    put_u32_le(record + 16, crc ^ 0xffffffffu);

    if (!write_bytes(f, record, record_size) ||
        (!payload.empty() && !write_bytes(f, payload.data(), payload.size())) ||
        !flush_and_sync(f)) {
        return fail_io();
    }
    const int close_result = std::fclose(f);
    f = nullptr;
    if (close_result != 0 || !fsync_parent_dir(delta_path)) {
        const bool truncated = truncate_file_to(delta_path, old_size);
        const bool record_complete =
            !truncated && delta_log_ends_at_state(
                delta_path, idx, state_kind, state_crc, state_wide);
        if (record_complete) {
            update_delta_tail_cache(idx, delta_path, state_kind, state_crc, state_wide);
        } else if (truncated) {
            invalidate_delta_tail_cache(idx);
        }
        return {
            truncated ? GGML_VEC_INDEX_E_IO : GGML_VEC_INDEX_E_INTERNAL,
            record_complete,
        };
    }
    update_delta_tail_cache(idx, delta_path, state_kind, state_crc, state_wide);
    return { GGML_VEC_INDEX_OK, true };
}

static int write_empty_delta_log_unlocked(const ggml_vec_index & idx, const char * delta_path) {
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

        uint8_t header[kTvidHeaderSizeV4] = {};
        const DeltaStateWide state_wide = index_state_wide(idx);
        fill_delta_header(idx, DeltaLogFormat::v4, 0, state_wide, header);
        if (!write_bytes(temp.stream, header, kTvidHeaderSizeV4) ||
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

static int write_empty_delta_log(const ggml_vec_index & idx, const char * delta_path) {
    if (delta_path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    DeltaLogLock delta_lock(delta_path);
    if (!delta_lock.ok()) {
        return GGML_VEC_INDEX_E_IO;
    }
    return write_empty_delta_log_unlocked(idx, delta_path);
}

static bool snapshot_matches_index(const ggml_vec_index & idx, const char * snapshot_path) {
    std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> loaded(
        ggml_vec_index_load(snapshot_path),
        ggml_vec_index_free);
    return loaded != nullptr &&
        loaded->dim == idx.dim &&
        loaded->bit_width == idx.bit_width &&
        active_count(*loaded) == active_count(idx) &&
        delta_state_wide_equal(index_state_wide(*loaded), index_state_wide(idx));
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

static bool append_u64_payload(std::vector<uint8_t> & payload, uint64_t value) {
    uint8_t bytes[8];
    put_u64_le(bytes, value);
    payload.insert(payload.end(), bytes, bytes + sizeof(bytes));
    return true;
}

static bool append_u32_payload(std::vector<uint8_t> & payload, uint32_t value) {
    uint8_t bytes[4];
    put_u32_le(bytes, value);
    payload.insert(payload.end(), bytes, bytes + sizeof(bytes));
    return true;
}

bool build_add_delta_payload_f32(
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
        append_u64_payload(payload, ids[i]);
    }
    for (size_t i = 0; i < vector_count; ++i) {
        append_u32_payload(payload, float_to_u32(vectors[i]));
    }
    return true;
}

bool build_add_delta_payload_from_slots(
        const ggml_vec_index_t * idx,
        size_t base_slot,
        int n,
        std::vector<uint8_t> & payload) {
    const size_t n_sz = static_cast<size_t>(n);
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    const size_t id_bytes = n_sz * sizeof(uint64_t);
    uint64_t payload_size = 0;
    if (is_q4(*idx)) {
        uint64_t code_bytes = 0;
        if (!checked_mul_u64(n_sz, q4_row_bytes(dim_sz), code_bytes) ||
            !checked_add_u64(id_bytes, n_sz * sizeof(uint32_t), payload_size) ||
            !checked_add_u64(payload_size, code_bytes, payload_size)) {
            return false;
        }
    } else if (is_q8(*idx)) {
        uint64_t code_bytes = 0;
        if (!checked_mul_u64(n_sz, dim_sz, code_bytes) ||
            !checked_add_u64(id_bytes, n_sz * sizeof(uint32_t), payload_size) ||
            !checked_add_u64(payload_size, code_bytes, payload_size)) {
            return false;
        }
    } else {
        uint64_t vector_count = 0;
        uint64_t vector_bytes = 0;
        if (!checked_mul_u64(n_sz, dim_sz, vector_count) ||
            !checked_mul_u64(vector_count, sizeof(uint32_t), vector_bytes) ||
            !checked_add_u64(id_bytes, vector_bytes, payload_size)) {
            return false;
        }
    }
    if (payload_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    payload.clear();
    payload.reserve(static_cast<size_t>(payload_size));
    for (int i = 0; i < n; ++i) {
        append_u64_payload(payload, idx->slot_to_id[base_slot + static_cast<size_t>(i)]);
    }

    if (is_q4(*idx)) {
        for (int i = 0; i < n; ++i) {
            append_u32_payload(
                payload,
                float_to_u32(idx->q4_scale[base_slot + static_cast<size_t>(i)]));
        }
        const size_t row_bytes = q4_row_bytes(dim_sz);
        const uint8_t * data = q4_data_ptr(*idx) + base_slot * row_bytes;
        payload.insert(payload.end(), data, data + n_sz * row_bytes);
    } else if (is_q8(*idx)) {
        for (int i = 0; i < n; ++i) {
            append_u32_payload(
                payload,
                float_to_u32(idx->q8_scale[base_slot + static_cast<size_t>(i)]));
        }
        const int8_t * data = q8_data_ptr(*idx) + base_slot * dim_sz;
        const auto * bytes = reinterpret_cast<const uint8_t *>(data);
        payload.insert(payload.end(), bytes, bytes + n_sz * dim_sz);
    } else {
        const float * data = f32_data_ptr(*idx) + base_slot * dim_sz;
        for (size_t i = 0; i < n_sz * dim_sz; ++i) {
            append_u32_payload(payload, float_to_u32(data[i]));
        }
    }
    return true;
}

std::vector<uint8_t> build_remove_delta_payload(uint64_t id) {
    std::vector<uint8_t> payload(sizeof(uint64_t));
    put_u64_le(payload.data(), id);
    return payload;
}

static thread_local int g_last_load_error = GGML_VEC_INDEX_OK;

static ggml_vec_index_t * load_fail(int error) {
    g_last_load_error = error;
    return nullptr;
}

static int load_status_from_last_error(void) {
    return g_last_load_error == GGML_VEC_INDEX_OK ? GGML_VEC_INDEX_E_IO : g_last_load_error;
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

int ggml_vec_index_load_ex(const char * path, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    ggml_vec_index_t * idx = ggml_vec_index_load(path);
    if (idx == nullptr) {
        return load_status_from_last_error();
    }
    *out = idx;
    return GGML_VEC_INDEX_OK;
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    g_last_load_error = GGML_VEC_INDEX_OK;
    try {
        if (path == nullptr) {
            return load_fail(GGML_VEC_INDEX_E_INVALID_ARG);
        }
        std::ifstream f;
#ifdef _WIN32
        std::wstring wide_path;
        if (!utf8_to_wide(path, wide_path)) {
            return load_fail(GGML_VEC_INDEX_E_INVALID_ARG);
        }
        f.open(std::filesystem::path(wide_path), std::ios::binary);
#else
        f.open(path, std::ios::binary);
#endif
        if (!f.is_open()) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        uint8_t header[kTvimHeaderSize] = {};
        f.read(reinterpret_cast<char *>(header), kTvimV1HeaderSize);
        if (!f || f.gcount() != static_cast<std::streamsize>(kTvimV1HeaderSize)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (std::memcmp(header, kTvimMagic, 4) != 0) {
            return load_fail(GGML_VEC_INDEX_E_BAD_MAGIC);
        }

        const uint8_t version = header[4];
        if (version != kTvimVersionV1 && version != kTvimVersion) {
            return load_fail(GGML_VEC_INDEX_E_BAD_VERSION);
        }
        if (version == kTvimVersion) {
            f.read(
                reinterpret_cast<char *>(header + kTvimV1HeaderSize),
                kTvimHeaderSize - kTvimV1HeaderSize);
            if (!f ||
                f.gcount() != static_cast<std::streamsize>(
                    kTvimHeaderSize - kTvimV1HeaderSize)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        }

        const uint8_t flags = version == kTvimVersion ? header[7] : 0;
        if ((version == kTvimVersionV1 && (header[6] != 0 || header[7] != 0)) ||
            (version == kTvimVersion &&
             ((flags & ~kFlagCRC32C) != 0 || get_u32_le(header + 28) != 0))) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        const int serialized_bit_width = static_cast<int>(header[5]);
        if ((version == kTvimVersionV1 &&
             (serialized_bit_width <= 0 || serialized_bit_width > 32)) ||
            (version == kTvimVersion && !is_supported_bit_width(serialized_bit_width))) {
            return load_fail(GGML_VEC_INDEX_E_IO);
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
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (version == kTvimVersion &&
            ((bit_width == 4 && (kind != kStorageQ4 || qparam_type != kQParamScaleF32 ||
                                  qparam_bytes != 4 || comp_bytes != 0)) ||
             (bit_width == 8 && (kind != kStorageQ8 || qparam_type != kQParamScaleF32 ||
                                  qparam_bytes != 4 || comp_bytes != 1)) ||
             (bit_width == 32 && (kind != kStorageF32 || qparam_type != kQParamNone ||
                                   qparam_bytes != 0 || comp_bytes != 4)))) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        uint64_t expected_size = 0;
        if (!expected_file_size(
                version == kTvimVersion ? kTvimHeaderSize : kTvimV1HeaderSize,
                n_le,
                dim_le,
                qparam_bytes,
                comp_bytes,
                expected_size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if ((flags & kFlagCRC32C) != 0 &&
            !checked_add_u64(expected_size, kTvimChecksumSize, expected_size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (expected_size >
            static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        f.seekg(0, std::ios::end);
        const std::streamoff actual_size = f.tellg();
        if (!f || actual_size != static_cast<std::streamoff>(expected_size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        f.seekg(
            static_cast<std::streamoff>(
                version == kTvimVersion ? kTvimHeaderSize : kTvimV1HeaderSize),
            std::ios::beg);
        if (!f) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        const int dim = static_cast<int>(dim_le);

        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(dim, bit_width),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return load_fail(GGML_VEC_INDEX_E_OOM);
        }
        const size_t dim_sz = static_cast<size_t>(dim);
        const size_t n      = static_cast<size_t>(n_le);
        if (n > kMaxIndexLen) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (n != 0 && dim_sz > std::numeric_limits<size_t>::max() / n) {
            return load_fail(GGML_VEC_INDEX_E_IO);
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
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                    v = u32_to_float(bits);
                    if (!std::isfinite(v)) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
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
                    return load_fail(GGML_VEC_INDEX_E_IO);
                }
                scale = u32_to_float(bits);
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    return load_fail(GGML_VEC_INDEX_E_IO);
                }
            }

            std::vector<uint8_t> * q4_data = is_q4(*idx) ? &idx->q4_data : nullptr;
            if (is_q4(*idx)) {
                if (!q4_data->empty()) {
                    if (q4_data->size() >
                        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                    f.read(
                        reinterpret_cast<char *>(q4_data->data()),
                        static_cast<std::streamsize>(q4_data->size() * sizeof(uint8_t)));
                    if (!f) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
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
                            return load_fail(GGML_VEC_INDEX_E_IO);
                        }
                    }
                    if ((dim_sz & 1) != 0 && (row[row_bytes - 1] >> 4) != 8) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                }
            } else {
                if (!idx->q8_data.empty()) {
                    if (idx->q8_data.size() >
                        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                    f.read(
                        reinterpret_cast<char *>(idx->q8_data.data()),
                        static_cast<std::streamsize>(idx->q8_data.size() * sizeof(int8_t)));
                    if (!f) {
                        return load_fail(GGML_VEC_INDEX_E_IO);
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
                            return load_fail(GGML_VEC_INDEX_E_IO);
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
                    return load_fail(GGML_VEC_INDEX_E_IO);
                }
                v = u32_to_float(bits);
                if (!std::isfinite(v)) {
                    return load_fail(GGML_VEC_INDEX_E_IO);
                }
            }
        }

        for (uint64_t & id : idx->slot_to_id) {
            const bool read_ok = checksummed ?
                read_u64_le_crc(f, id, ids_crc) :
                read_u64_le(f, id);
            if (!read_ok) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
            if (!is_valid_id(id)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
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
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        }

        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = idx->slot_to_id[slot];
            const bool inserted =
                idx->id_to_slot.emplace(id, slot).second;
            if (!inserted) {
                // Duplicate id in persisted file: corrupted.
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        }

        rebuild_state_hash(*idx);
        g_last_load_error = GGML_VEC_INDEX_OK;
        return idx.release();
    } catch (const std::bad_alloc &) {
        return load_fail(GGML_VEC_INDEX_E_OOM);
    } catch (...) {
        return load_fail(GGML_VEC_INDEX_E_INTERNAL);
    }
}

int ggml_vec_index_load_mmap_ex(const char * path, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    ggml_vec_index_t * idx = ggml_vec_index_load_mmap(path);
    if (idx == nullptr) {
        return load_status_from_last_error();
    }
    *out = idx;
    return GGML_VEC_INDEX_OK;
}

ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path) {
    g_last_load_error = GGML_VEC_INDEX_OK;
    try {
        if (path == nullptr || !host_is_little_endian()) {
            return load_fail(GGML_VEC_INDEX_E_INVALID_ARG);
        }

        auto mapped = std::make_unique<MappedFile>();
        if (!map_file_readonly(path, *mapped) || mapped->size < kTvimHeaderSize) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        const auto * bytes = static_cast<const uint8_t *>(mapped->data);
        if (std::memcmp(bytes, kTvimMagic, 4) != 0) {
            return load_fail(GGML_VEC_INDEX_E_BAD_MAGIC);
        }
        if (bytes[4] != kTvimVersion) {
            return load_fail(GGML_VEC_INDEX_E_BAD_VERSION);
        }

        const uint8_t flags = bytes[7];
        if ((flags & ~kFlagCRC32C) != 0 || get_u32_le(bytes + 28) != 0) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        const int bit_width = static_cast<int>(bytes[5]);
        if (!is_supported_bit_width(bit_width)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        const uint8_t kind = bytes[6];
        const uint32_t dim_le = get_u32_le(bytes + 8);
        const uint32_t n_le = get_u32_le(bytes + 12);
        const uint32_t qparam_type = get_u32_le(bytes + 16);
        const uint32_t qparam_bytes = get_u32_le(bytes + 20);
        const uint32_t comp_bytes = get_u32_le(bytes + 24);
        if (dim_le == 0 || dim_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if ((bit_width == 4 && (kind != kStorageQ4 || qparam_type != kQParamScaleF32 ||
                                qparam_bytes != 4 || comp_bytes != 0)) ||
            (bit_width == 8 && (kind != kStorageQ8 || qparam_type != kQParamScaleF32 ||
                                qparam_bytes != 4 || comp_bytes != 1)) ||
            (bit_width == 32 && (kind != kStorageF32 || qparam_type != kQParamNone ||
                                 qparam_bytes != 0 || comp_bytes != 4))) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        uint64_t expected_size = 0;
        if (!expected_file_size(
                kTvimHeaderSize,
                n_le,
                dim_le,
                qparam_bytes,
                comp_bytes,
                expected_size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        const bool checksummed = (flags & kFlagCRC32C) != 0;
        if (checksummed && !checked_add_u64(expected_size, kTvimChecksumSize, expected_size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (expected_size != static_cast<uint64_t>(mapped->size)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }

        uint64_t qparams_bytes_u64 = 0;
        uint64_t vectors_bytes_u64 = 0;
        uint64_t ids_bytes_u64 = 0;
        uint64_t component_count_u64 = 0;
        if (!checked_mul_u64(n_le, qparam_bytes, qparams_bytes_u64) ||
            !checked_mul_u64(n_le, sizeof(uint64_t), ids_bytes_u64)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        if (comp_bytes == 0) {
            uint64_t row_bytes = 0;
            if (!checked_add_u64(dim_le, 1, row_bytes)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
            row_bytes /= 2;
            if (!checked_mul_u64(n_le, row_bytes, vectors_bytes_u64)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        } else if (!checked_mul_u64(n_le, dim_le, component_count_u64) ||
                   !checked_mul_u64(component_count_u64, comp_bytes, vectors_bytes_u64)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
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
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        }

        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(static_cast<int>(dim_le), bit_width),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return load_fail(GGML_VEC_INDEX_E_OOM);
        }

        const size_t n = static_cast<size_t>(n_le);
        if (n > kMaxIndexLen) {
            return load_fail(GGML_VEC_INDEX_E_IO);
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
                    return load_fail(GGML_VEC_INDEX_E_IO);
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
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                }
                if ((dim_sz & 1) != 0 && (row[row_bytes - 1] >> 4) != 8) {
                    return load_fail(GGML_VEC_INDEX_E_IO);
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
                        return load_fail(GGML_VEC_INDEX_E_IO);
                    }
                }
            }
            idx->mapped_q8_data = q8;
        } else {
            const auto * f32 = reinterpret_cast<const float *>(
                bytes + static_cast<size_t>(vectors_offset));
            const size_t count = n * dim_sz;
            if (!all_finite(f32, count)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
            idx->mapped_data = f32;
        }

        const uint8_t * ids = bytes + static_cast<size_t>(ids_offset);
        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = get_u64_le(ids + slot * sizeof(uint64_t));
            if (!is_valid_id(id)) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
            idx->slot_to_id[slot] = id;
            if (!idx->id_to_slot.emplace(id, slot).second) {
                return load_fail(GGML_VEC_INDEX_E_IO);
            }
        }

        // Mmap handles are read-only snapshot views. Delta state identities are not
        // needed unless a mutable handle participates in delta-log operations.
        idx->read_only_mmap = true;
        idx->mapped_source_path = path;
        idx->mapped_vector_bytes = static_cast<size_t>(vectors_bytes_u64);
        idx->mapped_file = std::move(mapped);
        g_last_load_error = GGML_VEC_INDEX_OK;
        return idx.release();
    } catch (const std::bad_alloc &) {
        return load_fail(GGML_VEC_INDEX_E_OOM);
    } catch (...) {
        return load_fail(GGML_VEC_INDEX_E_INTERNAL);
    }
}

namespace {

bool expected_add_delta_payload_size_f32(uint64_t n, uint64_t dim, uint64_t & size) {
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

bool expected_add_delta_payload_size_native(
        uint64_t n,
        uint64_t dim,
        int bit_width,
        uint64_t & size) {
    uint64_t id_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t code_bytes = 0;
    if (!checked_mul_u64(n, sizeof(uint64_t), id_bytes) ||
        !checked_mul_u64(n, sizeof(uint32_t), scale_bytes)) {
        return false;
    }
    if (bit_width == 4) {
        uint64_t row_bytes = 0;
        if (!checked_add_u64(dim, 1, row_bytes)) {
            return false;
        }
        row_bytes /= 2;
        if (!checked_mul_u64(n, row_bytes, code_bytes)) {
            return false;
        }
    } else if (bit_width == 8) {
        if (!checked_mul_u64(n, dim, code_bytes)) {
            return false;
        }
    } else {
        return expected_add_delta_payload_size_f32(n, dim, size);
    }
    return checked_add_u64(id_bytes, scale_bytes, size) &&
        checked_add_u64(size, code_bytes, size);
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

int ggml_vec_index_add_encoded_unlocked(
        ggml_vec_index_t * idx,
        const uint64_t * ids,
        int n,
        const float * scales,
        const uint8_t * q4_codes,
        const int8_t * q8_codes,
        bool finalize) {
    size_t base_slot = 0;
    bool resized = false;

    auto rollback = [&]() noexcept {
        if (idx == nullptr || !resized) {
            return;
        }
        rollback_appended_slots_unlocked(idx, base_slot, ids, n);
    };

    try {
        if (idx == nullptr || ids == nullptr || scales == nullptr || n < 0 ||
            (!is_q4(*idx) && !is_q8(*idx)) ||
            (is_q4(*idx) && q4_codes == nullptr) ||
            (is_q8(*idx) && q8_codes == nullptr)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (idx->read_only_mmap) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }
        const int duplicate_status = check_logged_add_duplicates(idx, n, ids);
        if (duplicate_status != GGML_VEC_INDEX_OK) {
            return duplicate_status;
        }

        base_slot = idx->slot_to_id.size();
        const size_t dim_sz = static_cast<size_t>(idx->dim);
        const size_t n_sz = static_cast<size_t>(n);
        if (n_sz > kMaxIndexLen || active_count(*idx) > kMaxIndexLen - n_sz ||
            n_sz > std::numeric_limits<size_t>::max() - base_slot) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t new_slots = base_slot + n_sz;
        if (is_q4(*idx)) {
            const size_t row_bytes = q4_row_bytes(dim_sz);
            if (new_slots > std::numeric_limits<size_t>::max() / row_bytes) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            for (size_t slot = 0; slot < n_sz; ++slot) {
                const float scale = scales[slot];
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    return GGML_VEC_INDEX_E_INVALID_ARG;
                }
                const uint8_t * row = q4_codes + slot * row_bytes;
                for (size_t i = 0; i < dim_sz; ++i) {
                    const uint8_t byte = row[i / 2];
                    const uint8_t nibble = (i & 1) == 0 ?
                        static_cast<uint8_t>(byte & 0x0f) :
                        static_cast<uint8_t>(byte >> 4);
                    if (nibble == 0 ||
                        !std::isfinite(static_cast<float>(q4_decode(nibble)) * scale)) {
                        return GGML_VEC_INDEX_E_INVALID_ARG;
                    }
                }
                if ((dim_sz & 1) != 0 && (row[row_bytes - 1] >> 4) != 8) {
                    return GGML_VEC_INDEX_E_INVALID_ARG;
                }
            }
            resized = true;
            idx->q4_data.resize(new_slots * row_bytes);
            idx->q4_scale.resize(new_slots);
            std::memcpy(
                idx->q4_data.data() + base_slot * row_bytes,
                q4_codes,
                n_sz * row_bytes);
            std::memcpy(idx->q4_scale.data() + base_slot, scales, n_sz * sizeof(float));
        } else {
            if (dim_sz != 0 && new_slots > std::numeric_limits<size_t>::max() / dim_sz) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            for (size_t slot = 0; slot < n_sz; ++slot) {
                const float scale = scales[slot];
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    return GGML_VEC_INDEX_E_INVALID_ARG;
                }
                const int8_t * row = q8_codes + slot * dim_sz;
                for (size_t i = 0; i < dim_sz; ++i) {
                    if (row[i] == std::numeric_limits<int8_t>::min() ||
                        !std::isfinite(static_cast<float>(row[i]) * scale)) {
                        return GGML_VEC_INDEX_E_INVALID_ARG;
                    }
                }
            }
            resized = true;
            idx->q8_data.resize(new_slots * dim_sz);
            idx->q8_scale.resize(new_slots);
            std::memcpy(
                idx->q8_data.data() + base_slot * dim_sz,
                q8_codes,
                n_sz * dim_sz * sizeof(int8_t));
            std::memcpy(idx->q8_scale.data() + base_slot, scales, n_sz * sizeof(float));
        }

        idx->slot_to_id.resize(new_slots);
        idx->slot_active.resize(new_slots, 0);
        test_maybe_throw_bad_alloc();
        idx->id_to_slot.reserve(new_slots);
        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            idx->slot_to_id[slot] = ids[i];
            idx->slot_active[slot] = 1;
            idx->id_to_slot.emplace(ids[i], slot);
        }
        idx->n_active += n_sz;
        for (size_t slot = base_slot; slot < new_slots; ++slot) {
            add_state_hash(*idx, slot_state_hash(*idx, slot));
        }
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

bool replay_add_delta_f32(
        ggml_vec_index_t * idx,
        uint32_t n,
        const std::vector<uint8_t> & payload) {
    if (n == 0 || n > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    uint64_t expected_payload = 0;
    if (!expected_add_delta_payload_size_f32(n, static_cast<uint64_t>(idx->dim), expected_payload) ||
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

bool replay_add_delta_native(
        ggml_vec_index_t * idx,
        uint32_t n,
        const std::vector<uint8_t> & payload) {
    if (n == 0 || n > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (!is_quantized(*idx)) {
        return replay_add_delta_f32(idx, n, payload);
    }

    const size_t n_sz = static_cast<size_t>(n);
    const size_t dim_sz = static_cast<size_t>(idx->dim);
    uint64_t expected_payload = 0;
    if (!expected_add_delta_payload_size_native(
            n,
            static_cast<uint64_t>(idx->dim),
            idx->bit_width,
            expected_payload) ||
        expected_payload != payload.size()) {
        return false;
    }

    std::vector<uint64_t> ids(n_sz);
    std::vector<float> scales(n_sz);
    const uint8_t * ptr = payload.data();
    for (size_t i = 0; i < n_sz; ++i) {
        ids[i] = get_u64_le(ptr);
        ptr += sizeof(uint64_t);
    }
    for (float & scale : scales) {
        scale = u32_to_float(get_u32_le(ptr));
        ptr += sizeof(uint32_t);
    }

    int status = GGML_VEC_INDEX_E_INVALID_ARG;
    if (is_q4(*idx)) {
        const size_t code_bytes = n_sz * q4_row_bytes(dim_sz);
        std::vector<uint8_t> codes(ptr, ptr + code_bytes);
        status = ggml_vec_index_add_encoded_unlocked(
            idx,
            ids.data(),
            static_cast<int>(n),
            scales.data(),
            codes.data(),
            nullptr,
            true);
    } else {
        const size_t code_bytes = n_sz * dim_sz;
        std::vector<int8_t> codes(code_bytes);
        std::memcpy(codes.data(), ptr, code_bytes);
        status = ggml_vec_index_add_encoded_unlocked(
            idx,
            ids.data(),
            static_cast<int>(n),
            scales.data(),
            nullptr,
            codes.data(),
            true);
    }
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
    // Writers only append remove records for live ids; a miss means the log no
    // longer matches the snapshot lineage and should be treated as corruption.
    return ggml_vec_index_remove(idx, id) == 1;
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
    DeltaLogFormat format = DeltaLogFormat::v4;
    DeltaStateKind state_kind = DeltaStateKind::wide_state;
    if (!f ||
        std::memcmp(header, kTvidMagic, 4) != 0 ||
        !delta_log_format_from_version(header[4], format) ||
        header[5] != static_cast<uint8_t>(idx->bit_width) ||
        header[6] != 0 ||
        header[7] != 0 ||
        get_u32_le(header + 8) != static_cast<uint32_t>(idx->dim)) {
        return false;
    }
    state_kind = delta_state_kind_for_format(format);
    const size_t header_size = delta_header_size_for_format(format);
    if (file_size < header_size) {
        return false;
    }

    uint32_t base_crc = 0;
    DeltaStateWide base_wide;
    if (format == DeltaLogFormat::v4) {
        uint8_t wide_state[kTvidWideStateSize] = {};
        f.read(reinterpret_cast<char *>(wide_state), sizeof(wide_state));
        if (!f || get_u32_le(header + 12) != 0) {
            return false;
        }
        base_wide = get_delta_state_wide(wide_state);
    } else {
        base_crc = get_u32_le(header + 12);
    }
    const uint32_t snapshot_crc = current_delta_state(*idx, state_kind);
    const DeltaStateWide snapshot_wide = current_delta_state_wide(*idx);
    const bool apply_records =
        delta_state_matches(state_kind, snapshot_crc, snapshot_wide, base_crc, base_wide);
    uint32_t last_state_crc = base_crc;
    DeltaStateWide last_state_wide = base_wide;

    uint64_t offset = header_size;
    while (offset < file_size) {
        uint8_t record[kTvidRecordHeaderSizeV4] = {};
        const size_t record_size = delta_record_header_size_for_format(format);
        f.read(reinterpret_cast<char *>(record), static_cast<std::streamsize>(record_size));
        if (f.gcount() == 0 && f.eof()) {
            break;
        }
        if (f.gcount() != static_cast<std::streamsize>(record_size)) {
            break; // torn trailing record header
        }
        offset += record_size;

        const uint8_t op = record[0];
        const uint32_t n = get_u32_le(record + 4);
        const uint64_t payload_bytes = get_u64_le(record + 8);
        const uint32_t expected_crc = get_u32_le(record + 16);
        const uint32_t state_crc = format == DeltaLogFormat::v4 ? 0 : get_u32_le(record + 20);
        const DeltaStateWide state_wide =
            format == DeltaLogFormat::v4 ? get_delta_state_wide(record + 24) : DeltaStateWide{};
        if (record[1] != 0 || record[2] != 0 || record[3] != 0 ||
            (op != kTvidOpAdd && op != kTvidOpRemove) ||
            (format == DeltaLogFormat::v4 && get_u32_le(record + 20) != 0)) {
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
        if (format == DeltaLogFormat::v4) {
            crc = crc32c_update(crc, record + 24, kTvidWideStateSize);
        } else {
            crc = crc32c_update(crc, record + 20, 4);
        }
        if (!payload.empty()) {
            crc = crc32c_update(crc, payload.data(), payload.size());
        }
        if ((crc ^ 0xffffffffu) != expected_crc) {
            return false;
        }

        if (apply_records) {
            if (op == kTvidOpAdd) {
                const bool replayed =
                    (format == DeltaLogFormat::v3 || format == DeltaLogFormat::v4) ?
                    replay_add_delta_native(idx, n, payload) :
                    replay_add_delta_f32(idx, n, payload);
                if (!replayed) {
                    return false;
                }
            } else {
                if (!replay_remove_delta(idx, n, payload)) {
                    return false;
                }
            }
            if (!delta_state_matches(
                    state_kind,
                    current_delta_state(*idx, state_kind),
                    current_delta_state_wide(*idx),
                    state_crc,
                    state_wide)) {
                return false;
            }
        }
        last_state_crc = state_crc;
        last_state_wide = state_wide;
    }

    if (apply_records) {
        return delta_state_matches(
            state_kind,
            current_delta_state(*idx, state_kind),
            current_delta_state_wide(*idx),
            last_state_crc,
            last_state_wide);
    }
    if (!delta_state_matches(
            state_kind,
            snapshot_crc,
            snapshot_wide,
            last_state_crc,
            last_state_wide)) {
        return false;
    }
    idx->delta_log_rebase_pending = !delta_state_matches(
        state_kind, base_crc, base_wide, snapshot_crc, snapshot_wide);
    idx->delta_log_rebase_crc = snapshot_crc;
    idx->delta_log_rebase_wide = snapshot_wide;
    idx->delta_log_rebase_state_kind = delta_state_kind_cache_value(state_kind);
    return true;
}

} // namespace

int ggml_vec_index_load_with_delta_ex(
    const char * snapshot_path,
    const char * delta_path,
    ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    ggml_vec_index_t * idx = ggml_vec_index_load_with_delta(snapshot_path, delta_path);
    if (idx == nullptr) {
        return load_status_from_last_error();
    }
    *out = idx;
    return GGML_VEC_INDEX_OK;
}

ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path) {
    g_last_load_error = GGML_VEC_INDEX_OK;
    try {
        if (snapshot_path == nullptr || delta_path == nullptr) {
            return load_fail(GGML_VEC_INDEX_E_INVALID_ARG);
        }
        DeltaLogLock delta_lock(delta_path);
        if (!delta_lock.ok()) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_load(snapshot_path),
            ggml_vec_index_free);
        if (idx == nullptr) {
            return load_fail(load_status_from_last_error());
        }
        test_wait_after_load_with_delta_snapshot();
        if (!replay_delta_log(idx.get(), delta_path)) {
            return load_fail(GGML_VEC_INDEX_E_IO);
        }
        idx->delta_log_bound = true;
        g_last_load_error = GGML_VEC_INDEX_OK;
        return idx.release();
    } catch (const std::bad_alloc &) {
        return load_fail(GGML_VEC_INDEX_E_OOM);
    } catch (...) {
        return load_fail(GGML_VEC_INDEX_E_INTERNAL);
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
        if (idx->read_only_mmap) {
            rebuild_state_hash(*idx);
        }
        DeltaLogLock delta_lock(delta_path);
        if (!delta_lock.ok()) {
            return GGML_VEC_INDEX_E_IO;
        }
        if (!delta_log_matches_index_state(delta_path, *idx)) {
            return GGML_VEC_INDEX_E_IO;
        }
        const int write_status = ggml_vec_index_write_unlocked(idx, snapshot_path);
        if (write_status != GGML_VEC_INDEX_OK) {
            if (snapshot_matches_index(*idx, snapshot_path)) {
                return GGML_VEC_INDEX_E_PARTIAL_COMPACT;
            }
            return write_status;
        }
        const int delta_status = write_empty_delta_log_unlocked(*idx, delta_path);
        if (delta_status != GGML_VEC_INDEX_OK) {
            invalidate_delta_tail_cache(*idx);
            return GGML_VEC_INDEX_E_PARTIAL_COMPACT;
        }
        update_delta_tail_cache(
            *idx,
            delta_path,
            DeltaStateKind::wide_state,
            0,
            index_state_wide(*idx));
        idx->delta_log_rebase_pending = false;
        idx->delta_log_rebase_crc = 0;
        idx->delta_log_rebase_wide = {};
        idx->delta_log_rebase_state_kind = 0;
        idx->delta_log_bound = true;
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}
