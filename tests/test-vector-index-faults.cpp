#include "ggml-vector-index.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" void ggml_vec_index_test_set_oom_countdown(int64_t countdown);
extern "C" void ggml_vec_index_test_set_write_fail_after(int64_t bytes);

namespace {

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

void reset_fault_hooks() {
    ggml_vec_index_test_set_oom_countdown(-1);
    ggml_vec_index_test_set_write_fail_after(-1);
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

std::string temp_file_prefix(const std::string & path) {
    return std::filesystem::path(path).filename().string() + ".tmp.";
}

void remove_temp_siblings(const std::string & path) {
    const std::filesystem::path p(path);
    const std::filesystem::path parent =
        p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    const std::string prefix = temp_file_prefix(path);
    for (const auto & entry : std::filesystem::directory_iterator(parent)) {
        const std::string name = entry.path().filename().string();
        if (name.compare(0, prefix.size(), prefix) == 0) {
            std::filesystem::remove(entry.path());
        }
    }
}

void expect_no_temp_siblings(const std::string & path) {
    const std::filesystem::path p(path);
    const std::filesystem::path parent =
        p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    const std::string prefix = temp_file_prefix(path);
    for (const auto & entry : std::filesystem::directory_iterator(parent)) {
        const std::string name = entry.path().filename().string();
        CHECK(name.compare(0, prefix.size(), prefix) != 0);
    }
}

} // namespace

int main() {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = { 101, 102 };

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(
        idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);

    {
        const std::array<float, 8> new_vectors = {
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        };
        const std::array<uint64_t, 2> new_ids = { 201, 202 };

        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_add(
            idx, new_vectors.data(), 2, new_ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
        CHECK(ggml_vec_index_len(idx) == 2);

        // Fail before the second map insertion, after the first was committed.
        ggml_vec_index_test_set_oom_countdown(3);
        CHECK(ggml_vec_index_add(
            idx, new_vectors.data(), 2, new_ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
        CHECK(ggml_vec_index_len(idx) == 2);
        CHECK(ggml_vec_index_contains(idx, new_ids[0]) == 0);
        CHECK(ggml_vec_index_contains(idx, new_ids[1]) == 0);
    }

    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> ids{};
        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_search(
            idx, base_vectors.data(), 1, /*k=*/1,
            scores.data(), ids.data()) == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();
    }

    const std::string path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-fault-test.tvim").string();
    std::filesystem::remove(path);
    remove_temp_siblings(path);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> old_snapshot = read_file_bytes(path);

    const std::array<float, 4> extra_vector = { 0.5f, 0.5f, 0.5f, 0.5f };
    const uint64_t extra_id = 301;
    CHECK(ggml_vec_index_add(
        idx, extra_vector.data(), 1, &extra_id) == GGML_VEC_INDEX_OK);

    ggml_vec_index_test_set_oom_countdown(0);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_OOM);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    // The first checkpoint passes; the second fails after temp creation.
    ggml_vec_index_test_set_oom_countdown(1);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_OOM);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    ggml_vec_index_test_set_write_fail_after(40);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(read_file_bytes(path) == old_snapshot);
    expect_no_temp_siblings(path);

    ggml_vec_index_test_set_oom_countdown(0);
    CHECK(ggml_vec_index_load(path.c_str()) == nullptr);
    reset_fault_hooks();

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 2);
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(idx);
    std::filesystem::remove(path);

    std::printf("test-vector-index-faults: OK\n");
    return 0;
}
