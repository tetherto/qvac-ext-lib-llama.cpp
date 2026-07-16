#include "ggml-vector-index.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

extern "C" void ggml_vec_index_test_set_oom_countdown(int64_t countdown);
extern "C" void ggml_vec_index_test_set_write_fail_after(int64_t bytes);
extern "C" void ggml_vec_index_test_set_truncate_fail(int fail);
extern "C" void ggml_vec_index_test_set_parent_fsync_fail(int fail);
extern "C" void ggml_vec_index_test_set_delta_append_wait_target(int target);
extern "C" void ggml_vec_index_test_reset_delta_tail_scan_count(void);
extern "C" int64_t ggml_vec_index_test_get_delta_tail_scan_count(void);

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
    ggml_vec_index_test_set_truncate_fail(0);
    ggml_vec_index_test_set_parent_fsync_fail(0);
    ggml_vec_index_test_set_delta_append_wait_target(0);
    ggml_vec_index_test_reset_delta_tail_scan_count();
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

void test_quantized_logged_faults(int bit_width) {
    constexpr int dim = 4;
    const std::array<float, 8> base_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> base_ids = {
        static_cast<uint64_t>(800 + bit_width),
        static_cast<uint64_t>(900 + bit_width),
    };
    const std::array<float, 4> logged_vector = { 0.25f, -0.5f, 0.75f, -1.0f };
    const std::array<float, 4> extra_vector = { -0.125f, 0.375f, -0.625f, 0.875f };

    const std::string suffix = std::to_string(bit_width);
    const std::string snapshot_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-quant-fault-base-" + suffix + ".tvim")).string();
    const std::string delta_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-quant-fault-log-" + suffix + ".tvid")).string();
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(
        idx, base_vectors.data(), static_cast<int>(base_ids.size()), base_ids.data()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

    const uint64_t failed_initial_id = static_cast<uint64_t>(1000 + bit_width);
    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &failed_initial_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_initial_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 2);
    if (std::filesystem::exists(delta_path)) {
        CHECK(std::filesystem::file_size(delta_path) == 0);
    }

    const uint64_t logged_id = static_cast<uint64_t>(1100 + bit_width);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_len(idx) == 3);
    const std::vector<uint8_t> old_delta = read_file_bytes(delta_path);

    const uint64_t failed_rollback_id = static_cast<uint64_t>(1200 + bit_width);
    ggml_vec_index_test_set_write_fail_after(8);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &failed_rollback_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_INTERNAL);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_rollback_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 3);
    CHECK(read_file_bytes(delta_path) == old_delta);

    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) ==
          GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(read_file_bytes(delta_path) == old_delta);

    CHECK(ggml_vec_index_remove_logged(idx, base_ids[0], delta_path.c_str()) == 1);
    CHECK(ggml_vec_index_contains(idx, base_ids[0]) == 0);
    CHECK(ggml_vec_index_len(idx) == 2);

    auto * replayed = ggml_vec_index_load_with_delta(snapshot_path.c_str(), delta_path.c_str());
    CHECK(replayed != nullptr);
    CHECK(ggml_vec_index_bit_width(replayed) == bit_width);
    CHECK(ggml_vec_index_len(replayed) == 2);
    CHECK(ggml_vec_index_contains(replayed, base_ids[0]) == 0);
    CHECK(ggml_vec_index_contains(replayed, base_ids[1]) == 1);
    CHECK(ggml_vec_index_contains(replayed, logged_id) == 1);

    ggml_vec_index_free(replayed);
    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
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

    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> ids{};
        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
            scores.data(), ids.data()) == GGML_VEC_INDEX_OK);

        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_E_OOM);
        reset_fault_hooks();

        CHECK(ggml_vec_index_search_ivf(
            idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
            scores.data(), ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ids[0] == base_ids[0]);
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

    {
        const std::string parent_fsync_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-parent-fsync-test.tvim").string();
        std::filesystem::remove(parent_fsync_path);

        auto * parent_fsync_idx = ggml_vec_index_create(dim, /*bit_width=*/32);
        CHECK(parent_fsync_idx != nullptr);
        CHECK(ggml_vec_index_add(
            parent_fsync_idx, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(parent_fsync_idx, parent_fsync_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        const std::vector<uint8_t> before_parent_fsync =
            read_file_bytes(parent_fsync_path);

        const uint64_t parent_fsync_id = 302;
        CHECK(ggml_vec_index_add(
            parent_fsync_idx, extra_vector.data(), 1, &parent_fsync_id) ==
            GGML_VEC_INDEX_OK);
        ggml_vec_index_test_set_parent_fsync_fail(1);
        CHECK(ggml_vec_index_write(parent_fsync_idx, parent_fsync_path.c_str()) ==
              GGML_VEC_INDEX_E_IO);
        reset_fault_hooks();
        CHECK(read_file_bytes(parent_fsync_path) != before_parent_fsync);

        auto * parent_fsync_loaded = ggml_vec_index_load(parent_fsync_path.c_str());
        CHECK(parent_fsync_loaded != nullptr);
        CHECK(ggml_vec_index_len(parent_fsync_loaded) == 3);
        CHECK(ggml_vec_index_contains(parent_fsync_loaded, parent_fsync_id) == 1);
        ggml_vec_index_free(parent_fsync_loaded);
        ggml_vec_index_free(parent_fsync_idx);
        std::filesystem::remove(parent_fsync_path);
    }

    const std::string delta_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-fault-test.tvid").string();
    std::filesystem::remove(delta_path);

    const std::array<float, 4> logged_vector = { 0.0f, 0.0f, 1.0f, 0.0f };
    const uint64_t logged_id = 401;
    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 3);
    if (std::filesystem::exists(delta_path)) {
        CHECK(std::filesystem::file_size(delta_path) == 0);
    }

    const uint64_t allowed_id = base_ids[0];
    std::array<float, 1> logged_scores{};
    std::array<uint64_t, 1> logged_out_ids{};
    auto * stale_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(stale_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, stale_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(stale_filter);
    const std::vector<uint8_t> old_delta = read_file_bytes(delta_path);

    auto * rollback_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(rollback_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    const uint64_t failed_internal_id = 402;
    ggml_vec_index_test_set_write_fail_after(8);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, logged_vector.data(), 1, &failed_internal_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_INTERNAL);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, failed_internal_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(read_file_bytes(delta_path) == old_delta);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, rollback_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(logged_out_ids[0] == allowed_id);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(logged_out_ids[0] == allowed_id);
    ggml_vec_index_filter_free(rollback_filter);

    auto * committed_add_filter = ggml_vec_index_filter_create(idx, &allowed_id, 1);
    CHECK(committed_add_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    const uint64_t committed_add_id = 403;
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        idx, extra_vector.data(), 1, &committed_add_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, committed_add_id) == 1);
    CHECK(ggml_vec_index_len(idx) == 5);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, committed_add_filter, base_vectors.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(committed_add_filter);
    const std::vector<uint8_t> delta_after_committed_add = read_file_bytes(delta_path);

    ggml_vec_index_test_set_write_fail_after(8);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 1);
    CHECK(read_file_bytes(delta_path) == delta_after_committed_add);

    auto * remove_filter = ggml_vec_index_filter_create(idx, &logged_id, 1);
    CHECK(remove_filter != nullptr);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
          == GGML_VEC_INDEX_OK);
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_remove_logged(idx, logged_id, delta_path.c_str()) == 1);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(idx, logged_id) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(read_file_bytes(delta_path) != delta_after_committed_add);
    CHECK(ggml_vec_index_search_prepared_filtered(
        idx, remove_filter, logged_vector.data(), 1, /*k=*/1,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(
        idx, base_vectors.data(), 1, /*k=*/1, /*nprobe=*/2,
        logged_scores.data(), logged_out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(remove_filter);

    const std::string committed_add_snapshot_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-committed-add-base.tvim").string();
    const std::string committed_add_delta_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-committed-add-log.tvid").string();
    std::filesystem::remove(committed_add_snapshot_path);
    std::filesystem::remove(committed_add_delta_path);
    std::filesystem::remove(committed_add_delta_path + ".lock");

    auto * committed_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(committed_base != nullptr);
    CHECK(ggml_vec_index_add(
        committed_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(committed_base, committed_add_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t committed_replay_id = 601;
    ggml_vec_index_test_set_parent_fsync_fail(1);
    ggml_vec_index_test_set_truncate_fail(1);
    CHECK(ggml_vec_index_add_logged(
        committed_base, extra_vector.data(), 1,
        &committed_replay_id, committed_add_delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    reset_fault_hooks();
    CHECK(ggml_vec_index_contains(committed_base, committed_replay_id) == 1);
    auto * committed_replayed = ggml_vec_index_load_with_delta(
        committed_add_snapshot_path.c_str(), committed_add_delta_path.c_str());
    CHECK(committed_replayed != nullptr);
    CHECK(ggml_vec_index_len(committed_replayed) == 3);
    CHECK(ggml_vec_index_contains(committed_replayed, committed_replay_id) == 1);
    ggml_vec_index_free(committed_replayed);
    ggml_vec_index_free(committed_base);
    std::filesystem::remove(committed_add_snapshot_path);
    std::filesystem::remove(committed_add_delta_path);
    std::filesystem::remove(committed_add_delta_path + ".lock");

    const std::string compact_parent_snapshot_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-compact-parent-fsync-base.tvim").string();
    const std::string compact_parent_delta_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-compact-parent-fsync-log.tvid").string();
    std::filesystem::remove(compact_parent_snapshot_path);
    std::filesystem::remove(compact_parent_delta_path);
    std::filesystem::remove(compact_parent_delta_path + ".lock");

    auto * compact_parent = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(compact_parent != nullptr);
    CHECK(ggml_vec_index_add(
        compact_parent, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(compact_parent, compact_parent_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t compact_parent_id = 602;
    CHECK(ggml_vec_index_add_logged(
        compact_parent, extra_vector.data(), 1,
        &compact_parent_id, compact_parent_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_test_set_parent_fsync_fail(1);
    CHECK(ggml_vec_index_compact_delta(
        compact_parent,
        compact_parent_snapshot_path.c_str(),
        compact_parent_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
    reset_fault_hooks();
    CHECK(std::filesystem::file_size(compact_parent_delta_path) > 16);
    auto * compact_parent_replayed = ggml_vec_index_load_with_delta(
        compact_parent_snapshot_path.c_str(), compact_parent_delta_path.c_str());
    CHECK(compact_parent_replayed != nullptr);
    CHECK(ggml_vec_index_len(compact_parent_replayed) == 3);
    CHECK(ggml_vec_index_contains(compact_parent_replayed, compact_parent_id) == 1);
    ggml_vec_index_free(compact_parent_replayed);
    ggml_vec_index_free(compact_parent);
    std::filesystem::remove(compact_parent_snapshot_path);
    std::filesystem::remove(compact_parent_delta_path);
    std::filesystem::remove(compact_parent_delta_path + ".lock");

    const std::string cached_tail_snapshot_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-cached-tail-base.tvim").string();
    const std::string cached_tail_delta_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-cached-tail-log.tvid").string();
    std::filesystem::remove(cached_tail_snapshot_path);
    std::filesystem::remove(cached_tail_delta_path);
    std::filesystem::remove(cached_tail_delta_path + ".lock");

    auto * cached_tail = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(cached_tail != nullptr);
    CHECK(ggml_vec_index_add(
        cached_tail, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(cached_tail, cached_tail_snapshot_path.c_str()) ==
          GGML_VEC_INDEX_OK);
    const uint64_t cached_tail_id_a = 701;
    const uint64_t cached_tail_id_b = 702;
    ggml_vec_index_test_reset_delta_tail_scan_count();
    CHECK(ggml_vec_index_add_logged(
        cached_tail, logged_vector.data(), 1,
        &cached_tail_id_a, cached_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(
        cached_tail, extra_vector.data(), 1,
        &cached_tail_id_b, cached_tail_delta_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_remove_logged(
        cached_tail, base_ids[0], cached_tail_delta_path.c_str()) == 1);
    CHECK(ggml_vec_index_test_get_delta_tail_scan_count() == 0);

    auto * cached_tail_replayed = ggml_vec_index_load_with_delta(
        cached_tail_snapshot_path.c_str(), cached_tail_delta_path.c_str());
    CHECK(cached_tail_replayed != nullptr);
    CHECK(ggml_vec_index_len(cached_tail_replayed) == 3);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, base_ids[0]) == 0);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, cached_tail_id_a) == 1);
    CHECK(ggml_vec_index_contains(cached_tail_replayed, cached_tail_id_b) == 1);
    ggml_vec_index_free(cached_tail_replayed);
    ggml_vec_index_free(cached_tail);
    std::filesystem::remove(cached_tail_snapshot_path);
    std::filesystem::remove(cached_tail_delta_path);
    std::filesystem::remove(cached_tail_delta_path + ".lock");

    test_quantized_logged_faults(/*bit_width=*/8);
    test_quantized_logged_faults(/*bit_width=*/4);

    const std::string shared_snapshot_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-shared-delta-base.tvim").string();
    const std::string shared_delta_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-shared-delta-log.tvid").string();
    std::filesystem::remove(shared_snapshot_path);
    std::filesystem::remove(shared_delta_path);
    std::filesystem::remove(shared_delta_path + ".lock");

    auto * shared_base = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(shared_base != nullptr);
    CHECK(ggml_vec_index_add(
        shared_base, base_vectors.data(), 2, base_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(shared_base, shared_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(shared_base);

    auto * shared_a = ggml_vec_index_load(shared_snapshot_path.c_str());
    auto * shared_b = ggml_vec_index_load(shared_snapshot_path.c_str());
    CHECK(shared_a != nullptr);
    CHECK(shared_b != nullptr);
    const uint64_t shared_id_a = 501;
    const uint64_t shared_id_b = 502;
    std::atomic<bool> start_shared_appends{ false };
    int status_a = GGML_VEC_INDEX_E_INTERNAL;
    int status_b = GGML_VEC_INDEX_E_INTERNAL;
    ggml_vec_index_test_set_delta_append_wait_target(2);
    std::thread thread_a([&]() {
        while (!start_shared_appends.load()) {
            std::this_thread::yield();
        }
        status_a = ggml_vec_index_add_logged(
            shared_a, logged_vector.data(), 1, &shared_id_a, shared_delta_path.c_str());
    });
    std::thread thread_b([&]() {
        while (!start_shared_appends.load()) {
            std::this_thread::yield();
        }
        status_b = ggml_vec_index_add_logged(
            shared_b, extra_vector.data(), 1, &shared_id_b, shared_delta_path.c_str());
    });
    start_shared_appends.store(true);
    thread_a.join();
    thread_b.join();
    reset_fault_hooks();

    CHECK((status_a == GGML_VEC_INDEX_OK && status_b == GGML_VEC_INDEX_E_IO) ||
          (status_b == GGML_VEC_INDEX_OK && status_a == GGML_VEC_INDEX_E_IO));
    CHECK(ggml_vec_index_contains(shared_a, shared_id_a) ==
          (status_a == GGML_VEC_INDEX_OK ? 1 : 0));
    CHECK(ggml_vec_index_contains(shared_b, shared_id_b) ==
          (status_b == GGML_VEC_INDEX_OK ? 1 : 0));

    auto * shared_replayed = ggml_vec_index_load_with_delta(
        shared_snapshot_path.c_str(), shared_delta_path.c_str());
    CHECK(shared_replayed != nullptr);
    CHECK(ggml_vec_index_len(shared_replayed) == 3);
    CHECK(ggml_vec_index_contains(shared_replayed, base_ids[0]) == 1);
    CHECK(ggml_vec_index_contains(shared_replayed, base_ids[1]) == 1);
    CHECK(ggml_vec_index_contains(shared_replayed, shared_id_a) ==
          (status_a == GGML_VEC_INDEX_OK ? 1 : 0));
    CHECK(ggml_vec_index_contains(shared_replayed, shared_id_b) ==
          (status_b == GGML_VEC_INDEX_OK ? 1 : 0));

    ggml_vec_index_free(shared_replayed);
    ggml_vec_index_free(shared_a);
    ggml_vec_index_free(shared_b);
    std::filesystem::remove(shared_snapshot_path);
    std::filesystem::remove(shared_delta_path);
    std::filesystem::remove(shared_delta_path + ".lock");

    ggml_vec_index_free(idx);
    std::filesystem::remove(path);
    std::filesystem::remove(delta_path);

    std::printf("test-vector-index-faults: OK\n");
    return 0;
}
