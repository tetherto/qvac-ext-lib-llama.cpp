#pragma once
//
// ggml-vector-index: vector-index C API.
//
// This public C API supports full f32 storage (`bit_width=32`), q8 storage
// (`bit_width=8`), and packed q4
// storage (`bit_width=4`) with CPU search directly against quantized codes.
// q8 and q4 use NEON when available. Supported x86 CMake builds compile AVX2
// kernels separately and runtime-dispatch them from a non-AVX2 baseline.
//
// Threading: read-only APIs on the same handle can run concurrently. Mutations,
// persistence writes, compaction, and IVF builds are serialized with reads and
// with each other. The caller must still keep the handle alive for the duration
// of every API call. Prepared filter handles must also remain alive for the
// full duration of any `ggml_vec_index_search_prepared_filtered` call using
// them; do not free a filter concurrently with a search that uses it.
//
// Endianness: persistence format is fixed little-endian.

#include <stdint.h>

#ifndef GGML_API
#    ifdef GGML_SHARED
#        if defined(_WIN32) && !defined(__MINGW32__)
#            ifdef GGML_BUILD
#                define GGML_API __declspec(dllexport) extern
#            else
#                define GGML_API __declspec(dllimport) extern
#            endif
#        else
#            define GGML_API __attribute__ ((visibility ("default"))) extern
#        endif
#    else
#        define GGML_API extern
#    endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a vector index instance.
struct ggml_vec_index;
typedef struct ggml_vec_index ggml_vec_index_t;

// Prepared filtered-search handle. Valid only while the source index remains
// alive and at the generation it was created from; any successful add/remove
// invalidates existing filters.
struct ggml_vec_index_filter;
typedef struct ggml_vec_index_filter ggml_vec_index_filter_t;

// Error codes returned from int-valued APIs. 0 = OK. Negative = failure.
// `_remove` is the exception: it returns 1 on removal and 0 on miss.
enum ggml_vec_index_error {
    GGML_VEC_INDEX_OK            =  0,
    GGML_VEC_INDEX_E_INVALID_ARG = -2,
    GGML_VEC_INDEX_E_DUPLICATE   = -3,
    GGML_VEC_INDEX_E_IO          = -4,
    GGML_VEC_INDEX_E_BAD_MAGIC   = -5,
    GGML_VEC_INDEX_E_BAD_VERSION = -6,
    GGML_VEC_INDEX_E_OOM         = -7,
    GGML_VEC_INDEX_E_PARTIAL_COMPACT = -8,
    GGML_VEC_INDEX_E_INTERNAL    = -99,
};

// Lifecycle.
//
// `dim` must be > 0. `bit_width` must be 4, 8, or 32. `bit_width=4` and
// `bit_width=8` store per-vector symmetric quantized codes with one f32 scale
// per vector. `bit_width=32` stores full f32 vectors. Returns NULL on bad args.
GGML_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
// All vector components must be finite. UINT64_MAX is reserved for search
// result padding and is not a valid id. Live index length is capped at INT_MAX.
GGML_API int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids);

// Removes the entry for `id` by marking its internal slot deleted. Physical
// storage is compacted only when writing a snapshot. Returns 1 if removed,
// 0 if not present, negative on error.
GGML_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Physically removes deleted slots from in-memory storage. This does not write
// to disk. If any slots are removed, prepared filters and IVF state are
// invalidated. Returns 0 on success, negative on error.
GGML_API int ggml_vec_index_compact(ggml_vec_index_t * idx);

// Logged mutations for incremental persistence. These update `idx` and append
// a durable delta record to `delta_path`. Replay the log on top of a full .tvim
// snapshot with `ggml_vec_index_load_with_delta`.
//
// Delta logs are state-bound and single-writer per snapshot lineage. Use one
// evolving writer handle for a given {snapshot, delta_path} pair. If another
// handle or process appends to the same log, stale writers are rejected and
// must reload with `ggml_vec_index_load_with_delta` before appending again.
// If an append error occurs after a complete replayable record is observed,
// the mutation is treated as committed and the API returns OK.
GGML_API int ggml_vec_index_add_logged(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids,
    const char       * delta_path);

// Same return convention as `ggml_vec_index_remove`: 1 if removed, 0 if not
// present, negative on error.
GGML_API int ggml_vec_index_remove_logged(
    ggml_vec_index_t * idx,
    uint64_t           id,
    const char       * delta_path);

// Returns 1 if the id is in the index, 0 otherwise. NULL handles return 0.
// Read-only.
GGML_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// Placeholder for cache warming / codebook resolution after a bulk add.
// Currently a no-op. Use `ggml_vec_index_build_ivf` to build ANN state.
GGML_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Builds an in-memory IVF-flat approximate nearest-neighbor structure. This is
// not persisted in .tvim files; call again after loading if ANN search is
// needed. Successful add/remove calls invalidate the IVF structure.
// `n_lists` is capped to the current index length. `n_iter` controls centroid
// refinement; 0 uses deterministic initial centroids only.
GGML_API int ggml_vec_index_build_ivf(
    ggml_vec_index_t * idx,
    int                n_lists,
    int                n_iter);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar). If the index
// holds fewer than k entries, the remaining slots in each row are filled
// with UINT64_MAX ids; callers must use out_ids[i] == UINT64_MAX to identify
// padding. Padded score slots are filled with -FLT_MAX for compatibility, but
// that value can also be a legitimate finite dot product. Read-only against
// the index (does not mutate state).
// Exact search scans all live entries; use filtered or IVF search to reduce
// the candidate set.
//
// Score semantics: dot product. For f32 storage this is a full-precision dot
// product. For q4/q8 storage, the query remains f32 and the dot product is
// computed against dequantized indexed components:
// `query[i] * (q_code * per_vector_scale)`, without expanding the stored
// matrix back to f32. Callers that want cosine similarity must L2-normalize
// their vectors before insert AND before query; the index does NOT normalize
// internally. All query components must be finite.
GGML_API int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids);

// Filtered top-k search. Only entries whose ids appear in `allowed_ids` are
// considered. Missing ids are ignored; duplicate filter ids are treated once.
// `allowed_ids` may be NULL only when `n_allowed == 0`, which produces only
// sentinel results. The same filter is applied to every query row.
GGML_API int ggml_vec_index_search_filtered(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    const uint64_t         * allowed_ids,
    int                      n_allowed,
    float                  * out_scores,
    uint64_t               * out_ids);

// Prepared filtered search. Creating a filter maps, sorts, and deduplicates
// `allowed_ids` once, so callers can reuse it for repeated searches over the
// same allowlist. The source index must outlive every filter created from it.
// Stale filters return GGML_VEC_INDEX_E_INVALID_ARG.
GGML_API ggml_vec_index_filter_t * ggml_vec_index_filter_create(
    const ggml_vec_index_t * idx,
    const uint64_t         * allowed_ids,
    int                      n_allowed);

GGML_API void ggml_vec_index_filter_free(ggml_vec_index_filter_t * filter);

GGML_API int ggml_vec_index_search_prepared_filtered(
    const ggml_vec_index_t        * idx,
    const ggml_vec_index_filter_t * filter,
    const float                   * queries,
    int                             n_q,
    int                             k,
    float                         * out_scores,
    uint64_t                      * out_ids);

// IVF-flat ANN top-k search. `ggml_vec_index_build_ivf` must have been called
// after the most recent mutation. `nprobe` controls how many centroid lists are
// searched; higher values improve recall and lower the latency win. `nprobe`
// must be >= 1. If nprobe is greater than the number of built lists, all lists
// are searched.
GGML_API int ggml_vec_index_search_ivf(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    int                      nprobe,
    float                  * out_scores,
    uint64_t               * out_ids);

// Persistence. Format is .tvim version 2; see bottom of this header.
GGML_API int ggml_vec_index_write(
    ggml_vec_index_t * idx,
    const char       * path);

// Loads v2 files and migrates v1 f32 snapshots. Legacy bit_width=8 snapshots
// are quantized to q8; all other legacy bit widths migrate to f32/32-bit.
// Returns NULL on failure.
GGML_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

// Loads a v2 .tvim snapshot with its vector section memory-mapped read-only.
// IDs and quantization scales are copied into memory for lookup and scoring.
// Mutating APIs return GGML_VEC_INDEX_E_INVALID_ARG on mmap-backed handles.
// `ggml_vec_index_write` can snapshot mmap-backed handles, but callers must
// write to a different path than the mapped source file.
// Requires a little-endian host; use `ggml_vec_index_load` on other hosts.
// Returns NULL on failure or unsupported file format.
GGML_API ggml_vec_index_t * ggml_vec_index_load_mmap(const char * path);

// Loads a full .tvim snapshot and replays an append-only delta log. Missing
// delta logs are treated as empty.
GGML_API ggml_vec_index_t * ggml_vec_index_load_with_delta(
    const char * snapshot_path,
    const char * delta_path);

// Compacts incremental persistence by writing a full snapshot and replacing
// the delta log with an empty matching .tvid header. Returns
// GGML_VEC_INDEX_E_PARTIAL_COMPACT if the snapshot was written but replacing
// the delta log failed; loading snapshot+delta remains idempotent in this state.
GGML_API int ggml_vec_index_compact_delta(
    ggml_vec_index_t * idx,
    const char       * snapshot_path,
    const char       * delta_path);

// Stats. NULL handles return 0.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim version 2, all little-endian):
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version = 2
//   5       1      bit_width (4, 8, or 32)
//   6       1      storage kind (1 = f32, 2 = q8, 3 = q4)
//   7       1      flags (bit 0 = checksum trailer present)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      4      qparam_type (0 = none, 1 = per-vector f32 scale)
//   20      4      qparam_bytes_per_vector (0 or 4)
//   24      4      bytes_per_component (0 for packed q4, 1 for q8, 4 for f32)
//   28      4      reserved (zero)
//   32      ...    qparams:
//                    - f32: empty
//                    - q4/q8: N float32 scales
//   ...     ...    vectors:
//                    - f32: N*D float32 values, row-major
//                    - q8:  N*D int8 codes, row-major
//                    - q4:  N*ceil(D/2) packed unsigned nibbles, row-major
//   ...     N*8    ids (uint64)
//   ...     4      header CRC32C, when flag bit 0 is set
//   ...     4      qparams CRC32C, when flag bit 0 is set
//   ...     4      vectors CRC32C, when flag bit 0 is set
//   ...     4      ids CRC32C, when flag bit 0 is set
//
// Where N = n_vectors and D = dim. q8 uses symmetric per-vector quantization:
// scale = max(abs(v)) / 127, code = round(v / scale) clamped to [-127, 127].
// q4 uses scale = max(abs(v)) / 7, code = round(v / scale) clamped to [-7, 7],
// stored as unsigned nibble `code + 8` (0 is invalid). Zero vectors use
// scale = 1 and all-zero dequantized codes. Each CRC32C covers exactly its
// corresponding serialized section; the header CRC covers bytes [0, 32), and
// the CRC32C of an empty section is zero.
// Legacy v2 files with flags=0 and no checksum trailer remain readable.
// Writers emit checksummed v2 files. Readers reject unknown versions and v2
// flag bits; they also accept legacy v1 f32 snapshots. Legacy bit_width=8
// snapshots migrate to q8, while all other legacy widths migrate to f32.
//
// Delta log (.tvid version 2, all little-endian):
//
//   file header:
//     0   4   magic = "TVDL"
//     4   1   version = 2
//     5   1   bit_width (4, 8, or 32)
//     6   2   reserved (zero)
//     8   4   dim (uint32)
//     12  4   base snapshot state token
//
//   record header:
//     0   1   op (1 = add, 2 = remove)
//     1   3   reserved (zero)
//     4   4   n (add count; remove uses 1)
//     8   8   payload bytes
//     16  4   CRC32C over record header bytes [0, 16), state token, and payload
//     20  4   state token after applying this record
//
//   add payload:    N uint64 ids, then N*D float32 vectors
//   remove payload: one uint64 id
//
// The base snapshot token binds the log to the snapshot state it extends.
// Record state tokens let loading validate the final replay state and recognize
// a compacted snapshot when a process crashed before replacing the old delta
// log. Readers also accept legacy .tvid v1 logs, whose state field is a
// full-index CRC32C.

#ifdef __cplusplus
}
#endif
