#pragma once
//
// ggml-vector-index: TurboQuant-style ANN vector-index C API.
//
// POC NOTE
// --------
// This is the public C API for fabric's vector index. The implementation
// under `ggml/src/ggml-vector-index.cpp` is intentionally naive (full f32
// storage, scalar dot-product, min-heap top-k). The C API itself is final:
// downstream Bare addon bindings + JS wrappers depend on this signature and
// the on-disk file format documented at the bottom of this header.
// Quantization / SIMD / GPU kernels are future work and will be swapped in
// behind the same API without breaking callers.
//
// Threading: instances are NOT thread-safe. Callers must serialize access
// to a given handle. Multiple handles can be used concurrently.
//
// Endianness: persistence format is fixed little-endian. The POC asserts a
// LE host; we do not currently support big-endian platforms.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a vector index instance.
struct ggml_vec_index;
typedef struct ggml_vec_index ggml_vec_index_t;

// Error codes returned from int-valued APIs. 0 = OK. Negative = failure.
// `_remove` is the exception: it returns 1 on removal and 0 on miss.
enum ggml_vec_index_error {
    GGML_VEC_INDEX_OK            =  0,
    GGML_VEC_INDEX_E_INVALID_DIM = -1,
    GGML_VEC_INDEX_E_INVALID_ARG = -2,
    GGML_VEC_INDEX_E_DUPLICATE   = -3,
    GGML_VEC_INDEX_E_IO          = -4,
    GGML_VEC_INDEX_E_BAD_MAGIC   = -5,
    GGML_VEC_INDEX_E_BAD_VERSION = -6,
    GGML_VEC_INDEX_E_OOM         = -7,
    GGML_VEC_INDEX_E_INTERNAL    = -99,
};

// Lifecycle.
//
// `dim` must be > 0. `bit_width` is reserved for future quantization; the
// POC accepts any value in [1, 32] but ignores it (storage is always full
// f32). Returns NULL on bad args.
GGML_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
GGML_API int ggml_vec_index_add(
    ggml_vec_index_t * idx,
    const float      * vectors,
    int                n,
    const uint64_t   * ids);

// Removes the entry for `id` via swap-with-last (slot indices are NOT stable
// across removes; external ids ARE). Returns 1 if removed, 0 if not present,
// negative on error.
GGML_API int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id);

// Returns 1 if the id is in the index, 0 otherwise. Read-only.
GGML_API int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id);

// No-op for the POC. Placeholder for future cache warming / codebook
// resolution after a bulk add. Reserved as a mutating op (warm-up may
// materialize derived state inside the index).
GGML_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar). If the index
// holds fewer than k entries, the remaining slots in each row are filled
// with sentinel values: -FLT_MAX for scores, UINT64_MAX for ids. Read-only
// against the index (does not mutate state).
//
// Score semantics: scalar full-precision dot product. Callers that want
// cosine similarity must L2-normalize their vectors before insert AND
// before query; the index does NOT normalize internally.
GGML_API int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
    float                  * out_scores,
    uint64_t               * out_ids);

// Persistence. Format is .tvim version 1; see bottom of this header.
GGML_API int ggml_vec_index_write(
    ggml_vec_index_t * idx,
    const char       * path);

// Returns NULL on failure (caller can inspect errno for I/O specifics).
GGML_API ggml_vec_index_t * ggml_vec_index_load(const char * path);

// Stats.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim version 1, all little-endian):
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version = 1
//   5       1      bit_width
//   6       2      reserved (zero)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      N*D*4  vectors (float32, row-major)
//   ...     N*8    ids (uint64)
//
// Where N = n_vectors and D = dim. There is no checksum in v1; future
// versions may append one without breaking back-compat (version-gated).

#ifdef __cplusplus
}
#endif
