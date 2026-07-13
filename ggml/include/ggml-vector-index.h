#pragma once
//
// ggml-vector-index: TurboVec-style vector-index C API.
//
// This is the public C API for fabric's vector index. The implementation
// under `ggml/src/ggml-vector-index.cpp` supports full f32 storage
// (`bit_width=32`) and production q8 storage (`bit_width=8`) with CPU search
// directly against quantized codes. ARM builds use NEON when available; x86
// builds use AVX2 at runtime when supported.
//
// Threading: instances are NOT thread-safe. Callers must serialize access
// to a given handle. Multiple handles can be used concurrently.
//
// Endianness: persistence format is fixed little-endian.

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
// `dim` must be > 0. `bit_width` must be 8 or 32. `bit_width=8` stores
// per-vector symmetric q8 codes with one f32 scale per vector. `bit_width=32`
// stores full f32 vectors. Returns NULL on bad args.
GGML_API ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width);

GGML_API void ggml_vec_index_free(ggml_vec_index_t * idx);

// Mutation.
//
// Adds `n` vectors of length `dim` each (row-major, contiguous in `vectors`),
// associating each with the corresponding `ids[i]` (caller-owned external id).
// Returns 0 on success. Returns GGML_VEC_INDEX_E_DUPLICATE if any id already
// exists in the index; in that case the index is unchanged (atomic add).
// All vector components must be finite.
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

// Placeholder for cache warming / codebook resolution after a bulk add.
// Currently a no-op.
GGML_API void ggml_vec_index_prepare(ggml_vec_index_t * idx);

// Top-k search. `queries` is `n_q * dim` row-major. `out_scores` and
// `out_ids` are caller-allocated buffers of size `n_q * k`. Each row is
// sorted descending by score (higher = closer / more similar). If the index
// holds fewer than k entries, the remaining slots in each row are filled
// with sentinel values: -FLT_MAX for scores, UINT64_MAX for ids. Read-only
// against the index (does not mutate state).
//
// Score semantics: dot product. For f32 storage this is a full-precision dot
// product. For q8 storage, the query remains f32 and each indexed component is
// scored as `q8_code * per_vector_scale` without expanding the stored matrix
// back to f32. Callers that want cosine similarity must L2-normalize their
// vectors before insert AND before query; the index does NOT normalize
// internally. All query components must be finite.
GGML_API int ggml_vec_index_search(
    const ggml_vec_index_t * idx,
    const float            * queries,
    int                      n_q,
    int                      k,
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

// Stats.
GGML_API int ggml_vec_index_len(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_dim(const ggml_vec_index_t * idx);
GGML_API int ggml_vec_index_bit_width(const ggml_vec_index_t * idx);

// File format (.tvim version 2, all little-endian):
//
//   offset  size   field
//   ------  -----  -------------------------------------------------------
//   0       4      magic = "TVPI" (bytes 0x54, 0x56, 0x50, 0x49)
//   4       1      version = 2
//   5       1      bit_width (8 or 32)
//   6       1      storage kind (1 = f32, 2 = q8)
//   7       1      flags (bit 0 = checksum trailer present)
//   8       4      dim (uint32)
//   12      4      n_vectors (uint32)
//   16      4      qparam_type (0 = none, 1 = per-vector f32 scale)
//   20      4      qparam_bytes_per_vector (0 or 4)
//   24      4      bytes_per_component (1 or 4)
//   28      4      reserved (zero)
//   32      ...    qparams:
//                    - f32: empty
//                    - q8:  N float32 scales
//   ...     ...    vectors:
//                    - f32: N*D float32 values, row-major
//                    - q8:  N*D int8 codes, row-major
//   ...     N*8    ids (uint64)
//   ...     4      header CRC32C, when flag bit 0 is set
//   ...     4      qparams CRC32C, when flag bit 0 is set
//   ...     4      vectors CRC32C, when flag bit 0 is set
//   ...     4      ids CRC32C, when flag bit 0 is set
//
// Where N = n_vectors and D = dim. q8 uses symmetric per-vector quantization:
// scale = max(abs(v)) / 127, code = round(v / scale) clamped to [-127, 127].
// Zero vectors use scale = 1 and all-zero codes. Each CRC32C covers exactly its
// corresponding serialized section; the header CRC covers bytes [0, 32), and
// the CRC32C of an empty section is zero.
// Legacy v2 files with flags=0 and no checksum trailer remain readable.
// Writers emit checksummed v2 files. Readers reject unknown versions and v2
// flag bits; they also accept legacy v1 f32 snapshots. Legacy bit_width=8
// snapshots migrate to q8, while all other legacy widths migrate to f32.

#ifdef __cplusplus
}
#endif
