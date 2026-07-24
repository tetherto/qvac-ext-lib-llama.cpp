# Vector Index

`ggml-vector-index` is an opt-in C API for local vector search. It stores caller
provided ids with dense vectors, supports exact and approximate top-k search,
and can persist indexes to disk.

This candidate component is currently standalone. It is not enabled in default
builds and is not wired into the llama runtime, server, or app paths. Consumers
should enable it explicitly and link the vector-index target directly.

## Build

Enable the library with `GGML_VECTOR_INDEX`:

```sh
cmake -B build -DGGML_VECTOR_INDEX=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target ggml-vector-index test-vector-index bench-vector-index
```

Installed CMake packages export the target as `ggml::vector-index`.

```cmake
target_link_libraries(my_target PRIVATE ggml::vector-index)
```

NEON kernels are selected automatically on supported ARM builds. On x86,
AVX2 kernels are built separately when `GGML_NATIVE=ON` or `GGML_AVX2=ON` and
are selected at runtime. Scalar and SIMD reduction order can produce small
score differences across CPU architectures.

## Storage Modes

Create an index with a fixed dimension and bit width:

- `bit_width=32`: full f32 vectors.
- `bit_width=8`: per-vector symmetric q8 storage with f32 scales.
- `bit_width=4`: per-vector symmetric packed q4 storage with f32 scales.

`ggml_vec_index_create_turbovec_q2` and `ggml_vec_index_create_turbovec_q4`
create separate TurboQuant q2/q4 modes on 64-bit targets for positive
dimensions up to 65536 that are multiples of 8. They store Lloyd-Max q2/q4
codes in Rust-style bit-plane rows with one score-correction scale per vector.
Vectors and queries use a deterministic dense full-dimension Gaussian QR
rotation before LUT scoring. The rotation is materialized as a dense `dim x dim`
matrix on first use; very large accepted dimensions are format-valid, but
add/search can return
`GGML_VEC_INDEX_E_OOM` if the rotation state cannot be allocated.
`ggml_vec_index_prepare` is best-effort and does not report allocation status.
The first non-empty add fits TQ+ per-coordinate calibration when it contains at least
1000 vectors, then reuses that calibration for later adds. TurboVec snapshots
use `.tvim` v3 to persist the calibration. Regular snapshot write/load is
supported; mmap loading and logged mutations are reserved for a later format
update. Search also keeps a 32-vector blocked copy of the packed codes in memory
for NEON/AVX2 LUT scoring; this cache is updated after adds and rebuilt after
compaction and snapshot loading.

Search scores are dot products. The index does not normalize vectors internally.
For cosine similarity, normalize vectors before insertion and normalize queries
before search.

## Basic Usage

```c
#include "ggml-vector-index.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    const int dim = 4;
    const float vectors[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const uint64_t ids[] = { 101, 102 };
    const float query[] = { 0.8f, 0.2f, 0.0f, 0.0f };
    float scores[2];
    uint64_t out_ids[2];

    ggml_vec_index_t * idx = ggml_vec_index_create(dim, 32);
    if (idx == NULL) {
        return 1;
    }

    int rc = ggml_vec_index_add(idx, vectors, 2, ids);
    if (rc != GGML_VEC_INDEX_OK) {
        fprintf(stderr, "add failed: %s\n", ggml_vec_index_error_to_string(rc));
        ggml_vec_index_free(idx);
        return 1;
    }

    rc = ggml_vec_index_search(idx, query, 1, 2, scores, out_ids);
    if (rc == GGML_VEC_INDEX_OK) {
        printf("best id=%llu score=%f\n",
            (unsigned long long) out_ids[0],
            scores[0]);
    }

    ggml_vec_index_free(idx);
    return rc == GGML_VEC_INDEX_OK ? 0 : 1;
}
```

`ggml_vec_index_search` fills each result row in descending score order. If fewer
than `k` live entries are available, padded ids are set to `UINT64_MAX`.

## Search Modes

Exact search scans all live entries:

- `ggml_vec_index_search`
- `ggml_vec_index_search_filtered`
- `ggml_vec_index_filter_create`
- `ggml_vec_index_search_prepared_filtered`

IVF-flat search builds heap-owned approximate-nearest-neighbor state:

- `ggml_vec_index_build_ivf`
- `ggml_vec_index_search_ivf`

`ggml_vec_index_prepare` is an optional cache warmup for TurboVec q2/q4
rotation and codebook state; other storage modes ignore it. New callers should
use `ggml_vec_index_build_ivf` when approximate-search preparation is needed.

Call `ggml_vec_index_build_ivf` after loading an index and after successful
add/remove mutations. IVF state is not persisted in `.tvim` snapshots. Higher
`nprobe` values search more lists and generally improve recall at higher cost.
IVF uses the same dot-product score as exact search, assigning vectors and
queries to arithmetic centroids with dot-product scoring. Low `nprobe` values
are a recall/latency heuristic; probing all built lists gives exact-search
candidate coverage. For cosine-like IVF behavior, normalize vectors before
insertion and normalize queries before search.

## Persistence

Snapshots use the `.tvim` format. Delta logs use `.tvid`.

- `ggml_vec_index_write` writes a full snapshot.
- `ggml_vec_index_load` loads a snapshot into memory.
- `ggml_vec_index_add_logged` and `ggml_vec_index_remove_logged` apply mutations
  and append replayable delta records.
- `ggml_vec_index_load_with_delta` loads a snapshot and replays a delta log.
- `ggml_vec_index_compact_delta` writes a new snapshot and replaces the delta
  log with an empty matching log.

TurboVec `.tvim` snapshots are llama.cpp vector-index containers, not Rust
`turbovec` `.tv` or Rust `IdMapIndex` `.tvim` files. The formats share the
`TVPI` magic in current fixtures, but their headers and payload layout differ:
llama.cpp uses a 32-byte vector-index header with storage kind, qparam,
calibration-byte, id, and checksum sections, while Rust `TurboQuantIndex` `.tv`
uses its own compact header and has no external-id section. `ggml_vec_index_load`
intentionally rejects Rust TurboVec files; rebuild the llama.cpp index from
vectors when interchange is needed.

Delta logs are bound to the state of the snapshot they extend. Use one evolving
writer handle for a given snapshot and delta path pair. If another handle or
process writes to the same log, stale writers must reload from snapshot plus
delta before appending again. A bound handle rejects logged mutations or
compaction with a different delta path. Loading validates each replayed record
against its stored post-state identity.

After a handle has been loaded with a delta log or has used logged mutations,
content changes must continue through `ggml_vec_index_add_logged`,
`ggml_vec_index_remove_logged`, or `ggml_vec_index_compact_delta`. Plain
add/remove/compact calls and ordinary snapshot writes are rejected on
delta-bound handles.

Readers still accept legacy v1/v2 delta logs. New q4/q8 adds are not appended
to those f32-payload log formats; compact first so subsequent quantized adds use
native-code v4 records.

## mmap Loading

`ggml_vec_index_load_mmap` loads a v2 snapshot with the vector section mapped
read-only. Ids and quantization scales are copied into memory.

On mmap-backed handles:

- Search APIs are allowed.
- `ggml_vec_index_build_ivf` is allowed because it only builds heap-owned search
  state.
- Index-content mutations such as add, remove, compact, and logged mutations
  return `GGML_VEC_INDEX_E_INVALID_ARG`.
- `ggml_vec_index_write` is allowed only when writing to a path different from
  the mapped source file and the handle is not delta-bound.
- `ggml_vec_index_compact_delta` is allowed when writing the compacted snapshot
  to a path different from the mapped source file; it rebuilds the state identity
  before replacing the delta log.

The mmap loader is snapshot-only and does not replay `.tvid` delta logs. Use
`ggml_vec_index_load_with_delta` when delta replay is needed.

The persisted formats are little-endian. Regular load paths decode fields into
host values; mmap loading requires a little-endian host because vector bytes are
read directly from the mapped file.

## Threading

Read-only APIs on the same handle can run concurrently. Mutations, persistence
writes, compaction, and IVF builds are serialized with reads and with each
other. The caller must keep index and prepared-filter handles alive for the full
duration of every API call that uses them.

## Tests and Benchmark

Run the regression tests:

```sh
cmake --build build --target test-vector-index test-vector-index-faults
./build/bin/test-vector-index
./build/bin/test-vector-index-faults
```

Run the synthetic benchmark:

```sh
cmake --build build --target bench-vector-index
./build/bin/bench-vector-index
```

The benchmark reports q8/q4 quality against f32 exact search, exact and IVF
latency, mmap load timing, delta replay/compaction timing, and delete-heavy
behavior.
