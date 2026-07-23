# TurboVec Integration Plan

This plan treats TurboVec as a candidate downstream index layer for vectors
produced by `@qvac/embed-llamacpp`. The encoder remains unchanged. TurboVec
would replace or augment the current JS cosine + IVF index that consumes those
embeddings.

## Current Branch Scope

`ggml-vector-index` currently provides a C API for exact search, filtered search,
IVF-flat search, snapshot persistence, mmap loading, and delta logs. Its q4/q8
storage modes are generic per-vector max-absolute scalar quantizers. The
separate TurboVec q2/q4 modes are TurboQuant-specific scalar ports with regular
snapshot persistence and Rust top-k golden tests.

Do not add `bit_width=2` to the existing q4/q8 path and call it TurboVec. The
TurboVec q2/q4 constructors keep these quantizers distinct from the generic
storage modes.

## TurboVec Requirements

TurboVec 0.9 uses:

- bit widths in `{2, 3, 4}`
- dimensions that are positive multiples of 8
- L2 normalization before encoding
- deterministic orthogonal rotation
- dimension-dependent Lloyd-Max boundaries and centroids for the rotated
  unit-sphere coordinate distribution
- TQ+ per-coordinate shift and scale calibration
- bit-plane packed codes with `bytes_per_row = bit_width * (dim / 8)`
- per-vector score scales using the RaBitQ-style length correction
- query-side rotation, inverse calibration, LUT construction, and compressed
  scoring

## Implementation Route

Preferred investigation route:

1. Integrate the Rust `turbovec` crate as the reference implementation at the
   addon boundary that already owns `@qvac/embed-llamacpp` output.
2. Benchmark it against the existing JS cosine + IVF index using the same real
   embeddings, queries, and ids.
3. Capture recall, top-k overlap, latency, memory, persistence size, load time,
   and package/build cost.
4. Use Rust-generated golden files for encoded codes, scores, top-k results,
   and persistence round trips.
5. Decide whether to keep the Rust integration or port the TurboVec algorithm
   into this C++ library.

If a C++ port is required, implement it in this order:

1. Add a distinct TurboVec/TurboQuant storage kind rather than reusing q4/q8.
2. Port deterministic rotation and Lloyd-Max codebook generation or checked-in
   generated codebooks.
3. Port encoding without SIMD first: normalize, rotate, calibrate, quantize,
   bit-plane pack, and compute per-vector score scales.
4. Port scalar search: rotate query, apply inverse calibration, build LUTs, and
   score packed codes.
5. Add persistence with a new storage identifier/version and golden tests
   against Rust TurboVec files.
6. Add NEON and x86 SIMD scoring after scalar compatibility is verified.
7. Integrate or compare with IVF only after compressed full-scan quality and
   latency are understood.

## Non-Goals For The First Milestone

- Do not reinterpret existing `.tvim` q4/q8 snapshots as TurboVec snapshots.
- Do not add generic ternary q2 support as a substitute for TurboVec q2.
- Do not change `@qvac/embed-llamacpp` embedding generation.
- Do not make TurboVec default until benchmark results justify it.
