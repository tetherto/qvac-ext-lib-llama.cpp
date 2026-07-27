// ggml-vector-index.cpp - POC scalar implementation of the fabric vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors as a contiguous std::vector<float>. ID map uses
// std::unordered_map<uint64_t, size_t> for lookup and a parallel vector for
// the slot->id reverse map. Remove uses swap-with-last.
//
// Search: naive scalar dot product across all slots + min-heap of size k.
// No SIMD, no GPU. Correctness over speed; the optimization phase will swap
// the storage layout and kernel without touching the C API.

#include "ggml-vector-index.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
constexpr uint8_t  kTvimVersion    = 1;
constexpr size_t   kTvimHeaderSize = 16;

static_assert(sizeof(float) == sizeof(uint32_t), "ggml-vector-index requires float32");

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

bool write_u32_le(std::ofstream & f, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    f.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    return static_cast<bool>(f);
}

bool write_u64_le(std::ofstream & f, uint64_t v) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    f.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    return static_cast<bool>(f);
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

} // namespace

// Lifetime-managed instance state. Lives behind the opaque
// `ggml_vec_index_t` typedef.
struct ggml_vec_index {
    int dim       = 0;
    int bit_width = 32;

    // Flat row-major storage: `data[slot * dim + i]` is component i of vec slot.
    std::vector<float> data;

    // slot -> external id (parallel to logical slot index).
    std::vector<uint64_t> slot_to_id;

    // external id -> slot.
    std::unordered_map<uint64_t, size_t> id_to_slot;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width) {
    try {
        if (dim <= 0) {
            return nullptr;
        }
        if (bit_width <= 0 || bit_width > 32) {
            // POC ignores bit_width but still validates the range so callers
            // surface bad config early.
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
        for (int i = 0; i < n; ++i) {
            if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
            for (int j = i + 1; j < n; ++j) {
                if (ids[i] == ids[j]) {
                    return GGML_VEC_INDEX_E_DUPLICATE;
                }
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

        idx->data.resize(new_slots * dim_sz);
        resized = true;
        idx->slot_to_id.resize(new_slots);
        idx->id_to_slot.reserve(new_slots);

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            std::memcpy(
                idx->data.data() + slot * dim_sz,
                vectors + static_cast<size_t>(i) * dim_sz,
                dim_sz * sizeof(float));
            idx->slot_to_id[slot] = ids[i];
            idx->id_to_slot.emplace(ids[i], slot);
        }
    } catch (const std::bad_alloc &) {
        if (idx != nullptr && resized) {
            for (int i = 0; i < n; ++i) {
                idx->id_to_slot.erase(ids[i]);
            }
            idx->data.resize(base_slot * dim_sz);
            idx->slot_to_id.resize(base_slot);
        }
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        if (idx != nullptr && resized) {
            for (int i = 0; i < n; ++i) {
                idx->id_to_slot.erase(ids[i]);
            }
            idx->data.resize(base_slot * dim_sz);
            idx->slot_to_id.resize(base_slot);
        }
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
            std::memcpy(
                idx->data.data() + slot * dim_sz,
                idx->data.data() + last * dim_sz,
                dim_sz * sizeof(float));
            const uint64_t moved_id = idx->slot_to_id[last];
            idx->slot_to_id[slot]   = moved_id;
            idx->id_to_slot[moved_id] = slot;
        }

        idx->slot_to_id.pop_back();
        idx->data.resize(last * dim_sz);
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
    // POC no-op. Future: warm caches, materialize codebooks, etc.
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

    std::priority_queue<ScoreId, std::vector<ScoreId>, MinHeapCmp> heap;

    for (size_t slot = 0; slot < n_slots; ++slot) {
        const float s = dot(
            query, idx.data.data() + slot * static_cast<size_t>(dim), dim);
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
        if (idx->data.size() != n * dim_sz) {
            return GGML_VEC_INDEX_E_INTERNAL;
        }

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            return GGML_VEC_INDEX_E_IO;
        }

        // Header: 16 bytes. Layout matches the comment block in the header file.
        uint8_t header[kTvimHeaderSize] = {};
        std::memcpy(header, kTvimMagic, 4);
        header[4] = kTvimVersion;
        header[5] = static_cast<uint8_t>(idx->bit_width);
        header[6] = 0;
        header[7] = 0;
        const uint32_t dim_le = static_cast<uint32_t>(idx->dim);
        const uint32_t n_le   = static_cast<uint32_t>(idx->slot_to_id.size());
        put_u32_le(header + 8, dim_le);
        put_u32_le(header + 12, n_le);

        f.write(reinterpret_cast<const char *>(header), sizeof(header));
        if (!f) { return GGML_VEC_INDEX_E_IO; }

        for (float v : idx->data) {
            if (!write_u32_le(f, float_to_u32(v))) {
                return GGML_VEC_INDEX_E_IO;
            }
        }

        for (uint64_t id : idx->slot_to_id) {
            if (!write_u64_le(f, id)) {
                return GGML_VEC_INDEX_E_IO;
            }
        }

        f.flush();
        if (!f) { return GGML_VEC_INDEX_E_IO; }
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
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            return nullptr;
        }

        uint8_t header[kTvimHeaderSize] = {};
        f.read(reinterpret_cast<char *>(header), sizeof(header));
        if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(header))) {
            return nullptr;
        }
        if (std::memcmp(header, kTvimMagic, 4) != 0) {
            return nullptr;
        }
        if (header[4] != kTvimVersion) {
            return nullptr;
        }

        const int bit_width = static_cast<int>(header[5]);
        const uint32_t dim_le = get_u32_le(header + 8);
        const uint32_t n_le   = get_u32_le(header + 12);
        if (dim_le == 0 || dim_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
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

        idx->data.resize(n * dim_sz);
        idx->slot_to_id.resize(n);
        idx->id_to_slot.reserve(n);

        for (float & v : idx->data) {
            uint32_t bits = 0;
            if (!read_u32_le(f, bits)) {
                return nullptr;
            }
            v = u32_to_float(bits);
        }

        for (uint64_t & id : idx->slot_to_id) {
            if (!read_u64_le(f, id)) {
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
