// ggml-vector-index-turbovec.cpp - TurboQuant q2/q4 index helpers.

#include "ggml-vector-index-internal.h"

#include "ggml-tbq-quants.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GGML_VEC_INDEX_TV_USE_NEON 1
#else
#define GGML_VEC_INDEX_TV_USE_NEON 0
#endif

namespace {

constexpr int kTurboVecQ4BlockDim = 128;
constexpr int kTurboVecQ4BlockBytes = kTurboVecQ4BlockDim / 2;
constexpr int kTurboVecQ2BlockBytes = kTurboVecQ4BlockDim / 4;
constexpr uint64_t kTurboVecQ4SignSeed = 42;

struct TurboVecQ4Signs {
    float values[kTurboVecQ4BlockDim];
};

struct TurboVecCodebook {
    std::vector<float> boundaries;
    std::vector<float> centroids;
};

static uint64_t rotl64_local(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct Xoshiro256 {
    uint64_t s[4];
};

static uint64_t rng_next(Xoshiro256 & rng) {
    const uint64_t result = rotl64_local(rng.s[1] * 5, 7) * 9;
    const uint64_t t = rng.s[1] << 17;
    rng.s[2] ^= rng.s[0];
    rng.s[3] ^= rng.s[1];
    rng.s[1] ^= rng.s[2];
    rng.s[0] ^= rng.s[3];
    rng.s[2] ^= t;
    rng.s[3] = rotl64_local(rng.s[3], 45);
    return result;
}

static void rng_seed(Xoshiro256 & rng, uint64_t seed) {
    for (uint64_t & state : rng.s) {
        seed += 0x9e3779b97f4a7c15ull;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        state = z ^ (z >> 31);
    }
}

static const TurboVecQ4Signs & turbovec_q4_signs() {
    static const TurboVecQ4Signs signs = []() {
        TurboVecQ4Signs out{};
        Xoshiro256 rng{};
        rng_seed(rng, kTurboVecQ4SignSeed);
        for (float & value : out.values) {
            value = (rng_next(rng) & 1) ? 1.0f : -1.0f;
        }
        return out;
    }();
    return signs;
}

static void rotate_block(float * block) {
    tq_forward_inplace(block, kTurboVecQ4BlockDim, turbovec_q4_signs().values);
}

static void inverse_rotate_block(float * block) {
    tq_inverse_inplace(block, kTurboVecQ4BlockDim, turbovec_q4_signs().values);
}

static float block_norm(const float * block) {
    double acc = 0.0;
    for (int i = 0; i < kTurboVecQ4BlockDim; ++i) {
        acc += static_cast<double>(block[i]) * static_cast<double>(block[i]);
    }
    return static_cast<float>(std::sqrt(acc));
}

static float vector_norm(const float * values, int dim) {
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        acc += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
    return static_cast<float>(std::sqrt(acc));
}

static float float_score_from_double_local(double score) {
    if (score > static_cast<double>(FLT_MAX)) {
        return FLT_MAX;
    }
    if (score < -static_cast<double>(FLT_MAX)) {
        return -FLT_MAX;
    }
    return static_cast<float>(score);
}

static double beta_cf(double a, double b, double x) {
    constexpr int max_iter = 200;
    constexpr double eps = 3.0e-14;
    constexpr double fpmin = std::numeric_limits<double>::min() / eps;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < fpmin) {
        d = fpmin;
    }
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= max_iter; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < eps) {
            break;
        }
    }

    return h;
}

static double regularized_beta(double x, double a, double b) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    const double bt = std::exp(
        std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
        a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * beta_cf(a, b, x) / a;
    }
    return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
}

static double beta_pdf_01(double x, double a) {
    if (x <= 0.0 || x >= 1.0) {
        return 0.0;
    }
    return std::exp(
        (a - 1.0) * (std::log(x) + std::log1p(-x)) -
        (std::lgamma(a) + std::lgamma(a) - std::lgamma(2.0 * a)));
}

template <typename Fn>
static double adaptive_simpson_rec(
        const Fn & f,
        double a,
        double b,
        double fa,
        double fb,
        double fm,
        double whole,
        double tol,
        int depth) {
    const double mid = (a + b) * 0.5;
    const double m1 = (a + mid) * 0.5;
    const double m2 = (mid + b) * 0.5;
    const double fm1 = f(m1);
    const double fm2 = f(m2);
    const double left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    const double right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    const double refined = left + right;
    if (depth == 0 || std::fabs(refined - whole) < 15.0 * tol) {
        return refined + (refined - whole) / 15.0;
    }
    return adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol * 0.5, depth - 1) +
        adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol * 0.5, depth - 1);
}

template <typename Fn>
static double adaptive_simpson(const Fn & f, double a, double b, double tol, int max_depth) {
    const double mid = (a + b) * 0.5;
    const double fa = f(a);
    const double fb = f(b);
    const double fm = f(mid);
    const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return adaptive_simpson_rec(f, a, b, fa, fb, fm, whole, tol, max_depth);
}

static TurboVecCodebook make_turbovec_codebook(int bits, int dim) {
    const double a = (static_cast<double>(dim) - 1.0) * 0.5;
    const int n_levels = 1 << bits;
    const double std_dev = std::sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a));
    const double spread = 3.0 * std_dev;
    std::vector<double> centroids(static_cast<size_t>(n_levels));
    for (int i = 0; i < n_levels; ++i) {
        centroids[static_cast<size_t>(i)] =
            -spread + 2.0 * spread * static_cast<double>(i) / static_cast<double>(n_levels - 1);
    }

    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> edges(static_cast<size_t>(n_levels) + 1);
        edges[0] = -1.0;
        for (int i = 0; i < n_levels - 1; ++i) {
            edges[static_cast<size_t>(i) + 1] =
                (centroids[static_cast<size_t>(i)] + centroids[static_cast<size_t>(i) + 1]) * 0.5;
        }
        edges[static_cast<size_t>(n_levels)] = 1.0;

        std::vector<double> next(static_cast<size_t>(n_levels), 0.0);
        for (int i = 0; i < n_levels; ++i) {
            const double lo = edges[static_cast<size_t>(i)];
            const double hi = edges[static_cast<size_t>(i) + 1];
            const double cdf_lo = regularized_beta((lo + 1.0) * 0.5, a, a);
            const double cdf_hi = regularized_beta((hi + 1.0) * 0.5, a, a);
            const double prob = cdf_hi - cdf_lo;
            if (prob < 1e-15) {
                next[static_cast<size_t>(i)] = centroids[static_cast<size_t>(i)];
            } else {
                const double mean = adaptive_simpson(
                    [a](double x) {
                        const double t = (x + 1.0) * 0.5;
                        return x * beta_pdf_01(t, a) * 0.5;
                    },
                    lo,
                    hi,
                    1e-14,
                    50);
                next[static_cast<size_t>(i)] = mean / prob;
            }
        }

        double max_change = 0.0;
        for (int i = 0; i < n_levels; ++i) {
            max_change = std::max(
                max_change,
                std::fabs(centroids[static_cast<size_t>(i)] - next[static_cast<size_t>(i)]));
        }
        centroids = std::move(next);
        if (max_change < 1e-12) {
            break;
        }
    }

    TurboVecCodebook out;
    out.boundaries.resize(static_cast<size_t>(n_levels - 1));
    out.centroids.resize(static_cast<size_t>(n_levels));
    for (int i = 0; i < n_levels - 1; ++i) {
        out.boundaries[static_cast<size_t>(i)] =
            static_cast<float>((centroids[static_cast<size_t>(i)] + centroids[static_cast<size_t>(i) + 1]) * 0.5);
    }
    for (int i = 0; i < n_levels; ++i) {
        out.centroids[static_cast<size_t>(i)] = static_cast<float>(centroids[static_cast<size_t>(i)]);
    }
    return out;
}

static const TurboVecCodebook & turbovec_codebook(int bits, int dim) {
    static std::mutex mutex;
    static std::unordered_map<int, TurboVecCodebook> cache;
    const int key = bits * 1000000 + dim;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, make_turbovec_codebook(bits, dim)).first;
    }
    return it->second;
}

static uint8_t turbovec_q2_quantize_val(float value, const float * boundaries) {
    uint8_t code = 0;
    for (int i = 0; i < 3; ++i) {
        if (value > boundaries[i]) {
            ++code;
        }
    }
    return code;
}

static uint8_t turbovec_get_bitplane_code(const uint8_t * row, int coord, int bits, int dim) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const size_t byte_pos = static_cast<size_t>(coord) / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (7 - (coord & 7)));
    uint8_t code = 0;
    for (int p = 0; p < bits; ++p) {
        if ((row[static_cast<size_t>(p) * bytes_per_plane + byte_pos] & mask) != 0) {
            code |= static_cast<uint8_t>(1u << p);
        }
    }
    return code;
}

static void turbovec_set_bitplane_code(uint8_t * row, int coord, int bits, int dim, uint8_t code) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const size_t byte_pos = static_cast<size_t>(coord) / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (7 - (coord & 7)));
    for (int p = 0; p < bits; ++p) {
        if ((code & (1u << p)) != 0) {
            row[static_cast<size_t>(p) * bytes_per_plane + byte_pos] |= mask;
        }
    }
}

static uint8_t turbovec_group_code_byte(const uint8_t * row, int group, int bits, int dim) {
    const int codes_per_byte = 8 / bits;
    const int dim_start = group * codes_per_byte;
    uint8_t byte = 0;
    for (int c = 0; c < codes_per_byte; ++c) {
        const uint8_t code = turbovec_get_bitplane_code(row, dim_start + c, bits, dim);
        const int shift = (codes_per_byte - 1 - c) * bits;
        byte |= static_cast<uint8_t>(code << shift);
    }
    return byte;
}

static void build_turbovec_query_lut(
        const float * rotated_query,
        const float * centroids,
        int bits,
        int dim,
        std::vector<uint8_t> & out_values,
        float & out_scale,
        float & out_bias) {
    const int codes_per_byte = 8 / bits;
    const int codes_per_nibble = codes_per_byte / 2;
    const int n_byte_groups = dim / codes_per_byte;
    const uint16_t code_mask = static_cast<uint16_t>((1u << bits) - 1u);
    constexpr float max_lut = 127.0f;

    out_values.assign(static_cast<size_t>(n_byte_groups) * 32, 0);
    out_scale = 1.0f;
    out_bias = 0.0f;
    std::vector<float> float_vals(static_cast<size_t>(n_byte_groups) * 32, 0.0f);
    std::vector<float> mins(static_cast<size_t>(n_byte_groups) * 2, 0.0f);
    float max_span = 0.0f;

    for (int g = 0; g < n_byte_groups; ++g) {
        const int dim_start = g * codes_per_byte;
        float lo_min = FLT_MAX;
        float lo_max = -FLT_MAX;
        for (uint16_t nibble = 0; nibble < 16; ++nibble) {
            float s = 0.0f;
            for (int c = 0; c < codes_per_nibble; ++c) {
                const int shift = (codes_per_nibble - 1 - c) * bits;
                const uint16_t code = static_cast<uint16_t>((nibble >> shift) & code_mask);
                s += rotated_query[static_cast<size_t>(dim_start + c)] * centroids[code];
            }
            float_vals[static_cast<size_t>(g) * 32 + nibble] = s;
            lo_min = std::min(lo_min, s);
            lo_max = std::max(lo_max, s);
        }

        float hi_min = FLT_MAX;
        float hi_max = -FLT_MAX;
        for (uint16_t nibble = 0; nibble < 16; ++nibble) {
            float s = 0.0f;
            for (int c = 0; c < codes_per_nibble; ++c) {
                const int shift = (codes_per_nibble - 1 - c) * bits;
                const uint16_t code = static_cast<uint16_t>((nibble >> shift) & code_mask);
                s += rotated_query[static_cast<size_t>(dim_start + codes_per_nibble + c)] * centroids[code];
            }
            float_vals[static_cast<size_t>(g) * 32 + 16 + nibble] = s;
            hi_min = std::min(hi_min, s);
            hi_max = std::max(hi_max, s);
        }

        mins[static_cast<size_t>(g) * 2] = lo_min;
        mins[static_cast<size_t>(g) * 2 + 1] = hi_min;
        out_bias += lo_min + hi_min;
        max_span = std::max(max_span, lo_max - lo_min);
        max_span = std::max(max_span, hi_max - hi_min);
    }

    out_scale = max_span > 1e-10f ? max_span / max_lut : 1.0f;
    const float inv_scale = 1.0f / out_scale;
    for (int g = 0; g < n_byte_groups; ++g) {
        const float lo_min = mins[static_cast<size_t>(g) * 2];
        const float hi_min = mins[static_cast<size_t>(g) * 2 + 1];
        for (int i = 0; i < 16; ++i) {
            const size_t lo = static_cast<size_t>(g) * 32 + static_cast<size_t>(i);
            const size_t hi = static_cast<size_t>(g) * 32 + 16 + static_cast<size_t>(i);
            out_values[lo] = static_cast<uint8_t>(
                std::max(0.0f, std::min(max_lut, std::round((float_vals[lo] - lo_min) * inv_scale))));
            out_values[hi] = static_cast<uint8_t>(
                std::max(0.0f, std::min(max_lut, std::round((float_vals[hi] - hi_min) * inv_scale))));
        }
    }
}

static float dot_turbovec_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int bits,
        int dim) {
    const int codes_per_byte = 8 / bits;
    const int n_byte_groups = dim / codes_per_byte;
    double score = static_cast<double>(lut_bias);
    for (int g = 0; g < n_byte_groups; ++g) {
        const uint8_t byte = turbovec_group_code_byte(codes, g, bits, dim);
        const uint8_t hi = static_cast<uint8_t>(byte >> 4);
        const uint8_t lo = static_cast<uint8_t>(byte & 0x0f);
        score += static_cast<double>(lut_scale) *
            static_cast<double>(lut[static_cast<size_t>(g) * 32 + hi]);
        score += static_cast<double>(lut_scale) *
            static_cast<double>(lut[static_cast<size_t>(g) * 32 + 16 + lo]);
    }
    score *= static_cast<double>(scales[0]);
    return float_score_from_double_local(score);
}

#if GGML_VEC_INDEX_TV_USE_NEON
static float horizontal_sum_local(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    const float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
#endif
}
#endif

} // namespace

bool turbovec_q2_supported_dim(int dim) {
    return dim > 0 && dim % kTurboVecQ4BlockDim == 0;
}

bool turbovec_q4_supported_dim(int dim) {
    return dim > 0 && dim % kTurboVecQ4BlockDim == 0;
}

size_t turbovec_q2_row_bytes(size_t dim) {
    return (dim / static_cast<size_t>(kTurboVecQ4BlockDim)) *
        static_cast<size_t>(kTurboVecQ2BlockBytes);
}

size_t turbovec_q2_scale_count(size_t dim) {
    GGML_UNUSED(dim);
    return 1;
}

size_t turbovec_q4_row_bytes(size_t dim) {
    return (dim / static_cast<size_t>(kTurboVecQ4BlockDim)) *
        static_cast<size_t>(kTurboVecQ4BlockBytes);
}

size_t turbovec_q4_scale_count(size_t dim) {
    GGML_UNUSED(dim);
    return 1;
}

void rotate_turbovec_query(const float * src, float * dst, int dim) {
    const size_t n_blocks = static_cast<size_t>(dim) / static_cast<size_t>(kTurboVecQ4BlockDim);
    for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        float * dst_block = dst + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim);
        std::memcpy(
            dst_block,
            src + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim),
            static_cast<size_t>(kTurboVecQ4BlockDim) * sizeof(float));
        rotate_block(dst_block);
    }
}

void quantize_turbovec_q2_row(const float * src, uint8_t * dst, float * scales, int dim) {
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim);
    const float * boundaries = codebook.boundaries.data();
    const float * centroids = codebook.centroids.data();

    std::memset(dst, 0, turbovec_q2_row_bytes(static_cast<size_t>(dim)));
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(src, rotated.data(), dim);
    const float norm = vector_norm(rotated.data(), dim);
    if (norm < 1e-15f) {
        scales[0] = 0.0f;
        return;
    }
    const float inv_norm = 1.0f / norm;
    double inner = 0.0;
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_q2_quantize_val(rotated[static_cast<size_t>(i)] * inv_norm, boundaries);
        turbovec_set_bitplane_code(dst, i, 2, dim, code);
        inner += static_cast<double>(rotated[static_cast<size_t>(i)] * inv_norm) *
            static_cast<double>(centroids[code]);
    }
    scales[0] = norm / std::max(static_cast<float>(inner), 1e-10f);
}

void decode_turbovec_q2_row(const uint8_t * codes, const float * scales, float * dst, int dim) {
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim);
    const float * centroids = codebook.centroids.data();
    std::vector<float> rotated(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_get_bitplane_code(codes, i, 2, dim);
        rotated[static_cast<size_t>(i)] = centroids[code] * scales[0];
    }
    const size_t n_blocks = static_cast<size_t>(dim) / static_cast<size_t>(kTurboVecQ4BlockDim);
    for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        float * block = rotated.data() + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim);
        inverse_rotate_block(block);
        std::memcpy(
            dst + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim),
            block,
            static_cast<size_t>(kTurboVecQ4BlockDim) * sizeof(float));
    }
}

float dot_turbovec_q2_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<uint8_t> lut;
    float lut_scale = 1.0f;
    float lut_bias = 0.0f;
    build_turbovec_q2_lut(rotated_query, dim, lut, lut_scale, lut_bias);
    return dot_turbovec_q2_lut_row(lut.data(), lut_scale, lut_bias, codes, scales, dim);
}

float dot_turbovec_q2_row(const float * query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(query, rotated.data(), dim);
    return dot_turbovec_q2_rotated_row(rotated.data(), codes, scales, dim);
}

void quantize_turbovec_q4_row(const float * src, uint8_t * dst, float * scales, int dim) {
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim);
    const float * boundaries = codebook.boundaries.data();
    const float * centroids = codebook.centroids.data();

    std::memset(dst, 0, turbovec_q4_row_bytes(static_cast<size_t>(dim)));
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(src, rotated.data(), dim);
    const float norm = vector_norm(rotated.data(), dim);
    if (norm < 1e-15f) {
        scales[0] = 0.0f;
        return;
    }
    const float inv_norm = 1.0f / norm;
    double inner = 0.0;
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = tq4_quantize_val(rotated[static_cast<size_t>(i)] * inv_norm, boundaries);
        turbovec_set_bitplane_code(dst, i, 4, dim, code);
        inner += static_cast<double>(rotated[static_cast<size_t>(i)] * inv_norm) *
            static_cast<double>(centroids[code]);
    }
    scales[0] = norm / std::max(static_cast<float>(inner), 1e-10f);
}

void build_turbovec_q2_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim);
    build_turbovec_query_lut(rotated_query, codebook.centroids.data(), 2, dim, lut, scale, bias);
}

float dot_turbovec_q2_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int dim) {
    return dot_turbovec_lut_row(lut, lut_scale, lut_bias, codes, scales, 2, dim);
}

void decode_turbovec_q4_row(const uint8_t * codes, const float * scales, float * dst, int dim) {
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim);
    const float * centroids = codebook.centroids.data();
    std::vector<float> rotated(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_get_bitplane_code(codes, i, 4, dim);
        rotated[static_cast<size_t>(i)] = centroids[code] * scales[0];
    }
    const size_t n_blocks = static_cast<size_t>(dim) / static_cast<size_t>(kTurboVecQ4BlockDim);
    for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        float * block = rotated.data() + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim);
        inverse_rotate_block(block);
        std::memcpy(
            dst + block_idx * static_cast<size_t>(kTurboVecQ4BlockDim),
            block,
            static_cast<size_t>(kTurboVecQ4BlockDim) * sizeof(float));
    }
}

void build_turbovec_q4_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim);
    build_turbovec_query_lut(rotated_query, codebook.centroids.data(), 4, dim, lut, scale, bias);
}

float dot_turbovec_q4_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int dim) {
    return dot_turbovec_lut_row(lut, lut_scale, lut_bias, codes, scales, 4, dim);
}

float dot_turbovec_q4_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<uint8_t> lut;
    float lut_scale = 1.0f;
    float lut_bias = 0.0f;
    build_turbovec_q4_lut(rotated_query, dim, lut, lut_scale, lut_bias);
    return dot_turbovec_q4_lut_row(lut.data(), lut_scale, lut_bias, codes, scales, dim);
}

float dot_turbovec_q4_row(const float * query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(query, rotated.data(), dim);
    return dot_turbovec_q4_rotated_row(rotated.data(), codes, scales, dim);
}
