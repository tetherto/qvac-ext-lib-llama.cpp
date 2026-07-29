#include "ggml-vector-index-avx2.h"

#include <cstddef>
#include <immintrin.h>

namespace ggml_vec_index_detail {
namespace {

inline float horizontal_sum(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

} // namespace

float dot_q8_avx2(const float * query, const int8_t * codes, float scale, int dim) {
    const __m256 scale_v = _mm256_set1_ps(scale);
    __m256 acc_v = _mm256_setzero_ps();

    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + i));
        const __m256 q = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)),
            scale_v);
        acc_v = _mm256_add_ps(acc_v, _mm256_mul_ps(_mm256_loadu_ps(query + i), q));
    }

    float acc = horizontal_sum(acc_v);
    for (; i < dim; ++i) {
        const float value = static_cast<float>(codes[i]) * scale;
        acc += query[i] * value;
    }
    return acc;
}

float dot_q4_avx2(const float * query, const uint8_t * codes, float scale, int dim) {
    const __m128i low_mask = _mm_set1_epi8(0x0f);
    const __m128i zero_point = _mm_set1_epi8(8);
    const __m256 scale_v = _mm256_set1_ps(scale);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m128i packed =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(codes + static_cast<size_t>(i) / 2));
        const __m128i low = _mm_and_si128(packed, low_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        const __m128i nibbles = _mm_unpacklo_epi8(low, high);
        const __m128i qbytes = _mm_sub_epi8(nibbles, zero_point);

        const __m256 q0 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qbytes)),
            scale_v);
        const __m256 q1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qbytes, 8))),
            scale_v);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(query + i), q0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(query + i + 8), q1));
    }

    float acc = horizontal_sum(acc0) + horizontal_sum(acc1);
    for (; i < dim; ++i) {
        const uint8_t byte = codes[static_cast<size_t>(i) / 2];
        const uint8_t nibble = (i & 1) == 0 ?
            static_cast<uint8_t>(byte & 0x0f) :
            static_cast<uint8_t>(byte >> 4);
        const float value = static_cast<float>(static_cast<int>(nibble) - 8) * scale;
        acc += query[i] * value;
    }
    return acc;
}

} // namespace ggml_vec_index_detail
