#pragma once

#include <cstdint>

namespace ggml_vec_index_detail {

float dot_q8_avx2(const float * query, const int8_t * codes, float scale, int dim);
float dot_q4_avx2(const float * query, const uint8_t * codes, float scale, int dim);

} // namespace ggml_vec_index_detail
