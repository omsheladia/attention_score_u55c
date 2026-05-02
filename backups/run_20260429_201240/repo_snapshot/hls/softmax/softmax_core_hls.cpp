#include "softmax_core_hls.hpp"

#include <cmath>

#if __has_include(<hls_math.h>)
  #include <hls_math.h>
#endif

namespace attention_score_u55c {
namespace softmax {

using namespace hls_common;

namespace {

inline float softmax_exp(float value) {
#if __has_include(<hls_math.h>)
  return hls::expf(value);
#else
  return std::exp(value);
#endif
}

}  // namespace

void softmax_core_hls(
    const float score_in[kScoreRowsPerTile][kScoreColsPerTile],
    float prob_out[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  float exp_buf[kScoreColsPerTile];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    float row_max = 0.0f;
    if ((row < query_row_count) && (key_col_count > 0)) {
      row_max = score_in[row][0];
      for (int col = 1; col < key_col_count; ++col) {
        if (score_in[row][col] > row_max) {
          row_max = score_in[row][col];
        }
      }

      float sum_exp = 0.0f;
      for (int col = 0; col < key_col_count; ++col) {
#pragma HLS PIPELINE II=3
        exp_buf[col] = softmax_exp(score_in[row][col] - row_max);
        sum_exp += exp_buf[col];
      }

      for (int col = key_col_count; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
        exp_buf[col] = 0.0f;
      }

      for (int col = 0; col < key_col_count; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = exp_buf[col] / sum_exp;
      }

      for (int col = key_col_count; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = 0.0f;
      }
    } else {
      for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = 0.0f;
      }
    }
  }
}

void softmax_u55c_kernel(
    const float* score_in,
    float* prob_out,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=score_in offset=slave bundle=gmem0 depth=512
#pragma HLS INTERFACE m_axi port=prob_out offset=slave bundle=gmem1 depth=512
#pragma HLS INTERFACE s_axilite port=score_in bundle=control
#pragma HLS INTERFACE s_axilite port=prob_out bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float score_in_local[kScoreRowsPerTile][kScoreColsPerTile];
  float prob_out_local[kScoreRowsPerTile][kScoreColsPerTile];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_in_local[row][col] = score_in[(row * kScoreColsPerTile) + col];
    }
  }

  softmax_core_hls(
      score_in_local,
      prob_out_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      prob_out[(row * kScoreColsPerTile) + col] = prob_out_local[row][col];
    }
  }
}

}  // namespace softmax
}  // namespace attention_score_u55c
