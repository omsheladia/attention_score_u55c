#include "v_weighted_sum_core_hls.hpp"

namespace attention_score_u55c {
namespace v_weighted_sum {

using hls_common::kHeadDim;
using hls_common::kScoreColsPerTile;
using hls_common::kScoreRowsPerTile;

void v_weighted_sum_core_hls(
    const float weights_tile[kScoreRowsPerTile][kScoreColsPerTile],
    const float v_tile[kScoreColsPerTile][kHeadDim],
    float out_tile[kScoreRowsPerTile][kHeadDim],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weights_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=v_tile cyclic factor=16 dim=1

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      float accum = 0.0f;
      for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS UNROLL factor=16
        if (row < query_row_count && col < key_col_count) {
          accum += weights_tile[row][col] * v_tile[col][dim];
        }
      }
      out_tile[row][dim] = (row < query_row_count) ? accum : 0.0f;
    }
  }
}

void v_weighted_sum_u55c_kernel(
    const float* weights_tile,
    const float* v_tile,
    float* out_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=weights_tile offset=slave bundle=gmem_weights depth=512
#pragma HLS INTERFACE m_axi port=v_tile offset=slave bundle=gmem_v depth=4096
#pragma HLS INTERFACE m_axi port=out_tile offset=slave bundle=gmem_out depth=512
#pragma HLS INTERFACE s_axilite port=weights_tile bundle=control
#pragma HLS INTERFACE s_axilite port=v_tile bundle=control
#pragma HLS INTERFACE s_axilite port=out_tile bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float weights_local[kScoreRowsPerTile][kScoreColsPerTile];
  float v_local[kScoreColsPerTile][kHeadDim];
  float out_local[kScoreRowsPerTile][kHeadDim];
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=v_local cyclic factor=16 dim=1

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      weights_local[row][col] = weights_tile[(row * kScoreColsPerTile) + col];
    }
  }

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      v_local[col][dim] = v_tile[(col * kHeadDim) + dim];
    }
  }

  v_weighted_sum_core_hls(
      weights_local,
      v_local,
      out_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      out_tile[(row * kHeadDim) + dim] = out_local[row][dim];
    }
  }
}

}  // namespace v_weighted_sum
}  // namespace attention_score_u55c
