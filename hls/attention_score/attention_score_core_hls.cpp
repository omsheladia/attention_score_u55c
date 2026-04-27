#include "attention_score_core_hls.hpp"

namespace attention_score_u55c {
namespace attention_score {

using namespace hls_common;

void attention_score_core_hls(
    const act_int8_t q_tile[kScoreRowsPerTile][kHeadDim],
    const act_int8_t k_tile[kScoreColsPerTile][kHeadDim],
    accum_int32_t score_tile[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      accum_int32_t accum = 0;

      if ((row < query_row_count) && (col < key_col_count)) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS UNROLL factor=8
          accum += static_cast<accum_int32_t>(q_tile[row][dim]) *
                   static_cast<accum_int32_t>(k_tile[col][dim]);
        }
      }

      score_tile[row][col] = accum;
    }
  }
}

void attention_score_u55c_kernel(
    const act_int8_t* q_tile,
    const act_int8_t* k_tile,
    accum_int32_t* score_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=q_tile offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=k_tile offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=score_tile offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=q_tile bundle=control
#pragma HLS INTERFACE s_axilite port=k_tile bundle=control
#pragma HLS INTERFACE s_axilite port=score_tile bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  act_int8_t q_local[kScoreRowsPerTile][kHeadDim];
  act_int8_t k_local[kScoreColsPerTile][kHeadDim];
  accum_int32_t score_local[kScoreRowsPerTile][kScoreColsPerTile];

#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=k_local cyclic factor=8 dim=2

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      q_local[row][dim] = q_tile[(row * kHeadDim) + dim];
    }
  }

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      k_local[col][dim] = k_tile[(col * kHeadDim) + dim];
    }
  }

  attention_score_core_hls(
      q_local,
      k_local,
      score_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_tile[(row * kScoreColsPerTile) + col] = score_local[row][col];
    }
  }
}

}  // namespace attention_score
}  // namespace attention_score_u55c
