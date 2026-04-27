#include "causal_mask_core_hls.hpp"

namespace attention_score_u55c {
namespace causal_mask {

using namespace hls_common;

void causal_mask_core_hls(
    const accum_int32_t score_in[kScoreRowsPerTile][kScoreColsPerTile],
    accum_int32_t score_out[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      accum_int32_t out_value = static_cast<accum_int32_t>(kMaskNegInf);

      if ((row < query_row_count) && (col < key_col_count)) {
        const int query_pos = static_cast<int>(query_pos_base) + row;
        const int key_pos = static_cast<int>(key_pos_base) + col;
        if (key_pos <= query_pos) {
          out_value = score_in[row][col];
        }
      }

      score_out[row][col] = out_value;
    }
  }
}

void causal_mask_u55c_kernel(
    const accum_int32_t* score_in,
    accum_int32_t* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=score_in offset=slave bundle=gmem0 depth=512
#pragma HLS INTERFACE m_axi port=score_out offset=slave bundle=gmem1 depth=512
#pragma HLS INTERFACE s_axilite port=score_in bundle=control
#pragma HLS INTERFACE s_axilite port=score_out bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  accum_int32_t score_in_local[kScoreRowsPerTile][kScoreColsPerTile];
  accum_int32_t score_out_local[kScoreRowsPerTile][kScoreColsPerTile];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_in_local[row][col] = score_in[(row * kScoreColsPerTile) + col];
    }
  }

  causal_mask_core_hls(
      score_in_local,
      score_out_local,
      static_cast<std::uint16_t>(query_pos_base),
      static_cast<std::uint16_t>(key_pos_base),
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_out[(row * kScoreColsPerTile) + col] = score_out_local[row][col];
    }
  }
}

}  // namespace causal_mask
}  // namespace attention_score_u55c
