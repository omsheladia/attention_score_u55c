#include "mask_scale_core_hls.hpp"

namespace attention_score_u55c {
namespace mask_scale {

using namespace hls_common;

void mask_scale_core_hls(
    const accum_int32_t score_in[kScoreRowsPerTile][kScoreColsPerTile],
    float score_out[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      const bool masked =
          (row >= query_row_count) ||
          (col >= key_col_count) ||
          ((static_cast<int>(key_pos_base) + col) > (static_cast<int>(query_pos_base) + row));

      const accum_int32_t masked_or_raw =
          masked ? static_cast<accum_int32_t>(kMaskNegInf) : score_in[row][col];
      score_out[row][col] = static_cast<float>(masked_or_raw) * total_scale;
    }
  }
}

void mask_scale_u55c_kernel(
    const accum_int32_t* score_in,
    float* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    float total_scale) {
#pragma HLS INTERFACE m_axi port=score_in offset=slave bundle=gmem0 depth=512
#pragma HLS INTERFACE m_axi port=score_out offset=slave bundle=gmem1 depth=512
#pragma HLS INTERFACE s_axilite port=score_in bundle=control
#pragma HLS INTERFACE s_axilite port=score_out bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=total_scale bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  accum_int32_t score_in_local[kScoreRowsPerTile][kScoreColsPerTile];
  float score_out_local[kScoreRowsPerTile][kScoreColsPerTile];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_in_local[row][col] = score_in[(row * kScoreColsPerTile) + col];
    }
  }

  mask_scale_core_hls(
      score_in_local,
      score_out_local,
      static_cast<std::uint16_t>(query_pos_base),
      static_cast<std::uint16_t>(key_pos_base),
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count),
      total_scale);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_out[(row * kScoreColsPerTile) + col] = score_out_local[row][col];
    }
  }
}

}  // namespace mask_scale
}  // namespace attention_score_u55c
