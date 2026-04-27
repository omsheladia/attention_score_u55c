#include "score_scale_core_hls.hpp"

namespace attention_score_u55c {
namespace score_scale {

using namespace hls_common;

void score_scale_core_hls(
    const accum_int32_t score_in[kScoreRowsPerTile][kScoreColsPerTile],
    float score_out[kScoreRowsPerTile][kScoreColsPerTile],
    float total_scale) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_out[row][col] = static_cast<float>(score_in[row][col]) * total_scale;
    }
  }
}

void score_scale_u55c_kernel(
    const accum_int32_t* score_in,
    float* score_out,
    float total_scale) {
#pragma HLS INTERFACE m_axi port=score_in offset=slave bundle=gmem0 depth=512
#pragma HLS INTERFACE m_axi port=score_out offset=slave bundle=gmem1 depth=512
#pragma HLS INTERFACE s_axilite port=score_in bundle=control
#pragma HLS INTERFACE s_axilite port=score_out bundle=control
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

  score_scale_core_hls(score_in_local, score_out_local, total_scale);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_out[(row * kScoreColsPerTile) + col] = score_out_local[row][col];
    }
  }
}

}  // namespace score_scale
}  // namespace attention_score_u55c
