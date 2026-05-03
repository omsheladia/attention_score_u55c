#ifndef ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace v_weighted_sum {

// Compute one partial contribution: out += weights_tile @ v_tile
//
// weights_tile : (kScoreRowsPerTile x kScoreColsPerTile) float32
//                One K-chunk slice of full-row softmax probabilities.
// v_tile       : (kScoreColsPerTile x kHeadDim)           float32
//                V values for this K-chunk.
// out_tile     : (kScoreRowsPerTile x kHeadDim)           float32
//                Partial attn_out contribution; host accumulates across chunks.

void v_weighted_sum_core_hls(
    const float weights_tile[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    const float v_tile[hls_common::kScoreColsPerTile][hls_common::kHeadDim],
    float out_tile[hls_common::kScoreRowsPerTile][hls_common::kHeadDim],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count);

void v_weighted_sum_u55c_kernel(
    const float* weights_tile,
    const float* v_tile,
    float* out_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count);

}  // namespace v_weighted_sum
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_
