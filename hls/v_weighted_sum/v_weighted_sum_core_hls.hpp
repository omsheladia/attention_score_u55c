#ifndef ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace v_weighted_sum {

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

void v_weighted_sum_resident_u55c_kernel(
    const float* weights_full,
    const float* v_full,
    float* out_full,
    std::uint32_t seq_len,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    std::uint32_t clear_accum);

// O3: loops over all K/V chunks internally; keeps 8x64 accumulator on chip;
// writes final attn_out once per Q chunk — eliminates HBM read-modify-write.
void v_weighted_sum_multik_u55c_kernel(
    const float* probs_full,
    const float* v_full,
    float* attn_out,
    std::uint32_t seq_len,
    std::uint32_t query_base,
    std::uint32_t query_row_count);

}  // namespace v_weighted_sum
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_V_WEIGHTED_SUM_CORE_HLS_HPP_
