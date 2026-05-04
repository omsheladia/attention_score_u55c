#ifndef ATTENTION_SCORE_U55C_HLS_SCORE_MASK_SCALE_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_SCORE_MASK_SCALE_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace score_mask_scale {

using hls_common::accum_int32_t;
using hls_common::act_int8_t;
using hls_common::word64_t;

constexpr int kMaskNegInf = -1000000000;

void score_mask_scale_core_hls(
    const act_int8_t q_tile[hls_common::kScoreRowsPerTile][hls_common::kHeadDim],
    const act_int8_t k_tile[hls_common::kScoreColsPerTile][hls_common::kHeadDim],
    float score_out[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale);

void score_mask_scale_u55c_kernel(
    const word64_t* q_tile,
    const word64_t* k_tile,
    float* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    float total_scale);

}  // namespace score_mask_scale
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_SCORE_MASK_SCALE_CORE_HLS_HPP_
