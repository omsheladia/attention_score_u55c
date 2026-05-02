#ifndef ATTENTION_SCORE_U55C_HLS_CAUSAL_MASK_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_CAUSAL_MASK_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace causal_mask {

using hls_common::accum_int32_t;

constexpr int kMaskNegInf = -1000000000;

void causal_mask_core_hls(
    const accum_int32_t score_in[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    accum_int32_t score_out[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count);

void causal_mask_u55c_kernel(
    const accum_int32_t* score_in,
    accum_int32_t* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count);

}  // namespace causal_mask
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_CAUSAL_MASK_CORE_HLS_HPP_
