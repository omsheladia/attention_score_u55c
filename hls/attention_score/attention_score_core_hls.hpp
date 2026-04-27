#ifndef ATTENTION_SCORE_U55C_HLS_ATTENTION_SCORE_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_ATTENTION_SCORE_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace attention_score {

using hls_common::accum_int32_t;
using hls_common::act_int8_t;

void attention_score_core_hls(
    const act_int8_t q_tile[hls_common::kScoreRowsPerTile][hls_common::kHeadDim],
    const act_int8_t k_tile[hls_common::kScoreColsPerTile][hls_common::kHeadDim],
    accum_int32_t score_tile[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count);

void attention_score_u55c_kernel(
    const act_int8_t* q_tile,
    const act_int8_t* k_tile,
    accum_int32_t* score_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count);

}  // namespace attention_score
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_ATTENTION_SCORE_CORE_HLS_HPP_
