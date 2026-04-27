#ifndef ATTENTION_SCORE_U55C_HLS_SOFTMAX_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_SOFTMAX_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace softmax {

void softmax_core_hls(
    const float score_in[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    float prob_out[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count);

void softmax_u55c_kernel(
    const float* score_in,
    float* prob_out,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count);

}  // namespace softmax
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_SOFTMAX_CORE_HLS_HPP_
