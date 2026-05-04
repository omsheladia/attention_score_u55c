#ifndef ATTENTION_SCORE_U55C_HLS_SOFTMAX_FULL_ROW_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_SOFTMAX_FULL_ROW_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace softmax_full_row {

constexpr int kFullRowMaxCols = 512;

void softmax_full_row_core_hls(
    const float score_in[hls_common::kScoreRowsPerTile][kFullRowMaxCols],
    float prob_out[hls_common::kScoreRowsPerTile][kFullRowMaxCols],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count);

void softmax_full_row_u55c_kernel(
    const float* score_in,
    float* prob_out,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count);

void softmax_full_row_resident_u55c_kernel(
    const float* logits_in,
    float* prob_out,
    std::uint32_t seq_len,
    std::uint32_t query_pos_base,
    std::uint32_t query_row_count);

}  // namespace softmax_full_row
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_SOFTMAX_FULL_ROW_HLS_HPP_
