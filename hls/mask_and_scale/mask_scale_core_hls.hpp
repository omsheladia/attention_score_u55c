#ifndef ATTENTION_SCORE_U55C_HLS_MASK_SCALE_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_MASK_SCALE_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace mask_scale {

using hls_common::accum_int32_t;

constexpr int kMaskNegInf = -1000000000;

void mask_scale_core_hls(
    const accum_int32_t score_in[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    float score_out[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale);

void mask_scale_u55c_kernel(
    const accum_int32_t* score_in,
    float* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    float total_scale);

}  // namespace mask_scale
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_MASK_SCALE_CORE_HLS_HPP_
