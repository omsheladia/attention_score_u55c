#ifndef ATTENTION_SCORE_U55C_HLS_SCORE_SCALE_CORE_HLS_HPP_
#define ATTENTION_SCORE_U55C_HLS_SCORE_SCALE_CORE_HLS_HPP_

#include <cstdint>

#include "../common/fixed_types.hpp"

namespace attention_score_u55c {
namespace score_scale {

using hls_common::accum_int32_t;

void score_scale_core_hls(
    const accum_int32_t score_in[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    float score_out[hls_common::kScoreRowsPerTile][hls_common::kScoreColsPerTile],
    float total_scale);

void score_scale_u55c_kernel(
    const accum_int32_t* score_in,
    float* score_out,
    float total_scale);

}  // namespace score_scale
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_SCORE_SCALE_CORE_HLS_HPP_
