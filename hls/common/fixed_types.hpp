#ifndef ATTENTION_SCORE_U55C_HLS_COMMON_FIXED_TYPES_HPP_
#define ATTENTION_SCORE_U55C_HLS_COMMON_FIXED_TYPES_HPP_

#include <cstdint>
#include <type_traits>

#if __has_include(<ap_int.h>)
  #include <ap_int.h>
  #define ATTENTION_SCORE_U55C_HAS_AP_INT 1
#else
  #define ATTENTION_SCORE_U55C_HAS_AP_INT 0

template <int W>
using ap_int = std::conditional_t<(W <= 8), std::int8_t,
               std::conditional_t<(W <= 16), std::int16_t,
               std::conditional_t<(W <= 32), std::int32_t,
                                  std::int64_t>>>;

template <int W>
using ap_uint = std::conditional_t<(W <= 8), std::uint8_t,
                std::conditional_t<(W <= 16), std::uint16_t,
                std::conditional_t<(W <= 32), std::uint32_t,
                                   std::uint64_t>>>;
#endif

namespace attention_score_u55c {
namespace hls_common {

constexpr int kHeadDim = 64;
constexpr int kScoreRowsPerTile = 8;
constexpr int kScoreColsPerTile = 64;

using act_int8_t = ap_int<8>;
using accum_int32_t = ap_int<32>;
using word32_t = ap_uint<32>;
using word64_t = ap_uint<64>;

}  // namespace hls_common
}  // namespace attention_score_u55c

#endif  // ATTENTION_SCORE_U55C_HLS_COMMON_FIXED_TYPES_HPP_
