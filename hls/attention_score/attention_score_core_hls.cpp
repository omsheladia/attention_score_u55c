#include "attention_score_core_hls.hpp"

namespace attention_score_u55c {
namespace attention_score {

using namespace hls_common;

namespace {

constexpr int kBytesPerWord = 8;
constexpr int kQPackedWords = (kScoreRowsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kKPackedWords = (kScoreColsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kScorePackedWords =
    (kScoreRowsPerTile * kScoreColsPerTile * static_cast<int>(sizeof(accum_int32_t))) /
    kBytesPerWord;

act_int8_t unpack_int8_lane(word64_t packed_word, int lane_idx) {
  const int shift = lane_idx * 8;
  const word64_t masked = (packed_word >> shift) & static_cast<word64_t>(0xff);
  return static_cast<act_int8_t>(static_cast<std::int8_t>(masked));
}

word64_t pack_score_pair(accum_int32_t first, accum_int32_t second) {
  const word64_t low = static_cast<word32_t>(first);
  const word64_t high = static_cast<word64_t>(static_cast<word32_t>(second)) << 32;
  return low | high;
}

}  // namespace

void attention_score_core_hls(
    const act_int8_t q_tile[kScoreRowsPerTile][kHeadDim],
    const act_int8_t k_tile[kScoreColsPerTile][kHeadDim],
    accum_int32_t score_tile[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      accum_int32_t accum = 0;

      if ((row < query_row_count) && (col < key_col_count)) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS UNROLL factor=16
          accum += static_cast<accum_int32_t>(q_tile[row][dim]) *
                   static_cast<accum_int32_t>(k_tile[col][dim]);
        }
      }

      score_tile[row][col] = accum;
    }
  }
}

void attention_score_u55c_kernel(
    const word64_t* q_tile,
    const word64_t* k_tile,
    word64_t* score_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=q_tile offset=slave bundle=gmem0 depth=64
#pragma HLS INTERFACE m_axi port=k_tile offset=slave bundle=gmem1 depth=512
#pragma HLS INTERFACE m_axi port=score_tile offset=slave bundle=gmem2 depth=256
#pragma HLS INTERFACE s_axilite port=q_tile bundle=control
#pragma HLS INTERFACE s_axilite port=k_tile bundle=control
#pragma HLS INTERFACE s_axilite port=score_tile bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  act_int8_t q_local[kScoreRowsPerTile][kHeadDim];
  act_int8_t k_local[kScoreColsPerTile][kHeadDim];
  accum_int32_t score_local[kScoreRowsPerTile][kScoreColsPerTile];

#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=k_local cyclic factor=16 dim=2

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int word_idx = 0; word_idx < (kHeadDim / kBytesPerWord); ++word_idx) {
#pragma HLS PIPELINE II=1
      const word64_t packed_word = q_tile[(row * (kHeadDim / kBytesPerWord)) + word_idx];
      for (int lane = 0; lane < kBytesPerWord; ++lane) {
#pragma HLS UNROLL
        q_local[row][(word_idx * kBytesPerWord) + lane] = unpack_int8_lane(packed_word, lane);
      }
    }
  }

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int word_idx = 0; word_idx < (kHeadDim / kBytesPerWord); ++word_idx) {
#pragma HLS PIPELINE II=1
      const word64_t packed_word = k_tile[(col * (kHeadDim / kBytesPerWord)) + word_idx];
      for (int lane = 0; lane < kBytesPerWord; ++lane) {
#pragma HLS UNROLL
        k_local[col][(word_idx * kBytesPerWord) + lane] = unpack_int8_lane(packed_word, lane);
      }
    }
  }

  attention_score_core_hls(
      q_local,
      k_local,
      score_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col_pair = 0; col_pair < (kScoreColsPerTile / 2); ++col_pair) {
#pragma HLS PIPELINE II=1
      const int col_base = col_pair * 2;
      score_tile[(row * (kScoreColsPerTile / 2)) + col_pair] =
          pack_score_pair(score_local[row][col_base], score_local[row][col_base + 1]);
    }
  }
}

}  // namespace attention_score
}  // namespace attention_score_u55c
