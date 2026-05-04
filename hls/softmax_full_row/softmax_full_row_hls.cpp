#include "softmax_full_row_hls.hpp"

#include <cmath>

#if __has_include(<hls_math.h>)
  #include <hls_math.h>
#endif

namespace attention_score_u55c {
namespace softmax_full_row {

using namespace hls_common;

namespace {

inline float softmax_exp(float value) {
#if __has_include(<hls_math.h>)
  return hls::expf(value);
#else
  return std::exp(value);
#endif
}

inline std::uint16_t clamp_key_cols(std::uint16_t key_col_count) {
  return (key_col_count > kFullRowMaxCols)
             ? static_cast<std::uint16_t>(kFullRowMaxCols)
             : key_col_count;
}

}  // namespace

void softmax_full_row_core_hls(
    const float score_in[kScoreRowsPerTile][kFullRowMaxCols],
    float prob_out[kScoreRowsPerTile][kFullRowMaxCols],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  const std::uint16_t active_cols = clamp_key_cols(key_col_count);
  float exp_buf[kFullRowMaxCols];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    if ((row < query_row_count) && (active_cols > 0)) {
      float row_max = score_in[row][0];
      for (int col = 1; col < active_cols; ++col) {
#pragma HLS PIPELINE II=1
        if (score_in[row][col] > row_max) {
          row_max = score_in[row][col];
        }
      }

      float sum_exp = 0.0f;
      for (int col = 0; col < active_cols; ++col) {
#pragma HLS PIPELINE II=3
        exp_buf[col] = softmax_exp(score_in[row][col] - row_max);
        sum_exp += exp_buf[col];
      }

      for (int col = active_cols; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
        exp_buf[col] = 0.0f;
      }

      for (int col = 0; col < active_cols; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = exp_buf[col] / sum_exp;
      }

      for (int col = active_cols; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = 0.0f;
      }
    } else {
      for (int col = 0; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
        prob_out[row][col] = 0.0f;
      }
    }
  }
}

void softmax_full_row_u55c_kernel(
    const float* score_in,
    float* prob_out,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=score_in offset=slave bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi port=prob_out offset=slave bundle=gmem1 depth=4096
#pragma HLS INTERFACE s_axilite port=score_in bundle=control
#pragma HLS INTERFACE s_axilite port=prob_out bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float score_in_local[kScoreRowsPerTile][kFullRowMaxCols];
  float prob_out_local[kScoreRowsPerTile][kFullRowMaxCols];

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
      score_in_local[row][col] = score_in[(row * kFullRowMaxCols) + col];
    }
  }

  softmax_full_row_core_hls(
      score_in_local,
      prob_out_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
      prob_out[(row * kFullRowMaxCols) + col] = prob_out_local[row][col];
    }
  }
}

void softmax_full_row_resident_u55c_kernel(
    const float* logits_in,
    float* prob_out,
    std::uint32_t seq_len,
    std::uint32_t query_pos_base,
    std::uint32_t query_row_count) {
#pragma HLS INTERFACE m_axi port=logits_in offset=slave bundle=gmem0 depth=262144
#pragma HLS INTERFACE m_axi port=prob_out offset=slave bundle=gmem1 depth=262144
#pragma HLS INTERFACE s_axilite port=logits_in bundle=control
#pragma HLS INTERFACE s_axilite port=prob_out bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float score_in_local[kScoreRowsPerTile][kFullRowMaxCols];
  float prob_out_local[kScoreRowsPerTile][kFullRowMaxCols];

  const std::uint16_t seq_len_u16 = clamp_key_cols(static_cast<std::uint16_t>(seq_len));
  const std::uint16_t query_base_u16 = static_cast<std::uint16_t>(query_pos_base);
  const std::uint16_t query_rows_u16 = static_cast<std::uint16_t>(query_row_count);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
      float value = 0.0f;
      if ((row < query_rows_u16) && (col < seq_len_u16)) {
        const int global_row = static_cast<int>(query_base_u16) + row;
        value = logits_in[(global_row * static_cast<int>(seq_len_u16)) + col];
      }
      score_in_local[row][col] = value;
    }
  }

  softmax_full_row_core_hls(score_in_local, prob_out_local, query_rows_u16, seq_len_u16);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kFullRowMaxCols; ++col) {
#pragma HLS PIPELINE II=1
      if ((row < query_rows_u16) && (col < seq_len_u16)) {
        const int global_row = static_cast<int>(query_base_u16) + row;
        prob_out[(global_row * static_cast<int>(seq_len_u16)) + col] =
            prob_out_local[row][col];
      }
    }
  }
}

}  // namespace softmax_full_row
}  // namespace attention_score_u55c
