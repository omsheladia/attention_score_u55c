#include "v_weighted_sum_core_hls.hpp"

namespace attention_score_u55c {
namespace v_weighted_sum {

using hls_common::kHeadDim;
using hls_common::kScoreColsPerTile;
using hls_common::kScoreRowsPerTile;

namespace {
constexpr int kMultikColChunk = 8;
constexpr int kMultikRowChunk = 4;
}

void v_weighted_sum_core_hls(
    const float weights_tile[kScoreRowsPerTile][kScoreColsPerTile],
    const float v_tile[kScoreColsPerTile][kHeadDim],
    float out_tile[kScoreRowsPerTile][kHeadDim],
    std::uint16_t query_row_count,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weights_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=v_tile cyclic factor=16 dim=1

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      float accum = 0.0f;
      for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS UNROLL factor=16
        if (row < query_row_count && col < key_col_count) {
          accum += weights_tile[row][col] * v_tile[col][dim];
        }
      }
      out_tile[row][dim] = (row < query_row_count) ? accum : 0.0f;
    }
  }
}

void v_weighted_sum_u55c_kernel(
    const float* weights_tile,
    const float* v_tile,
    float* out_tile,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
#pragma HLS INTERFACE m_axi port=weights_tile offset=slave bundle=gmem_weights depth=512
#pragma HLS INTERFACE m_axi port=v_tile offset=slave bundle=gmem_v depth=4096
#pragma HLS INTERFACE m_axi port=out_tile offset=slave bundle=gmem_out depth=512
#pragma HLS INTERFACE s_axilite port=weights_tile bundle=control
#pragma HLS INTERFACE s_axilite port=v_tile bundle=control
#pragma HLS INTERFACE s_axilite port=out_tile bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float weights_local[kScoreRowsPerTile][kScoreColsPerTile];
  float v_local[kScoreColsPerTile][kHeadDim];
  float out_local[kScoreRowsPerTile][kHeadDim];
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=v_local cyclic factor=16 dim=1

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      weights_local[row][col] = weights_tile[(row * kScoreColsPerTile) + col];
    }
  }

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      v_local[col][dim] = v_tile[(col * kHeadDim) + dim];
    }
  }

  v_weighted_sum_core_hls(
      weights_local,
      v_local,
      out_local,
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count));

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      out_tile[(row * kHeadDim) + dim] = out_local[row][dim];
    }
  }
}

void v_weighted_sum_resident_u55c_kernel(
    const float* weights_full,
    const float* v_full,
    float* out_full,
    std::uint32_t seq_len,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    std::uint32_t clear_accum) {
#pragma HLS INTERFACE m_axi port=weights_full offset=slave bundle=gmem_weights depth=262144
#pragma HLS INTERFACE m_axi port=v_full offset=slave bundle=gmem_v depth=32768
#pragma HLS INTERFACE m_axi port=out_full offset=slave bundle=gmem_out depth=32768
#pragma HLS INTERFACE s_axilite port=weights_full bundle=control
#pragma HLS INTERFACE s_axilite port=v_full bundle=control
#pragma HLS INTERFACE s_axilite port=out_full bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=clear_accum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float weights_local[kScoreRowsPerTile][kScoreColsPerTile];
  float v_local[kScoreColsPerTile][kHeadDim];
  float out_local[kScoreRowsPerTile][kHeadDim];
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=v_local cyclic factor=16 dim=1

  const std::uint16_t seq_len_u16 = static_cast<std::uint16_t>(seq_len);
  const std::uint16_t query_base_u16 = static_cast<std::uint16_t>(query_pos_base);
  const std::uint16_t key_base_u16 = static_cast<std::uint16_t>(key_pos_base);
  const std::uint16_t query_rows_u16 = static_cast<std::uint16_t>(query_row_count);
  const std::uint16_t key_cols_u16 = static_cast<std::uint16_t>(key_col_count);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      float value = 0.0f;
      if ((row < query_rows_u16) && (col < key_cols_u16)) {
        const int global_row = static_cast<int>(query_base_u16) + row;
        const int global_col = static_cast<int>(key_base_u16) + col;
        value = weights_full[(global_row * static_cast<int>(seq_len_u16)) + global_col];
      }
      weights_local[row][col] = value;
    }
  }

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      float value = 0.0f;
      if (col < key_cols_u16) {
        const int global_col = static_cast<int>(key_base_u16) + col;
        value = v_full[(global_col * kHeadDim) + dim];
      }
      v_local[col][dim] = value;
    }
  }

  v_weighted_sum_core_hls(weights_local, v_local, out_local, query_rows_u16, key_cols_u16);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      if (row < query_rows_u16) {
        const int global_row = static_cast<int>(query_base_u16) + row;
        const int out_idx = (global_row * kHeadDim) + dim;
        const float previous = (clear_accum != 0U) ? 0.0f : out_full[out_idx];
        out_full[out_idx] = previous + out_local[row][dim];
      }
    }
  }
}

void v_weighted_sum_multik_u55c_kernel(
    const float* probs_full,
    const float* v_full,
    float* attn_out,
    std::uint32_t seq_len,
    std::uint32_t query_base,
    std::uint32_t query_row_count) {
#pragma HLS INTERFACE m_axi port=probs_full offset=slave bundle=gmem_weights depth=262144
#pragma HLS INTERFACE m_axi port=v_full offset=slave bundle=gmem_v depth=32768
#pragma HLS INTERFACE m_axi port=attn_out offset=slave bundle=gmem_out depth=32768
#pragma HLS INTERFACE s_axilite port=probs_full bundle=control
#pragma HLS INTERFACE s_axilite port=v_full bundle=control
#pragma HLS INTERFACE s_axilite port=attn_out bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=query_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  float acc[kScoreRowsPerTile][kHeadDim];
  float acc_next[kScoreRowsPerTile][kHeadDim];
  float weights_local[kScoreRowsPerTile][kScoreColsPerTile];
  float v_local[kScoreColsPerTile][kHeadDim];
// O3 route-reduction variant: process four query rows at a time instead of
// all eight, while keeping every head dimension independently addressable.
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=kMultikRowChunk dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc_next cyclic factor=kMultikRowChunk dim=1
#pragma HLS ARRAY_PARTITION variable=acc_next complete dim=2
// Bank rows and columns so each row/column chunk can be accessed in parallel
// without recreating the 8-row x 64-column datapath that failed route.
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=kMultikRowChunk dim=1
#pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=kMultikColChunk dim=2
#pragma HLS ARRAY_PARTITION variable=v_local cyclic factor=kMultikColChunk dim=1

  const int seq = static_cast<int>(seq_len);
  const int q_base = static_cast<int>(query_base);
  const int q_rows = static_cast<int>(query_row_count);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      acc[row][dim] = 0.0f;
      acc_next[row][dim] = 0.0f;
    }
  }

  for (int key_base = 0; key_base < seq; key_base += kScoreColsPerTile) {
    const int k_cols = ((key_base + kScoreColsPerTile) <= seq)
                           ? kScoreColsPerTile
                           : (seq - key_base);

    for (int row = 0; row < kScoreRowsPerTile; ++row) {
      for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
        float value = 0.0f;
        if (row < q_rows && col < k_cols) {
          value = probs_full[((q_base + row) * seq) + (key_base + col)];
        }
        weights_local[row][col] = value;
      }
    }

    for (int col = 0; col < kScoreColsPerTile; ++col) {
      for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
        float value = 0.0f;
        if (col < k_cols) {
          value = v_full[((key_base + col) * kHeadDim) + dim];
        }
        v_local[col][dim] = value;
      }
    }

    // Compute small row/column chunks into acc_next, then copy back after the
    // pipelined loop. This keeps the accumulator recurrence out of the hot loop
    // while reducing row-wide routing pressure in the multi-K V kernel.
    for (int col_base = 0; col_base < kScoreColsPerTile; col_base += kMultikColChunk) {
      for (int row_base = 0; row_base < kScoreRowsPerTile; row_base += kMultikRowChunk) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
          for (int row_offset = 0; row_offset < kMultikRowChunk; ++row_offset) {
#pragma HLS UNROLL factor=kMultikRowChunk
            const int row = row_base + row_offset;
            float partial = 0.0f;
            for (int col_offset = 0; col_offset < kMultikColChunk; ++col_offset) {
#pragma HLS UNROLL factor=kMultikColChunk
              const int col = col_base + col_offset;
              partial += weights_local[row][col] * v_local[col][dim];
            }
            acc_next[row][dim] = acc[row][dim] + partial;
          }
        }

        for (int row_offset = 0; row_offset < kMultikRowChunk; ++row_offset) {
          const int row = row_base + row_offset;
          for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
            acc[row][dim] = acc_next[row][dim];
          }
        }
      }
    }
  }

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int dim = 0; dim < kHeadDim; ++dim) {
#pragma HLS PIPELINE II=1
      if (row < q_rows) {
        attn_out[((q_base + row) * kHeadDim) + dim] = acc[row][dim];
      }
    }
  }
}

}  // namespace v_weighted_sum
}  // namespace attention_score_u55c
