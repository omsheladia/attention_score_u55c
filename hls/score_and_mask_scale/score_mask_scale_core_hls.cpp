#include "score_mask_scale_core_hls.hpp"

#if __has_include(<hls_stream.h>)
  #include <hls_stream.h>
#else
  #include <queue>

namespace hls {
template <typename T>
class stream {
 public:
  void write(const T& value) {
    values_.push(value);
  }

  T read() {
    const T value = values_.front();
    values_.pop();
    return value;
  }

 private:
  std::queue<T> values_;
};
}  // namespace hls
#endif

namespace attention_score_u55c {
namespace score_mask_scale {

using namespace hls_common;

namespace {

constexpr int kBytesPerWord = 8;

struct ScoreRecord {
  std::uint16_t row;
  std::uint16_t col;
  accum_int32_t score;
};

struct ScaledRecord {
  std::uint16_t row;
  std::uint16_t col;
  float score;
};

act_int8_t unpack_int8_lane(word64_t packed_word, int lane_idx) {
  const int shift = lane_idx * 8;
  const word64_t masked = (packed_word >> shift) & static_cast<word64_t>(0xff);
  return static_cast<act_int8_t>(static_cast<std::int8_t>(masked));
}

void load_q_tile(
    const word64_t* q_tile,
    act_int8_t q_local[kScoreRowsPerTile][kHeadDim]) {
#pragma HLS INLINE off

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
}

void load_k_tile(
    const word64_t* k_tile,
    act_int8_t k_local[kScoreColsPerTile][kHeadDim]) {
#pragma HLS INLINE off

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
}

void load_q_tile_from_full(
    const word64_t* q_full,
    act_int8_t q_local[kScoreRowsPerTile][kHeadDim],
    std::uint16_t query_pos_base,
    std::uint16_t query_row_count) {
#pragma HLS INLINE off

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int word_idx = 0; word_idx < (kHeadDim / kBytesPerWord); ++word_idx) {
#pragma HLS PIPELINE II=1
      word64_t packed_word = 0;
      if (row < query_row_count) {
        const int global_row = static_cast<int>(query_pos_base) + row;
        packed_word = q_full[(global_row * (kHeadDim / kBytesPerWord)) + word_idx];
      }
      for (int lane = 0; lane < kBytesPerWord; ++lane) {
#pragma HLS UNROLL
        q_local[row][(word_idx * kBytesPerWord) + lane] = unpack_int8_lane(packed_word, lane);
      }
    }
  }
}

void load_k_tile_from_full(
    const word64_t* k_full,
    act_int8_t k_local[kScoreColsPerTile][kHeadDim],
    std::uint16_t key_pos_base,
    std::uint16_t key_col_count) {
#pragma HLS INLINE off

  for (int col = 0; col < kScoreColsPerTile; ++col) {
    for (int word_idx = 0; word_idx < (kHeadDim / kBytesPerWord); ++word_idx) {
#pragma HLS PIPELINE II=1
      word64_t packed_word = 0;
      if (col < key_col_count) {
        const int global_col = static_cast<int>(key_pos_base) + col;
        packed_word = k_full[(global_col * (kHeadDim / kBytesPerWord)) + word_idx];
      }
      for (int lane = 0; lane < kBytesPerWord; ++lane) {
#pragma HLS UNROLL
        k_local[col][(word_idx * kBytesPerWord) + lane] = unpack_int8_lane(packed_word, lane);
      }
    }
  }
}

void compute_score_stream(
    const act_int8_t q_tile[kScoreRowsPerTile][kHeadDim],
    const act_int8_t k_tile[kScoreColsPerTile][kHeadDim],
    hls::stream<ScoreRecord>& score_stream,
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

      ScoreRecord record;
      record.row = static_cast<std::uint16_t>(row);
      record.col = static_cast<std::uint16_t>(col);
      record.score = accum;
      score_stream.write(record);
    }
  }
}

void mask_scale_stream(
    hls::stream<ScoreRecord>& score_stream,
    hls::stream<ScaledRecord>& scaled_stream,
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale) {
#pragma HLS INLINE off

  for (int idx = 0; idx < (kScoreRowsPerTile * kScoreColsPerTile); ++idx) {
#pragma HLS PIPELINE II=1
    const ScoreRecord record = score_stream.read();
    const bool masked =
        (record.row >= query_row_count) ||
        (record.col >= key_col_count) ||
        ((static_cast<int>(key_pos_base) + record.col) >
         (static_cast<int>(query_pos_base) + record.row));

    const accum_int32_t masked_or_raw =
        masked ? static_cast<accum_int32_t>(kMaskNegInf) : record.score;

    ScaledRecord scaled;
    scaled.row = record.row;
    scaled.col = record.col;
    scaled.score = static_cast<float>(masked_or_raw) * total_scale;
    scaled_stream.write(scaled);
  }
}

void store_scaled_stream(
    hls::stream<ScaledRecord>& scaled_stream,
    float score_out[kScoreRowsPerTile][kScoreColsPerTile]) {
#pragma HLS INLINE off

  for (int idx = 0; idx < (kScoreRowsPerTile * kScoreColsPerTile); ++idx) {
#pragma HLS PIPELINE II=1
    const ScaledRecord scaled = scaled_stream.read();
    score_out[scaled.row][scaled.col] = scaled.score;
  }
}

void score_mask_scale_dataflow(
    const act_int8_t q_tile[kScoreRowsPerTile][kHeadDim],
    const act_int8_t k_tile[kScoreColsPerTile][kHeadDim],
    float score_out[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

  hls::stream<ScoreRecord> score_stream;
  hls::stream<ScaledRecord> scaled_stream;
#pragma HLS STREAM variable=score_stream depth=32
#pragma HLS STREAM variable=scaled_stream depth=32

  compute_score_stream(q_tile, k_tile, score_stream, query_row_count, key_col_count);
  mask_scale_stream(
      score_stream,
      scaled_stream,
      query_pos_base,
      key_pos_base,
      query_row_count,
      key_col_count,
      total_scale);
  store_scaled_stream(scaled_stream, score_out);
}

}  // namespace

void score_mask_scale_core_hls(
    const act_int8_t q_tile[kScoreRowsPerTile][kHeadDim],
    const act_int8_t k_tile[kScoreColsPerTile][kHeadDim],
    float score_out[kScoreRowsPerTile][kScoreColsPerTile],
    std::uint16_t query_pos_base,
    std::uint16_t key_pos_base,
    std::uint16_t query_row_count,
    std::uint16_t key_col_count,
    float total_scale) {
#pragma HLS INLINE off

  score_mask_scale_dataflow(
      q_tile,
      k_tile,
      score_out,
      query_pos_base,
      key_pos_base,
      query_row_count,
      key_col_count,
      total_scale);
}

void score_mask_scale_u55c_kernel(
    const word64_t* q_tile,
    const word64_t* k_tile,
    float* score_out,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    float total_scale) {
#pragma HLS INTERFACE m_axi port=q_tile offset=slave bundle=gmem_q depth=64
#pragma HLS INTERFACE m_axi port=k_tile offset=slave bundle=gmem_k depth=512
#pragma HLS INTERFACE m_axi port=score_out offset=slave bundle=gmem_out depth=512
#pragma HLS INTERFACE s_axilite port=q_tile bundle=control
#pragma HLS INTERFACE s_axilite port=k_tile bundle=control
#pragma HLS INTERFACE s_axilite port=score_out bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=total_scale bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  act_int8_t q_local[kScoreRowsPerTile][kHeadDim];
  act_int8_t k_local[kScoreColsPerTile][kHeadDim];
  float score_out_local[kScoreRowsPerTile][kScoreColsPerTile];

#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=k_local cyclic factor=16 dim=2

  load_q_tile(q_tile, q_local);
  load_k_tile(k_tile, k_local);

  score_mask_scale_core_hls(
      q_local,
      k_local,
      score_out_local,
      static_cast<std::uint16_t>(query_pos_base),
      static_cast<std::uint16_t>(key_pos_base),
      static_cast<std::uint16_t>(query_row_count),
      static_cast<std::uint16_t>(key_col_count),
      total_scale);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      score_out[(row * kScoreColsPerTile) + col] = score_out_local[row][col];
    }
  }
}

void score_mask_scale_resident_u55c_kernel(
    const word64_t* q_full,
    const word64_t* k_full,
    float* logits_out,
    std::uint32_t seq_len,
    std::uint32_t query_pos_base,
    std::uint32_t key_pos_base,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    float total_scale) {
#pragma HLS INTERFACE m_axi port=q_full offset=slave bundle=gmem_q depth=4096
#pragma HLS INTERFACE m_axi port=k_full offset=slave bundle=gmem_k depth=4096
#pragma HLS INTERFACE m_axi port=logits_out offset=slave bundle=gmem_out depth=262144
#pragma HLS INTERFACE s_axilite port=q_full bundle=control
#pragma HLS INTERFACE s_axilite port=k_full bundle=control
#pragma HLS INTERFACE s_axilite port=logits_out bundle=control
#pragma HLS INTERFACE s_axilite port=seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=query_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_pos_base bundle=control
#pragma HLS INTERFACE s_axilite port=query_row_count bundle=control
#pragma HLS INTERFACE s_axilite port=key_col_count bundle=control
#pragma HLS INTERFACE s_axilite port=total_scale bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  act_int8_t q_local[kScoreRowsPerTile][kHeadDim];
  act_int8_t k_local[kScoreColsPerTile][kHeadDim];
  float score_out_local[kScoreRowsPerTile][kScoreColsPerTile];

#pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=k_local cyclic factor=16 dim=2

  const std::uint16_t seq_len_u16 = static_cast<std::uint16_t>(seq_len);
  const std::uint16_t query_base_u16 = static_cast<std::uint16_t>(query_pos_base);
  const std::uint16_t key_base_u16 = static_cast<std::uint16_t>(key_pos_base);
  const std::uint16_t query_rows_u16 = static_cast<std::uint16_t>(query_row_count);
  const std::uint16_t key_cols_u16 = static_cast<std::uint16_t>(key_col_count);

  load_q_tile_from_full(q_full, q_local, query_base_u16, query_rows_u16);
  load_k_tile_from_full(k_full, k_local, key_base_u16, key_cols_u16);

  score_mask_scale_core_hls(
      q_local,
      k_local,
      score_out_local,
      query_base_u16,
      key_base_u16,
      query_rows_u16,
      key_cols_u16,
      total_scale);

  for (int row = 0; row < kScoreRowsPerTile; ++row) {
    for (int col = 0; col < kScoreColsPerTile; ++col) {
#pragma HLS PIPELINE II=1
      if ((row < query_rows_u16) && (col < key_cols_u16)) {
        const int global_row = static_cast<int>(query_base_u16) + row;
        const int global_col = static_cast<int>(key_base_u16) + col;
        logits_out[(global_row * static_cast<int>(seq_len_u16)) + global_col] =
            score_out_local[row][col];
      }
    }
  }
}

}  // namespace score_mask_scale
}  // namespace attention_score_u55c
