#include "score_mask_scale_core_hls.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::hls_common::act_int8_t;
using attention_score_u55c::hls_common::kHeadDim;
using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;
using attention_score_u55c::hls_common::word64_t;

constexpr int kBytesPerWord = 8;
constexpr int kQPackedWords = (kScoreRowsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kKPackedWords = (kScoreColsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kScoreElems = kScoreRowsPerTile * kScoreColsPerTile;

template <typename T, int N>
bool load_flat_array(const std::string& path, T (&dst)[N]) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  long double value = 0.0;
  for (int idx = 0; idx < N; ++idx) {
    if (!(handle >> value)) {
      std::cerr << "Unexpected EOF while reading " << path << "\n";
      return false;
    }
    dst[idx] = static_cast<T>(value);
  }

  return true;
}

bool load_kernel_meta(
    const std::string& path,
    std::uint32_t* query_pos_base,
    std::uint32_t* key_pos_base,
    std::uint32_t* query_row_count,
    std::uint32_t* key_col_count,
    float* total_scale) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  std::string key;
  while (handle >> key) {
    if (key == "query_pos_base") {
      handle >> *query_pos_base;
    } else if (key == "key_pos_base") {
      handle >> *key_pos_base;
    } else if (key == "query_row_count") {
      handle >> *query_row_count;
    } else if (key == "key_col_count") {
      handle >> *key_col_count;
    } else if (key == "total_scale") {
      handle >> *total_scale;
    } else {
      std::string ignored;
      handle >> ignored;
    }
  }

  return true;
}

word64_t pack_int8_word(const act_int8_t* src) {
  word64_t packed = 0;
  for (int lane = 0; lane < kBytesPerWord; ++lane) {
    const auto byte_value = static_cast<unsigned char>(static_cast<std::int8_t>(src[lane]));
    packed |= static_cast<word64_t>(byte_value) << (lane * 8);
  }
  return packed;
}

void pack_q_or_k_tile(
    const act_int8_t* src,
    word64_t* dst,
    int word_count) {
  for (int word_idx = 0; word_idx < word_count; ++word_idx) {
    dst[word_idx] = pack_int8_word(src + (word_idx * kBytesPerWord));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string base = "attention_score_u55c/sim/attention_score_tile";
  if (argc > 1) {
    base = argv[1];
  }

  act_int8_t q_tile[kScoreRowsPerTile * kHeadDim] = {};
  act_int8_t k_tile[kScoreColsPerTile * kHeadDim] = {};
  word64_t q_tile_packed[kQPackedWords] = {};
  word64_t k_tile_packed[kKPackedWords] = {};
  float score_out[kScoreElems] = {};
  float score_expected[kScoreElems] = {};
  std::uint32_t query_pos_base = 0;
  std::uint32_t key_pos_base = 0;
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;
  float total_scale = 0.0f;

  if (!load_flat_array(base + "/q_tile.txt", q_tile)) {
    return 1;
  }
  if (!load_flat_array(base + "/k_tile.txt", k_tile)) {
    return 1;
  }
  if (!load_flat_array(base + "/score_scaled.txt", score_expected)) {
    return 1;
  }
  if (!load_kernel_meta(
          base + "/kernel_meta.txt",
          &query_pos_base,
          &key_pos_base,
          &query_row_count,
          &key_col_count,
          &total_scale)) {
    return 1;
  }

  pack_q_or_k_tile(q_tile, q_tile_packed, kQPackedWords);
  pack_q_or_k_tile(k_tile, k_tile_packed, kKPackedWords);

  attention_score_u55c::score_mask_scale::score_mask_scale_u55c_kernel(
      q_tile_packed,
      k_tile_packed,
      score_out,
      query_pos_base,
      key_pos_base,
      query_row_count,
      key_col_count,
      total_scale);

  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (int idx = 0; idx < kScoreElems; ++idx) {
    const float diff = std::fabs(score_out[idx] - score_expected[idx]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > 1.0e-4f) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << "Mismatch at index " << idx
                  << ": got " << score_out[idx]
                  << ", expected " << score_expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  if (mismatch_count != 0) {
    std::cerr << "score_mask_scale test FAILED with " << mismatch_count
              << " mismatches, max diff " << max_diff << "\n";
    return 1;
  }

  std::cout << "score_mask_scale test PASSED, max diff " << max_diff << "\n";
  return 0;
}
