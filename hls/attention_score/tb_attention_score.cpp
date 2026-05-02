#include "attention_score_core_hls.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::hls_common::accum_int32_t;
using attention_score_u55c::hls_common::act_int8_t;
using attention_score_u55c::hls_common::kHeadDim;
using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;
using attention_score_u55c::hls_common::word64_t;

constexpr int kBytesPerWord = 8;
constexpr int kQPackedWords = (kScoreRowsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kKPackedWords = (kScoreColsPerTile * kHeadDim) / kBytesPerWord;
constexpr int kScorePackedWords =
    (kScoreRowsPerTile * kScoreColsPerTile * static_cast<int>(sizeof(accum_int32_t))) /
    kBytesPerWord;

template <typename T, int N>
bool load_flat_array(const std::string& path, T (&dst)[N]) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  long long value = 0;
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
    std::uint32_t* query_row_count,
    std::uint32_t* key_col_count) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  std::string key;
  std::uint32_t value = 0;
  bool saw_query_rows = false;
  bool saw_key_cols = false;

  while (handle >> key >> value) {
    if (key == "query_row_count") {
      *query_row_count = value;
      saw_query_rows = true;
    } else if (key == "key_col_count") {
      *key_col_count = value;
      saw_key_cols = true;
    }
  }

  if (!saw_query_rows || !saw_key_cols) {
    std::cerr << "Kernel meta file is missing query_row_count or key_col_count\n";
    return false;
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

void unpack_score_words(
    const word64_t* src,
    accum_int32_t* dst,
    int elem_count) {
  for (int word_idx = 0; word_idx < (elem_count / 2); ++word_idx) {
    const word64_t packed = src[word_idx];
    dst[(word_idx * 2)] = static_cast<accum_int32_t>(static_cast<std::int32_t>(packed & 0xffffffffULL));
    dst[(word_idx * 2) + 1] =
        static_cast<accum_int32_t>(static_cast<std::int32_t>((packed >> 32) & 0xffffffffULL));
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
  word64_t score_out_packed[kScorePackedWords] = {};
  accum_int32_t score_out[kScoreRowsPerTile * kScoreColsPerTile] = {};
  accum_int32_t score_expected[kScoreRowsPerTile * kScoreColsPerTile] = {};
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;

  if (!load_flat_array(base + "/q_tile.txt", q_tile)) {
    return 1;
  }
  if (!load_flat_array(base + "/k_tile.txt", k_tile)) {
    return 1;
  }
  if (!load_flat_array(base + "/score_raw.txt", score_expected)) {
    return 1;
  }
  if (!load_kernel_meta(base + "/kernel_meta.txt", &query_row_count, &key_col_count)) {
    return 1;
  }

  pack_q_or_k_tile(q_tile, q_tile_packed, kQPackedWords);
  pack_q_or_k_tile(k_tile, k_tile_packed, kKPackedWords);

  attention_score_u55c::attention_score::attention_score_u55c_kernel(
      q_tile_packed,
      k_tile_packed,
      score_out_packed,
      query_row_count,
      key_col_count);

  unpack_score_words(
      score_out_packed,
      score_out,
      kScoreRowsPerTile * kScoreColsPerTile);

  int mismatch_count = 0;
  for (int idx = 0; idx < (kScoreRowsPerTile * kScoreColsPerTile); ++idx) {
    if (score_out[idx] != score_expected[idx]) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << "Mismatch at index " << idx
                  << ": got " << static_cast<std::int32_t>(score_out[idx])
                  << ", expected " << static_cast<std::int32_t>(score_expected[idx])
                  << "\n";
      }
    }
  }

  if (mismatch_count != 0) {
    std::cerr << "attention_score test FAILED with " << mismatch_count << " mismatches\n";
    return 1;
  }

  std::cout << "attention_score test PASSED\n";
  return 0;
}
