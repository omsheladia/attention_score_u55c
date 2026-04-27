#include "causal_mask_core_hls.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::causal_mask::accum_int32_t;
using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;

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
    std::uint32_t* query_pos_base,
    std::uint32_t* key_pos_base,
    std::uint32_t* query_row_count,
    std::uint32_t* key_col_count) {
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
    } else {
      std::string ignored;
      handle >> ignored;
    }
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string base = "attention_score_u55c/sim/attention_score_tile";
  if (argc > 1) {
    base = argv[1];
  }

  accum_int32_t score_in[kScoreRowsPerTile * kScoreColsPerTile] = {};
  accum_int32_t score_out[kScoreRowsPerTile * kScoreColsPerTile] = {};
  accum_int32_t score_expected[kScoreRowsPerTile * kScoreColsPerTile] = {};
  std::uint32_t query_pos_base = 0;
  std::uint32_t key_pos_base = 0;
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;

  if (!load_flat_array(base + "/score_raw.txt", score_in)) {
    return 1;
  }
  if (!load_flat_array(base + "/score_masked.txt", score_expected)) {
    return 1;
  }
  if (!load_kernel_meta(
          base + "/kernel_meta.txt",
          &query_pos_base,
          &key_pos_base,
          &query_row_count,
          &key_col_count)) {
    return 1;
  }

  attention_score_u55c::causal_mask::causal_mask_u55c_kernel(
      score_in,
      score_out,
      query_pos_base,
      key_pos_base,
      query_row_count,
      key_col_count);

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
    std::cerr << "causal_mask test FAILED with " << mismatch_count << " mismatches\n";
    return 1;
  }

  std::cout << "causal_mask test PASSED\n";
  return 0;
}
