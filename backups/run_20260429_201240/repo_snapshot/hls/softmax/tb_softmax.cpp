#include "softmax_core_hls.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;

bool load_flat_float_array(const std::string& path, float* dst, int count) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  for (int idx = 0; idx < count; ++idx) {
    if (!(handle >> dst[idx])) {
      std::cerr << "Unexpected EOF while reading " << path << "\n";
      return false;
    }
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
  while (handle >> key) {
    if (key == "query_row_count") {
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

  constexpr int kElemCount = kScoreRowsPerTile * kScoreColsPerTile;
  float score_in[kElemCount] = {};
  float prob_out[kElemCount] = {};
  float prob_expected[kElemCount] = {};
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;

  if (!load_flat_float_array(base + "/score_scaled.txt", score_in, kElemCount)) {
    return 1;
  }
  if (!load_flat_float_array(base + "/score_softmax.txt", prob_expected, kElemCount)) {
    return 1;
  }
  if (!load_kernel_meta(base + "/kernel_meta.txt", &query_row_count, &key_col_count)) {
    return 1;
  }

  attention_score_u55c::softmax::softmax_u55c_kernel(
      score_in,
      prob_out,
      query_row_count,
      key_col_count);

  int mismatch_count = 0;
  for (int idx = 0; idx < kElemCount; ++idx) {
    const float diff = std::fabs(prob_out[idx] - prob_expected[idx]);
    if (diff > 1.0e-4f) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << "Mismatch at index " << idx
                  << ": got " << prob_out[idx]
                  << ", expected " << prob_expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  if (mismatch_count != 0) {
    std::cerr << "softmax test FAILED with " << mismatch_count << " mismatches\n";
    return 1;
  }

  std::cout << "softmax test PASSED\n";
  return 0;
}
