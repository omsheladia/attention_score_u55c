#include "score_scale_core_hls.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;
using attention_score_u55c::score_scale::accum_int32_t;

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

bool load_total_scale(const std::string& path, float* total_scale) {
  std::ifstream handle(path);
  if (!handle) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }

  std::string key;
  while (handle >> key) {
    if (key == "total_scale") {
      handle >> *total_scale;
      return true;
    }
    std::string ignored;
    handle >> ignored;
  }

  std::cerr << "Kernel meta file is missing total_scale\n";
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::string base = "attention_score_u55c/sim/attention_score_tile";
  if (argc > 1) {
    base = argv[1];
  }

  constexpr int kElemCount = kScoreRowsPerTile * kScoreColsPerTile;
  accum_int32_t score_in[kElemCount] = {};
  float score_out[kElemCount] = {};
  float score_expected[kElemCount] = {};
  float total_scale = 0.0f;

  if (!load_flat_array(base + "/score_masked.txt", score_in)) {
    return 1;
  }
  if (!load_flat_float_array(base + "/score_scaled.txt", score_expected, kElemCount)) {
    return 1;
  }
  if (!load_total_scale(base + "/kernel_meta.txt", &total_scale)) {
    return 1;
  }

  attention_score_u55c::score_scale::score_scale_u55c_kernel(
      score_in,
      score_out,
      total_scale);

  int mismatch_count = 0;
  for (int idx = 0; idx < kElemCount; ++idx) {
    const float diff = std::fabs(score_out[idx] - score_expected[idx]);
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
    std::cerr << "score_scale test FAILED with " << mismatch_count << " mismatches\n";
    return 1;
  }

  std::cout << "score_scale test PASSED\n";
  return 0;
}
