#include "softmax_full_row_hls.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using attention_score_u55c::hls_common::kScoreRowsPerTile;
using attention_score_u55c::softmax_full_row::kFullRowMaxCols;

constexpr int kElemCount = kScoreRowsPerTile * kFullRowMaxCols;

void fill_case(
    float* score_in,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    bool include_masked_values) {
  for (int idx = 0; idx < kElemCount; ++idx) {
    score_in[idx] = -123.0f;
  }

  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      float value = static_cast<float>((static_cast<int>(row) * 17 + static_cast<int>(col) * 5) % 41 - 20);
      value *= 0.03125f;
      if (include_masked_values && col > (row + 12U)) {
        value = -1000000000.0f;
      }
      score_in[(row * kFullRowMaxCols) + col] = value;
    }
  }
}

void reference_softmax(
    const float* score_in,
    float* expected,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
  for (int idx = 0; idx < kElemCount; ++idx) {
    expected[idx] = 0.0f;
  }

  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    float row_max = score_in[row * kFullRowMaxCols];
    for (std::uint32_t col = 1; col < key_col_count; ++col) {
      const float value = score_in[(row * kFullRowMaxCols) + col];
      if (value > row_max) {
        row_max = value;
      }
    }

    std::vector<float> exp_vals(key_col_count);
    float sum_exp = 0.0f;
    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      const float exp_value = std::exp(score_in[(row * kFullRowMaxCols) + col] - row_max);
      exp_vals[col] = exp_value;
      sum_exp += exp_value;
    }

    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      expected[(row * kFullRowMaxCols) + col] = exp_vals[col] / sum_exp;
    }
  }
}

bool run_case(
    const std::string& name,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count,
    bool include_masked_values) {
  float score_in[kElemCount] = {};
  float prob_out[kElemCount] = {};
  float expected[kElemCount] = {};

  fill_case(score_in, query_row_count, key_col_count, include_masked_values);
  reference_softmax(score_in, expected, query_row_count, key_col_count);

  attention_score_u55c::softmax_full_row::softmax_full_row_u55c_kernel(
      score_in,
      prob_out,
      query_row_count,
      key_col_count);

  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (int idx = 0; idx < kElemCount; ++idx) {
    const float diff = std::fabs(prob_out[idx] - expected[idx]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > 1.0e-4f) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << name << " mismatch at index " << idx
                  << ": got " << prob_out[idx]
                  << ", expected " << expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    float row_sum = 0.0f;
    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      row_sum += prob_out[(row * kFullRowMaxCols) + col];
    }
    const float sum_diff = std::fabs(row_sum - 1.0f);
    if (sum_diff > 1.0e-4f) {
      ++mismatch_count;
      std::cerr << name << " row " << row << " sum is " << row_sum << "\n";
    }
  }

  if (mismatch_count != 0) {
    std::cerr << name << " FAILED with " << mismatch_count
              << " mismatches, max diff " << max_diff << "\n";
    return false;
  }

  std::cout << name << " PASSED, max diff " << max_diff << "\n";
  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= run_case("softmax_full_row S64", 8, 64, false);
  ok &= run_case("softmax_full_row S128", 8, 128, false);
  ok &= run_case("softmax_full_row S256", 8, 256, false);
  ok &= run_case("softmax_full_row S512", 8, 512, false);
  ok &= run_case("softmax_full_row partial", 3, 37, false);
  ok &= run_case("softmax_full_row masked", 8, 512, true);

  if (!ok) {
    std::cerr << "softmax_full_row test FAILED\n";
    return 1;
  }

  std::cout << "softmax_full_row test PASSED\n";
  return 0;
}
