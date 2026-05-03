#include "v_weighted_sum_core_hls.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using attention_score_u55c::hls_common::kHeadDim;
using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;

constexpr int kWeightsElems = kScoreRowsPerTile * kScoreColsPerTile;  // 8 x 64
constexpr int kVElems       = kScoreColsPerTile * kHeadDim;           // 64 x 64
constexpr int kOutElems     = kScoreRowsPerTile * kHeadDim;           // 8 x 64

// Deterministic fill matching the Python export patterns.
void fill_weights(float* dst, std::uint32_t query_row_count, std::uint32_t key_col_count) {
  for (int idx = 0; idx < kWeightsElems; ++idx) {
    dst[idx] = 0.0f;
  }
  // Uniform distribution over active keys per row (simple, sum-to-1).
  const float w = (key_col_count > 0) ? (1.0f / static_cast<float>(key_col_count)) : 0.0f;
  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      dst[(row * kScoreColsPerTile) + col] = w;
    }
  }
}

void fill_v(float* dst) {
  // v[c][d] = ((c*3 + d*7) % 15 - 7) matching deterministic_v() in Python.
  for (int c = 0; c < kScoreColsPerTile; ++c) {
    for (int d = 0; d < kHeadDim; ++d) {
      dst[(c * kHeadDim) + d] =
          static_cast<float>(((c * 3) + (d * 7)) % 15 - 7);
    }
  }
}

void reference_v_weighted_sum(
    const float* weights,
    const float* v,
    float* expected,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
  for (int idx = 0; idx < kOutElems; ++idx) {
    expected[idx] = 0.0f;
  }
  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    for (int d = 0; d < kHeadDim; ++d) {
      float accum = 0.0f;
      for (std::uint32_t c = 0; c < key_col_count; ++c) {
        accum += weights[(row * kScoreColsPerTile) + c] * v[(c * kHeadDim) + d];
      }
      expected[(row * kHeadDim) + d] = accum;
    }
  }
}

bool run_case(
    const std::string& name,
    std::uint32_t query_row_count,
    std::uint32_t key_col_count) {
  float weights[kWeightsElems] = {};
  float v[kVElems] = {};
  float out[kOutElems] = {};
  float expected[kOutElems] = {};

  fill_weights(weights, query_row_count, key_col_count);
  fill_v(v);
  reference_v_weighted_sum(weights, v, expected, query_row_count, key_col_count);

  attention_score_u55c::v_weighted_sum::v_weighted_sum_u55c_kernel(
      weights,
      v,
      out,
      query_row_count,
      key_col_count);

  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (int idx = 0; idx < kOutElems; ++idx) {
    const float diff = std::fabs(out[idx] - expected[idx]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > 1.0e-3f) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << name << " mismatch at index " << idx
                  << ": got " << out[idx]
                  << ", expected " << expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  // Rows beyond query_row_count must be zero.
  for (std::uint32_t row = query_row_count; row < kScoreRowsPerTile; ++row) {
    for (int d = 0; d < kHeadDim; ++d) {
      const float v_out = out[(row * kHeadDim) + d];
      if (std::fabs(v_out) > 1.0e-6f) {
        ++mismatch_count;
        if (mismatch_count <= 8) {
          std::cerr << name << " inactive row " << row
                    << " dim " << d << " is non-zero: " << v_out << "\n";
        }
      }
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
  ok &= run_case("v_weighted_sum full tile  (8q x 64k)", 8, 64);
  ok &= run_case("v_weighted_sum partial    (4q x 32k)", 4, 32);
  ok &= run_case("v_weighted_sum single row (1q x 64k)", 1, 64);
  ok &= run_case("v_weighted_sum partial k  (8q x 10k)", 8, 10);

  if (!ok) {
    std::cerr << "v_weighted_sum test FAILED\n";
    return 1;
  }

  std::cout << "v_weighted_sum test PASSED\n";
  return 0;
}
