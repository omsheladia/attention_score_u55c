#include "v_weighted_sum_core_hls.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using attention_score_u55c::hls_common::kHeadDim;
using attention_score_u55c::hls_common::kScoreColsPerTile;
using attention_score_u55c::hls_common::kScoreRowsPerTile;

constexpr int kWeightsElems = kScoreRowsPerTile * kScoreColsPerTile;
constexpr int kVElems = kScoreColsPerTile * kHeadDim;
constexpr int kOutElems = kScoreRowsPerTile * kHeadDim;

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

  float weights_tile[kWeightsElems] = {};
  float v_tile[kVElems] = {};
  float out_tile[kOutElems] = {};
  float expected[kOutElems] = {};
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;

  if (!load_flat_float_array(base + "/score_softmax.txt", weights_tile, kWeightsElems)) {
    return 1;
  }
  if (!load_flat_float_array(base + "/v_tile.txt", v_tile, kVElems)) {
    return 1;
  }
  if (!load_flat_float_array(base + "/v_partial_expected.txt", expected, kOutElems)) {
    return 1;
  }
  if (!load_kernel_meta(base + "/kernel_meta.txt", &query_row_count, &key_col_count)) {
    return 1;
  }

  attention_score_u55c::v_weighted_sum::v_weighted_sum_u55c_kernel(
      weights_tile,
      v_tile,
      out_tile,
      query_row_count,
      key_col_count);

  std::vector<float> weights_full(static_cast<std::size_t>(key_col_count) * key_col_count, 0.0f);
  std::vector<float> v_full(static_cast<std::size_t>(key_col_count) * kHeadDim, 0.0f);
  std::vector<float> out_full(static_cast<std::size_t>(key_col_count) * kHeadDim, 0.0f);
  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    for (std::uint32_t col = 0; col < key_col_count; ++col) {
      weights_full[(static_cast<std::size_t>(row) * key_col_count) + col] =
          weights_tile[(row * kScoreColsPerTile) + col];
    }
  }
  for (std::uint32_t col = 0; col < key_col_count; ++col) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      v_full[(static_cast<std::size_t>(col) * kHeadDim) + dim] =
          v_tile[(col * kHeadDim) + dim];
    }
  }

  attention_score_u55c::v_weighted_sum::v_weighted_sum_resident_u55c_kernel(
      weights_full.data(),
      v_full.data(),
      out_full.data(),
      key_col_count,
      0,
      0,
      query_row_count,
      key_col_count,
      1);

  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (int idx = 0; idx < kOutElems; ++idx) {
    const float diff = std::fabs(out_tile[idx] - expected[idx]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > 1.0e-4f) {
      ++mismatch_count;
      if (mismatch_count <= 8) {
        std::cerr << "Mismatch at index " << idx
                  << ": got " << out_tile[idx]
                  << ", expected " << expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  for (std::uint32_t row = 0; row < query_row_count; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      const int tile_idx = static_cast<int>((row * kHeadDim) + dim);
      const std::size_t full_idx = (static_cast<std::size_t>(row) * kHeadDim) + dim;
      const float diff = std::fabs(out_full[full_idx] - expected[tile_idx]);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > 1.0e-4f) {
        ++mismatch_count;
        if (mismatch_count <= 8) {
          std::cerr << "Resident mismatch at row " << row << ", dim " << dim
                    << ": got " << out_full[full_idx]
                    << ", expected " << expected[tile_idx]
                    << ", diff " << diff << "\n";
        }
      }
    }
  }

  constexpr std::uint32_t kMultiSeqLen = 128;
  std::vector<float> multi_weights(static_cast<std::size_t>(kMultiSeqLen) * kMultiSeqLen, 0.0f);
  std::vector<float> multi_v(static_cast<std::size_t>(kMultiSeqLen) * kHeadDim, 0.0f);
  std::vector<float> multi_out(static_cast<std::size_t>(kMultiSeqLen) * kHeadDim, 0.0f);
  std::vector<float> multi_expected(kScoreRowsPerTile * kHeadDim, 0.0f);

  for (std::uint32_t row = 0; row < kScoreRowsPerTile; ++row) {
    for (std::uint32_t col = 0; col < kMultiSeqLen; ++col) {
      multi_weights[(static_cast<std::size_t>(row) * kMultiSeqLen) + col] =
          static_cast<float>((row * 13 + col * 7) % 17) / 64.0f;
    }
  }
  for (std::uint32_t col = 0; col < kMultiSeqLen; ++col) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      multi_v[(static_cast<std::size_t>(col) * kHeadDim) + dim] =
          static_cast<float>(static_cast<int>((col * 11 + dim * 5) % 29) - 14) / 16.0f;
    }
  }
  for (std::uint32_t row = 0; row < kScoreRowsPerTile; ++row) {
    for (std::uint32_t col = 0; col < kMultiSeqLen; ++col) {
      const float weight = multi_weights[(static_cast<std::size_t>(row) * kMultiSeqLen) + col];
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        multi_expected[(row * kHeadDim) + dim] +=
            weight * multi_v[(static_cast<std::size_t>(col) * kHeadDim) + dim];
      }
    }
  }

  attention_score_u55c::v_weighted_sum::v_weighted_sum_resident_u55c_kernel(
      multi_weights.data(),
      multi_v.data(),
      multi_out.data(),
      kMultiSeqLen,
      0,
      0,
      kScoreRowsPerTile,
      kScoreColsPerTile,
      1);
  attention_score_u55c::v_weighted_sum::v_weighted_sum_resident_u55c_kernel(
      multi_weights.data(),
      multi_v.data(),
      multi_out.data(),
      kMultiSeqLen,
      0,
      kScoreColsPerTile,
      kScoreRowsPerTile,
      kScoreColsPerTile,
      0);

  for (std::uint32_t row = 0; row < kScoreRowsPerTile; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      const std::size_t idx = (static_cast<std::size_t>(row) * kHeadDim) + dim;
      const float diff = std::fabs(multi_out[idx] - multi_expected[idx]);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > 1.0e-4f) {
        ++mismatch_count;
        if (mismatch_count <= 8) {
          std::cerr << "Resident multi-chunk mismatch at row " << row << ", dim " << dim
                    << ": got " << multi_out[idx]
                    << ", expected " << multi_expected[idx]
                    << ", diff " << diff << "\n";
        }
      }
    }
  }

  if (mismatch_count != 0) {
    std::cerr << "v_weighted_sum test FAILED with " << mismatch_count
              << " mismatches, max diff " << max_diff << "\n";
    return 1;
  }

  std::cout << "v_weighted_sum test PASSED, max diff " << max_diff << "\n";
  return 0;
}
