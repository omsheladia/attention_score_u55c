#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kScoreRowsPerTile = 8;
constexpr std::size_t kScoreColsPerTile = 64;
constexpr std::size_t kFullRowMaxCols = 512;
constexpr std::size_t kQTileElems = kScoreRowsPerTile * kHeadDim;
constexpr std::size_t kKTileElems = kScoreColsPerTile * kHeadDim;
constexpr std::size_t kScoreTileElems = kScoreRowsPerTile * kScoreColsPerTile;
constexpr std::size_t kVTileElems = kScoreColsPerTile * kHeadDim;
constexpr std::size_t kAttnOutTileElems = kScoreRowsPerTile * kHeadDim;
constexpr float kMaskNegInf = -1000000000.0f;
constexpr float kDefaultQScale = 0.03125f;
constexpr float kDefaultKScale = 0.02734375f;

struct KernelMeta {
  std::uint32_t query_row_count = 0;
  std::uint32_t key_col_count = 0;
  std::uint32_t query_pos_base = 0;
  std::uint32_t key_pos_base = 0;
  float q_scale = 0.0f;
  float k_scale = 0.0f;
  float total_scale = 0.0f;
};

struct Args {
  std::string xclbin_path;
  std::string vector_dir = "attention_score_u55c/sim/attention_score_tile";
  unsigned int device_index = 0;
  std::uint32_t seq_len = 0;
  float q_scale = kDefaultQScale;
  float k_scale = kDefaultKScale;
};

struct TimedStage {
  std::string name;
  double milliseconds = 0.0;
};

struct PendingMaskScale {
  bool valid = false;
  std::uint32_t key_base = 0;
  std::uint32_t key_cols = 0;
  std::chrono::steady_clock::time_point launch_time;
  std::optional<xrt::run> run;
};

template <typename T>
std::vector<T> read_text_vector(const std::string& path, std::size_t elem_count) {
  std::ifstream handle(path);
  if (!handle) {
    throw std::runtime_error("Failed to open " + path);
  }

  std::vector<T> values(elem_count);
  for (std::size_t idx = 0; idx < elem_count; ++idx) {
    long double raw_value = 0.0;
    if (!(handle >> raw_value)) {
      throw std::runtime_error("Unexpected EOF while reading " + path);
    }
    values[idx] = static_cast<T>(raw_value);
  }
  return values;
}

template <typename T>
std::vector<T> read_text_vector_all(const std::string& path) {
  std::ifstream handle(path);
  if (!handle) {
    throw std::runtime_error("Failed to open " + path);
  }

  std::vector<T> values;
  long double raw_value = 0.0;
  while (handle >> raw_value) {
    values.push_back(static_cast<T>(raw_value));
  }
  if (values.empty()) {
    throw std::runtime_error("No values found while reading " + path);
  }
  return values;
}

bool file_exists(const std::string& path) {
  std::ifstream handle(path);
  return static_cast<bool>(handle);
}

KernelMeta read_kernel_meta(const std::string& path) {
  std::ifstream handle(path);
  if (!handle) {
    throw std::runtime_error("Failed to open " + path);
  }

  KernelMeta meta;
  std::string key;
  while (handle >> key) {
    if (key == "query_row_count") {
      handle >> meta.query_row_count;
    } else if (key == "key_col_count") {
      handle >> meta.key_col_count;
    } else if (key == "query_pos_base") {
      handle >> meta.query_pos_base;
    } else if (key == "key_pos_base") {
      handle >> meta.key_pos_base;
    } else if (key == "q_scale") {
      handle >> meta.q_scale;
    } else if (key == "k_scale") {
      handle >> meta.k_scale;
    } else if (key == "total_scale") {
      handle >> meta.total_scale;
    } else {
      std::string ignored;
      handle >> ignored;
    }
  }

  return meta;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int idx = 1; idx < argc; ++idx) {
    const std::string arg = argv[idx];
    if (arg == "--xclbin" && (idx + 1) < argc) {
      args.xclbin_path = argv[++idx];
    } else if (arg == "--vectors" && (idx + 1) < argc) {
      args.vector_dir = argv[++idx];
    } else if (arg == "--device" && (idx + 1) < argc) {
      args.device_index = static_cast<unsigned int>(std::stoul(argv[++idx]));
    } else if (arg == "--seq-len" && (idx + 1) < argc) {
      args.seq_len = static_cast<std::uint32_t>(std::stoul(argv[++idx]));
    } else if (arg == "--q-scale" && (idx + 1) < argc) {
      args.q_scale = std::stof(argv[++idx]);
    } else if (arg == "--k-scale" && (idx + 1) < argc) {
      args.k_scale = std::stof(argv[++idx]);
    } else {
      throw std::runtime_error(
          "Usage: host_attention_score_chain --xclbin <path> [--vectors <dir>] "
          "[--device <idx>] [--seq-len <S>] [--q-scale <float>] [--k-scale <float>]");
    }
  }

  if (args.xclbin_path.empty()) {
    throw std::runtime_error("Missing required --xclbin argument");
  }
  if (args.seq_len > kFullRowMaxCols) {
    throw std::runtime_error("--seq-len must be <= 512 for the current full-row softmax kernel");
  }
  return args;
}

template <typename T>
void compare_exact(
    const std::vector<T>& got,
    const std::vector<T>& expected,
    const std::string& name) {
  if (got.size() != expected.size()) {
    throw std::runtime_error(name + " size mismatch");
  }

  std::size_t mismatches = 0;
  for (std::size_t idx = 0; idx < got.size(); ++idx) {
    if (got[idx] != expected[idx]) {
      ++mismatches;
      if (mismatches <= 8) {
        std::cerr << name << " mismatch at " << idx
                  << ": got " << static_cast<long long>(got[idx])
                  << ", expected " << static_cast<long long>(expected[idx]) << "\n";
      }
    }
  }

  if (mismatches != 0) {
    throw std::runtime_error(name + " verification failed");
  }
}

void compare_float(
    const std::vector<float>& got,
    const std::vector<float>& expected,
    const std::string& name,
    float tol) {
  if (got.size() != expected.size()) {
    throw std::runtime_error(name + " size mismatch");
  }

  std::size_t mismatches = 0;
  for (std::size_t idx = 0; idx < got.size(); ++idx) {
    const float diff = std::fabs(got[idx] - expected[idx]);
    if (diff > tol) {
      ++mismatches;
      if (mismatches <= 8) {
        std::cerr << std::fixed << std::setprecision(8)
                  << name << " mismatch at " << idx
                  << ": got " << got[idx]
                  << ", expected " << expected[idx]
                  << ", diff " << diff << "\n";
      }
    }
  }

  if (mismatches != 0) {
    throw std::runtime_error(name + " verification failed");
  }
}

template <typename LaunchFn>
double run_timed(const std::string& name, LaunchFn&& launch, bool verbose = true) {
  using Clock = std::chrono::steady_clock;

  if (verbose) {
    std::cout << "Running " << name << "\n";
  }
  const auto start = Clock::now();
  auto run = launch();
  run.wait();
  const auto stop = Clock::now();

  return std::chrono::duration<double, std::milli>(stop - start).count();
}

void print_timings(const std::vector<TimedStage>& stages, double total_ms) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Kernel timing summary (host wall-clock, launch through wait):\n";
  for (const auto& stage : stages) {
    std::cout << "  " << std::left << std::setw(28) << stage.name << std::right
              << stage.milliseconds << " ms\n";
  }
  std::cout << "  " << std::left << std::setw(28) << "total_chain" << std::right
            << total_ms << " ms\n";
  std::cout.unsetf(std::ios::floatfield);
}

int positive_mod(int value, int divisor) {
  const int result = value % divisor;
  return (result < 0) ? result + divisor : result;
}

std::int8_t deterministic_q_value(std::uint32_t row, std::uint32_t dim) {
  return static_cast<std::int8_t>(positive_mod(static_cast<int>(row * 5 + dim * 3), 15) - 7);
}

std::int8_t deterministic_k_value(std::uint32_t col, std::uint32_t dim) {
  return static_cast<std::int8_t>(positive_mod(static_cast<int>(col * 7) - static_cast<int>(dim * 2), 15) - 7);
}

float deterministic_v_value(std::uint32_t row, std::uint32_t dim) {
  const int raw = positive_mod(static_cast<int>(row * 11 + dim * 5), 23) - 11;
  return static_cast<float>(raw) / 8.0f;
}

std::vector<std::int8_t> make_synthetic_q(std::uint32_t seq_len) {
  std::vector<std::int8_t> q(seq_len * kHeadDim);
  for (std::uint32_t row = 0; row < seq_len; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      q[(row * kHeadDim) + dim] = deterministic_q_value(row, dim);
    }
  }
  return q;
}

std::vector<std::int8_t> make_synthetic_k(std::uint32_t seq_len) {
  std::vector<std::int8_t> k(seq_len * kHeadDim);
  for (std::uint32_t col = 0; col < seq_len; ++col) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      k[(col * kHeadDim) + dim] = deterministic_k_value(col, dim);
    }
  }
  return k;
}

std::vector<float> make_synthetic_v(std::uint32_t seq_len) {
  std::vector<float> v(seq_len * kHeadDim);
  for (std::uint32_t row = 0; row < seq_len; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      v[(row * kHeadDim) + dim] = deterministic_v_value(row, dim);
    }
  }
  return v;
}

float total_scale(float q_scale, float k_scale) {
  return q_scale * k_scale * (1.0f / std::sqrt(static_cast<float>(kHeadDim)));
}

void compute_cpu_reference(
    const std::vector<std::int8_t>& q_full,
    const std::vector<std::int8_t>& k_full,
    std::uint32_t seq_len,
    float scale,
    std::vector<std::int32_t>* raw_expected,
    std::vector<float>* logits_expected,
    std::vector<float>* softmax_expected) {
  raw_expected->assign(seq_len * seq_len, 0);
  logits_expected->assign(seq_len * seq_len, 0.0f);
  softmax_expected->assign(seq_len * seq_len, 0.0f);

  for (std::uint32_t row = 0; row < seq_len; ++row) {
    for (std::uint32_t col = 0; col < seq_len; ++col) {
      std::int32_t accum = 0;
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        accum += static_cast<std::int32_t>(q_full[(row * kHeadDim) + dim]) *
                 static_cast<std::int32_t>(k_full[(col * kHeadDim) + dim]);
      }
      (*raw_expected)[(row * seq_len) + col] = accum;
      const float masked_or_raw = (col > row) ? kMaskNegInf : static_cast<float>(accum);
      (*logits_expected)[(row * seq_len) + col] = masked_or_raw * scale;
    }
  }

  for (std::uint32_t row = 0; row < seq_len; ++row) {
    const std::size_t row_base = row * seq_len;
    float row_max = (*logits_expected)[row_base];
    for (std::uint32_t col = 1; col < seq_len; ++col) {
      row_max = std::max(row_max, (*logits_expected)[row_base + col]);
    }

    float sum_exp = 0.0f;
    for (std::uint32_t col = 0; col < seq_len; ++col) {
      const float exp_value = std::exp((*logits_expected)[row_base + col] - row_max);
      (*softmax_expected)[row_base + col] = exp_value;
      sum_exp += exp_value;
    }
    for (std::uint32_t col = 0; col < seq_len; ++col) {
      (*softmax_expected)[row_base + col] /= sum_exp;
    }
  }
}

void compute_attn_out_reference(
    const std::vector<float>& softmax,
    const std::vector<float>& v_full,
    std::uint32_t seq_len,
    std::vector<float>* attn_out_expected) {
  attn_out_expected->assign(seq_len * kHeadDim, 0.0f);
  for (std::uint32_t row = 0; row < seq_len; ++row) {
    for (std::uint32_t key = 0; key < seq_len; ++key) {
      const float weight = softmax[(row * seq_len) + key];
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        (*attn_out_expected)[(row * kHeadDim) + dim] +=
            weight * v_full[(key * kHeadDim) + dim];
      }
    }
  }
}

void copy_q_tile(
    const std::vector<std::int8_t>& q_full,
    std::uint32_t seq_len,
    std::uint32_t query_base,
    std::vector<std::int8_t>* q_tile) {
  q_tile->assign(kQTileElems, 0);
  const std::uint32_t rows = std::min<std::uint32_t>(kScoreRowsPerTile, seq_len - query_base);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      (*q_tile)[(row * kHeadDim) + dim] = q_full[((query_base + row) * kHeadDim) + dim];
    }
  }
}

void copy_k_tile(
    const std::vector<std::int8_t>& k_full,
    std::uint32_t seq_len,
    std::uint32_t key_base,
    std::vector<std::int8_t>* k_tile) {
  k_tile->assign(kKTileElems, 0);
  const std::uint32_t cols = std::min<std::uint32_t>(kScoreColsPerTile, seq_len - key_base);
  for (std::uint32_t col = 0; col < cols; ++col) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      (*k_tile)[(col * kHeadDim) + dim] = k_full[((key_base + col) * kHeadDim) + dim];
    }
  }
}

void copy_v_tile(
    const std::vector<float>& v_full,
    std::uint32_t seq_len,
    std::uint32_t key_base,
    std::vector<float>* v_tile) {
  v_tile->assign(kVTileElems, 0.0f);
  const std::uint32_t rows = std::min<std::uint32_t>(kScoreColsPerTile, seq_len - key_base);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      (*v_tile)[(row * kHeadDim) + dim] = v_full[((key_base + row) * kHeadDim) + dim];
    }
  }
}

std::vector<std::int32_t> read_i32_bo(xrt::bo& bo, std::size_t count) {
  std::vector<std::int32_t> values(count);
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo.read(values.data());
  return values;
}

std::vector<float> read_float_bo(xrt::bo& bo, std::size_t count) {
  std::vector<float> values(count);
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo.read(values.data());
  return values;
}

int run_single_tile(const Args& args, xrt::device& device, const xrt::uuid& uuid) {
  const auto q_tile = read_text_vector<std::int8_t>(args.vector_dir + "/q_tile.txt", kQTileElems);
  const auto k_tile = read_text_vector<std::int8_t>(args.vector_dir + "/k_tile.txt", kKTileElems);
  const auto score_raw_expected =
      read_text_vector<std::int32_t>(args.vector_dir + "/score_raw.txt", kScoreTileElems);
  const auto score_scaled_expected =
      read_text_vector<float>(args.vector_dir + "/score_scaled.txt", kScoreTileElems);
  const auto score_softmax_expected =
      read_text_vector<float>(args.vector_dir + "/score_softmax.txt", kScoreTileElems);
  const auto meta = read_kernel_meta(args.vector_dir + "/kernel_meta.txt");
  const bool has_attn_out = file_exists(args.vector_dir + "/attn_out.txt");
  const bool has_attn_ref = file_exists(args.vector_dir + "/attn_ref_float.txt");
  const bool run_v_stage = file_exists(args.vector_dir + "/v_full.txt") && (has_attn_out || has_attn_ref);

  std::vector<float> v_full;
  std::vector<float> attn_out_expected;
  float attn_out_tolerance = 1.0e-4f;
  if (run_v_stage) {
    v_full = read_text_vector<float>(
        args.vector_dir + "/v_full.txt",
        static_cast<std::size_t>(meta.key_col_count) * kHeadDim);
    if (has_attn_out) {
      attn_out_expected =
          read_text_vector<float>(args.vector_dir + "/attn_out.txt", kAttnOutTileElems);
    } else {
      attn_out_expected = read_text_vector<float>(
          args.vector_dir + "/attn_ref_float.txt",
          static_cast<std::size_t>(meta.query_row_count) * kHeadDim);
      attn_out_tolerance = 1.0e-3f;
    }
  }

  auto score_kernel = xrt::kernel(device, uuid, "attention_score_u55c_kernel");
  auto mask_scale_kernel = xrt::kernel(device, uuid, "mask_scale_u55c_kernel");
  auto softmax_kernel = xrt::kernel(device, uuid, "softmax_u55c_kernel");
  std::optional<xrt::kernel> v_weighted_sum_kernel;
  if (run_v_stage) {
    v_weighted_sum_kernel.emplace(device, uuid, "v_weighted_sum_u55c_kernel");
  }

  auto q_bo = xrt::bo(device, sizeof(std::int8_t) * q_tile.size(), score_kernel.group_id(0));
  auto k_bo = xrt::bo(device, sizeof(std::int8_t) * k_tile.size(), score_kernel.group_id(1));
  auto raw_score_bo =
      xrt::bo(device, sizeof(std::int32_t) * score_raw_expected.size(), score_kernel.group_id(2));
  auto scaled_score_bo =
      xrt::bo(device, sizeof(float) * score_scaled_expected.size(), mask_scale_kernel.group_id(1));
  auto softmax_prob_bo =
      xrt::bo(device, sizeof(float) * score_softmax_expected.size(), softmax_kernel.group_id(1));
  std::optional<xrt::bo> weights_tile_bo;
  std::optional<xrt::bo> v_tile_bo;
  std::optional<xrt::bo> attn_out_bo;
  if (run_v_stage) {
    weights_tile_bo.emplace(
        device,
        sizeof(float) * kScoreTileElems,
        v_weighted_sum_kernel->group_id(0));
    v_tile_bo.emplace(
        device,
        sizeof(float) * kVTileElems,
        v_weighted_sum_kernel->group_id(1));
    attn_out_bo.emplace(
        device,
        sizeof(float) * kAttnOutTileElems,
        v_weighted_sum_kernel->group_id(2));
  }

  q_bo.write(q_tile.data());
  k_bo.write(k_tile.data());
  q_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  k_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  using Clock = std::chrono::steady_clock;
  std::vector<TimedStage> timings;
  timings.reserve(run_v_stage ? 5 : 4);
  const auto chain_start = Clock::now();

  timings.push_back({"attention_score_u55c_kernel", run_timed("attention score kernel", [&]() {
                        return score_kernel(
                            q_bo,
                            k_bo,
                            raw_score_bo,
                            meta.query_row_count,
                            meta.key_col_count);
                      })});

  timings.push_back({"mask_scale_u55c_kernel", run_timed("mask+scale kernel", [&]() {
                        return mask_scale_kernel(
                            raw_score_bo,
                            scaled_score_bo,
                            meta.query_pos_base,
                            meta.key_pos_base,
                            meta.query_row_count,
                            meta.key_col_count,
                            meta.total_scale);
                      })});

  timings.push_back({"softmax_u55c_kernel", run_timed("softmax kernel", [&]() {
                        return softmax_kernel(
                            scaled_score_bo,
                            softmax_prob_bo,
                            meta.query_row_count,
                            meta.key_col_count);
                      })});

  if (run_v_stage) {
    auto softmax_for_v = read_float_bo(softmax_prob_bo, score_softmax_expected.size());
    weights_tile_bo->write(softmax_for_v.data());
    weights_tile_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<float> v_tile(kVTileElems, 0.0f);
    for (std::uint32_t row = 0; row < meta.key_col_count; ++row) {
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        v_tile[(row * kHeadDim) + dim] = v_full[(row * kHeadDim) + dim];
      }
    }
    v_tile_bo->write(v_tile.data());
    v_tile_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    timings.push_back({"v_weighted_sum_u55c_kernel", run_timed("v weighted sum kernel", [&]() {
                         return (*v_weighted_sum_kernel)(
                             *weights_tile_bo,
                             *v_tile_bo,
                             *attn_out_bo,
                             meta.query_row_count,
                             meta.key_col_count);
                       })});
  }

  const auto chain_stop = Clock::now();
  const double total_chain_ms =
      std::chrono::duration<double, std::milli>(chain_stop - chain_start).count();

  const auto score_raw_got = read_i32_bo(raw_score_bo, score_raw_expected.size());
  const auto score_scaled_got = read_float_bo(scaled_score_bo, score_scaled_expected.size());
  const auto score_softmax_got = read_float_bo(softmax_prob_bo, score_softmax_expected.size());

  compare_exact(score_raw_got, score_raw_expected, "score_raw");
  compare_float(score_scaled_got, score_scaled_expected, "score_scaled", 1.0e-4f);
  compare_float(score_softmax_got, score_softmax_expected, "score_softmax", 1.0e-4f);
  if (run_v_stage) {
    const auto attn_out_padded = read_float_bo(*attn_out_bo, kAttnOutTileElems);
    if (attn_out_expected.size() == kAttnOutTileElems) {
      compare_float(attn_out_padded, attn_out_expected, "attn_out", attn_out_tolerance);
    } else {
      std::vector<float> attn_out_active(attn_out_expected.size(), 0.0f);
      for (std::uint32_t row = 0; row < meta.query_row_count; ++row) {
        for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
          attn_out_active[(row * kHeadDim) + dim] = attn_out_padded[(row * kHeadDim) + dim];
        }
      }
      compare_float(attn_out_active, attn_out_expected, "attn_out", attn_out_tolerance);
    }
  }

  print_timings(timings, total_chain_ms);
  if (run_v_stage) {
    std::cout << "Attention output verification PASSED\n";
  }
  std::cout << "XRT chain verification PASSED\n";
  return 0;
}

int run_tiled_inputs(
    const std::string& label,
    const std::vector<std::int8_t>& q_full,
    const std::vector<std::int8_t>& k_full,
    const std::vector<float>& v_full,
    float scale,
    const std::vector<std::int32_t>& raw_expected,
    const std::vector<float>& logits_expected,
    const std::vector<float>& softmax_expected,
    const std::vector<float>& attn_out_expected,
    float attn_out_tolerance,
    xrt::device& device,
    const xrt::uuid& uuid) {
  if (q_full.empty() || k_full.empty() || v_full.empty()) {
    throw std::runtime_error("Tiled inputs must not be empty");
  }
  if ((q_full.size() % kHeadDim) != 0 || (k_full.size() % kHeadDim) != 0 ||
      (v_full.size() % kHeadDim) != 0) {
    throw std::runtime_error("Tiled input sizes must be multiples of head_dim");
  }

  const std::uint32_t seq_len = static_cast<std::uint32_t>(q_full.size() / kHeadDim);
  if (seq_len == 0) {
    throw std::runtime_error("Tiled sequence length must be positive");
  }
  if ((k_full.size() / kHeadDim) != seq_len || (v_full.size() / kHeadDim) != seq_len) {
    throw std::runtime_error("Current tiled host path expects Q, K, and V to share seq_len");
  }
  const std::size_t score_elems = static_cast<std::size_t>(seq_len) * seq_len;
  const std::size_t attn_elems = static_cast<std::size_t>(seq_len) * kHeadDim;
  if (raw_expected.size() != score_elems || logits_expected.size() != score_elems ||
      softmax_expected.size() != score_elems || attn_out_expected.size() != attn_elems) {
    throw std::runtime_error("Tiled reference file size mismatch");
  }

  const std::uint32_t q_chunks =
      (seq_len + static_cast<std::uint32_t>(kScoreRowsPerTile) - 1U) /
      static_cast<std::uint32_t>(kScoreRowsPerTile);
  const std::uint32_t k_chunks =
      (seq_len + static_cast<std::uint32_t>(kScoreColsPerTile) - 1U) /
      static_cast<std::uint32_t>(kScoreColsPerTile);

  std::cout << "Running " << label << ", S=" << seq_len
            << ", q_chunks=" << q_chunks
            << ", k_chunks=" << k_chunks << "\n";

  auto score_kernel = xrt::kernel(device, uuid, "attention_score_u55c_kernel");
  auto mask_scale_kernel = xrt::kernel(device, uuid, "mask_scale_u55c_kernel");
  auto softmax_full_row_kernel = xrt::kernel(device, uuid, "softmax_full_row_u55c_kernel");
  auto v_weighted_sum_kernel = xrt::kernel(device, uuid, "v_weighted_sum_u55c_kernel");

  std::vector<std::int8_t> q_tile(kQTileElems);
  std::vector<std::int8_t> k_tile(kKTileElems);
  std::vector<float> full_row_logits(kScoreRowsPerTile * kFullRowMaxCols, 0.0f);
  std::vector<float> full_row_probs(kScoreRowsPerTile * kFullRowMaxCols, 0.0f);
  std::vector<float> v_tile(kVTileElems, 0.0f);
  std::vector<float> weights_tile(kScoreTileElems, 0.0f);
  std::vector<float> attn_partial(kAttnOutTileElems, 0.0f);
  std::vector<std::int32_t> raw_got(seq_len * seq_len, 0);
  std::vector<float> logits_got(seq_len * seq_len, 0.0f);
  std::vector<float> softmax_got(seq_len * seq_len, 0.0f);
  std::vector<float> attn_out_got(seq_len * kHeadDim, 0.0f);

  std::vector<xrt::bo> q_bos;
  std::vector<xrt::bo> k_bos;
  std::vector<xrt::bo> raw_score_bos;
  std::vector<xrt::bo> scaled_score_bos;
  q_bos.reserve(2);
  k_bos.reserve(2);
  raw_score_bos.reserve(2);
  scaled_score_bos.reserve(2);
  for (std::size_t set = 0; set < 2; ++set) {
    q_bos.emplace_back(device, sizeof(std::int8_t) * kQTileElems, score_kernel.group_id(0));
    k_bos.emplace_back(device, sizeof(std::int8_t) * kKTileElems, score_kernel.group_id(1));
    raw_score_bos.emplace_back(device, sizeof(std::int32_t) * kScoreTileElems, score_kernel.group_id(2));
    scaled_score_bos.emplace_back(device, sizeof(float) * kScoreTileElems, mask_scale_kernel.group_id(1));
  }
  auto full_row_score_bo =
      xrt::bo(device, sizeof(float) * full_row_logits.size(), softmax_full_row_kernel.group_id(0));
  auto full_row_prob_bo =
      xrt::bo(device, sizeof(float) * full_row_probs.size(), softmax_full_row_kernel.group_id(1));
  auto weights_tile_bo =
      xrt::bo(device, sizeof(float) * weights_tile.size(), v_weighted_sum_kernel.group_id(0));
  auto v_tile_bo =
      xrt::bo(device, sizeof(float) * v_tile.size(), v_weighted_sum_kernel.group_id(1));
  auto attn_partial_bo =
      xrt::bo(device, sizeof(float) * attn_partial.size(), v_weighted_sum_kernel.group_id(2));

  using Clock = std::chrono::steady_clock;
  std::vector<TimedStage> timings = {
      {"attention_score_u55c_kernel", 0.0},
      {"mask_scale_u55c_kernel", 0.0},
      {"softmax_full_row_u55c_kernel", 0.0},
      {"v_weighted_sum_u55c_kernel", 0.0},
  };
  const auto chain_start = Clock::now();

  for (std::uint32_t query_base = 0; query_base < seq_len; query_base += kScoreRowsPerTile) {
    const std::uint32_t query_rows =
        std::min<std::uint32_t>(kScoreRowsPerTile, seq_len - query_base);
    std::fill(full_row_logits.begin(), full_row_logits.end(), 0.0f);

    copy_q_tile(q_full, seq_len, query_base, &q_tile);
    for (std::size_t set = 0; set < q_bos.size(); ++set) {
      q_bos[set].write(q_tile.data());
      q_bos[set].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    std::array<PendingMaskScale, 2> pending_masks;

    auto preload_k_tile = [&](std::size_t set, std::uint32_t key_base) {
      copy_k_tile(k_full, seq_len, key_base, &k_tile);
      k_bos[set].write(k_tile.data());
      k_bos[set].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };

    auto finish_pending_mask = [&](std::size_t set) {
      auto& pending = pending_masks[set];
      if (!pending.valid) {
        return;
      }

      pending.run->wait();
      const auto finish_time = Clock::now();
      timings[1].milliseconds +=
          std::chrono::duration<double, std::milli>(finish_time - pending.launch_time).count();

      const auto raw_tile = read_i32_bo(raw_score_bos[set], kScoreTileElems);
      const auto scaled_tile = read_float_bo(scaled_score_bos[set], kScoreTileElems);
      for (std::uint32_t row = 0; row < query_rows; ++row) {
        const std::size_t global_row_base = (query_base + row) * seq_len;
        const std::size_t tile_row_base = row * kScoreColsPerTile;
        const std::size_t full_row_base = row * kFullRowMaxCols;
        for (std::uint32_t col = 0; col < pending.key_cols; ++col) {
          raw_got[global_row_base + pending.key_base + col] = raw_tile[tile_row_base + col];
          logits_got[global_row_base + pending.key_base + col] = scaled_tile[tile_row_base + col];
          full_row_logits[full_row_base + pending.key_base + col] = scaled_tile[tile_row_base + col];
        }
      }

      pending.run.reset();
      pending.valid = false;
    };

    preload_k_tile(0, 0);

    for (std::uint32_t key_tile_idx = 0; key_tile_idx < k_chunks; ++key_tile_idx) {
      const std::uint32_t key_base =
          key_tile_idx * static_cast<std::uint32_t>(kScoreColsPerTile);
      const std::uint32_t key_cols =
          std::min<std::uint32_t>(kScoreColsPerTile, seq_len - key_base);
      const std::size_t set = key_tile_idx % 2U;
      finish_pending_mask(set);

      timings[0].milliseconds += run_timed("attention score tile", [&]() {
        return score_kernel(q_bos[set], k_bos[set], raw_score_bos[set], query_rows, key_cols);
      }, false);

      auto& pending = pending_masks[set];
      pending.valid = true;
      pending.key_base = key_base;
      pending.key_cols = key_cols;
      pending.launch_time = Clock::now();
      pending.run.emplace(mask_scale_kernel(
          raw_score_bos[set],
          scaled_score_bos[set],
          query_base,
          key_base,
          query_rows,
          key_cols,
          scale));

      const std::uint32_t next_key_tile_idx = key_tile_idx + 1U;
      if (next_key_tile_idx < k_chunks) {
        const std::uint32_t next_key_base =
            next_key_tile_idx * static_cast<std::uint32_t>(kScoreColsPerTile);
        preload_k_tile(next_key_tile_idx % 2U, next_key_base);
      }
    }

    finish_pending_mask(0);
    finish_pending_mask(1);

    full_row_score_bo.write(full_row_logits.data());
    full_row_score_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timings[2].milliseconds += run_timed("full-row softmax", [&]() {
      return softmax_full_row_kernel(
          full_row_score_bo,
          full_row_prob_bo,
          query_rows,
          seq_len);
    }, false);

    const auto prob_tile = read_float_bo(full_row_prob_bo, full_row_probs.size());
    std::vector<float> attn_accum(kAttnOutTileElems, 0.0f);
    for (std::uint32_t row = 0; row < query_rows; ++row) {
      const std::size_t global_row_base = (query_base + row) * seq_len;
      const std::size_t full_row_base = row * kFullRowMaxCols;
      for (std::uint32_t col = 0; col < seq_len; ++col) {
        softmax_got[global_row_base + col] = prob_tile[full_row_base + col];
      }
    }

    for (std::uint32_t key_base = 0; key_base < seq_len; key_base += kScoreColsPerTile) {
      const std::uint32_t key_cols =
          std::min<std::uint32_t>(kScoreColsPerTile, seq_len - key_base);
      std::fill(weights_tile.begin(), weights_tile.end(), 0.0f);
      for (std::uint32_t row = 0; row < query_rows; ++row) {
        const std::size_t weights_row_base = row * kScoreColsPerTile;
        const std::size_t full_row_base = row * kFullRowMaxCols;
        for (std::uint32_t col = 0; col < key_cols; ++col) {
          weights_tile[weights_row_base + col] = prob_tile[full_row_base + key_base + col];
        }
      }

      copy_v_tile(v_full, seq_len, key_base, &v_tile);
      weights_tile_bo.write(weights_tile.data());
      v_tile_bo.write(v_tile.data());
      weights_tile_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      v_tile_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      timings[3].milliseconds += run_timed("v weighted sum tile", [&]() {
        return v_weighted_sum_kernel(
            weights_tile_bo,
            v_tile_bo,
            attn_partial_bo,
            query_rows,
            key_cols);
      }, false);

      const auto partial_tile = read_float_bo(attn_partial_bo, attn_partial.size());
      for (std::uint32_t row = 0; row < query_rows; ++row) {
        const std::size_t tile_row_base = row * kHeadDim;
        for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
          attn_accum[tile_row_base + dim] += partial_tile[tile_row_base + dim];
        }
      }
    }

    for (std::uint32_t row = 0; row < query_rows; ++row) {
      const std::size_t global_row_base = (query_base + row) * kHeadDim;
      const std::size_t tile_row_base = row * kHeadDim;
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        attn_out_got[global_row_base + dim] = attn_accum[tile_row_base + dim];
      }
    }
  }

  const auto chain_stop = Clock::now();
  const double total_chain_ms =
      std::chrono::duration<double, std::milli>(chain_stop - chain_start).count();

  compare_exact(raw_got, raw_expected, "full_score_raw");
  compare_float(logits_got, logits_expected, "full_score_scaled", 1.0e-4f);
  compare_float(softmax_got, softmax_expected, "full_score_softmax", 1.0e-4f);
  compare_float(attn_out_got, attn_out_expected, "full_attn_out", attn_out_tolerance);

  print_timings(timings, total_chain_ms);
  std::cout << "Tiled sequence verification PASSED\n";
  std::cout << "Attention output verification PASSED\n";
  std::cout << "XRT chain verification PASSED\n";
  return 0;
}

int run_tiled_sequence(const Args& args, xrt::device& device, const xrt::uuid& uuid) {
  if (args.seq_len == 0) {
    throw std::runtime_error("--seq-len must be positive");
  }

  const std::uint32_t seq_len = args.seq_len;
  const auto q_full = make_synthetic_q(seq_len);
  const auto k_full = make_synthetic_k(seq_len);
  const auto v_full = make_synthetic_v(seq_len);
  const float scale = total_scale(args.q_scale, args.k_scale);

  std::vector<std::int32_t> raw_expected;
  std::vector<float> logits_expected;
  std::vector<float> softmax_expected;
  compute_cpu_reference(
      q_full,
      k_full,
      seq_len,
      scale,
      &raw_expected,
      &logits_expected,
      &softmax_expected);
  std::vector<float> attn_out_expected;
  compute_attn_out_reference(softmax_expected, v_full, seq_len, &attn_out_expected);

  return run_tiled_inputs(
      "tiled synthetic sequence",
      q_full,
      k_full,
      v_full,
      scale,
      raw_expected,
      logits_expected,
      softmax_expected,
      attn_out_expected,
      1.0e-4f,
      device,
      uuid);
}

int run_tiled_vector_sequence(const Args& args, xrt::device& device, const xrt::uuid& uuid) {
  const auto q_full = read_text_vector_all<std::int8_t>(args.vector_dir + "/q_full.txt");
  const auto k_full = read_text_vector_all<std::int8_t>(args.vector_dir + "/k_full.txt");
  const auto v_full = read_text_vector_all<float>(args.vector_dir + "/v_full.txt");
  const std::uint32_t seq_len = static_cast<std::uint32_t>(q_full.size() / kHeadDim);
  if (seq_len > kFullRowMaxCols) {
    throw std::runtime_error("Full-sequence vector directory exceeds current max seq_len 512");
  }

  const std::size_t score_elems = static_cast<std::size_t>(seq_len) * seq_len;
  const std::size_t attn_elems = static_cast<std::size_t>(seq_len) * kHeadDim;
  const auto raw_expected =
      read_text_vector<std::int32_t>(args.vector_dir + "/score_raw.txt", score_elems);
  const auto logits_expected =
      read_text_vector<float>(args.vector_dir + "/score_scaled.txt", score_elems);
  const auto softmax_expected =
      read_text_vector<float>(args.vector_dir + "/score_softmax.txt", score_elems);

  std::vector<float> attn_out_expected;
  float attn_out_tolerance = 1.0e-4f;
  if (file_exists(args.vector_dir + "/attn_out.txt")) {
    attn_out_expected = read_text_vector<float>(args.vector_dir + "/attn_out.txt", attn_elems);
  } else {
    attn_out_expected =
        read_text_vector<float>(args.vector_dir + "/attn_ref_float.txt", attn_elems);
    attn_out_tolerance = 1.0e-3f;
  }

  const auto meta = read_kernel_meta(args.vector_dir + "/kernel_meta.txt");
  return run_tiled_inputs(
      "tiled vector sequence from " + args.vector_dir,
      q_full,
      k_full,
      v_full,
      meta.total_scale,
      raw_expected,
      logits_expected,
      softmax_expected,
      attn_out_expected,
      attn_out_tolerance,
      device,
      uuid);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);

    std::cout << "Opening device " << args.device_index << "\n";
    auto device = xrt::device(args.device_index);
    auto uuid = device.load_xclbin(args.xclbin_path);

    if (args.seq_len != 0) {
      return run_tiled_sequence(args, device, uuid);
    }
    if (file_exists(args.vector_dir + "/q_full.txt") &&
        file_exists(args.vector_dir + "/k_full.txt")) {
      return run_tiled_vector_sequence(args, device, uuid);
    }
    return run_single_tile(args, device, uuid);
  } catch (const std::exception& ex) {
    std::cerr << "XRT host failed: " << ex.what() << "\n";
    return 1;
  }
}
