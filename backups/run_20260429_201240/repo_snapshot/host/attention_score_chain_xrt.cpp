#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
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
constexpr std::size_t kQTileElems = kScoreRowsPerTile * kHeadDim;
constexpr std::size_t kKTileElems = kScoreColsPerTile * kHeadDim;
constexpr std::size_t kScoreTileElems = kScoreRowsPerTile * kScoreColsPerTile;

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
    } else {
      throw std::runtime_error(
          "Usage: host_attention_score_chain --xclbin <path> [--vectors <dir>] [--device <idx>]");
    }
  }

  if (args.xclbin_path.empty()) {
    throw std::runtime_error("Missing required --xclbin argument");
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);

    const auto q_tile = read_text_vector<std::int8_t>(args.vector_dir + "/q_tile.txt", kQTileElems);
    const auto k_tile = read_text_vector<std::int8_t>(args.vector_dir + "/k_tile.txt", kKTileElems);
    const auto score_raw_expected =
        read_text_vector<std::int32_t>(args.vector_dir + "/score_raw.txt", kScoreTileElems);
    const auto score_masked_expected =
        read_text_vector<std::int32_t>(args.vector_dir + "/score_masked.txt", kScoreTileElems);
    const auto score_scaled_expected =
        read_text_vector<float>(args.vector_dir + "/score_scaled.txt", kScoreTileElems);
    const auto score_softmax_expected =
        read_text_vector<float>(args.vector_dir + "/score_softmax.txt", kScoreTileElems);
    const auto meta = read_kernel_meta(args.vector_dir + "/kernel_meta.txt");

    std::cout << "Opening device " << args.device_index << "\n";
    auto device = xrt::device(args.device_index);
    auto uuid = device.load_xclbin(args.xclbin_path);

    auto score_kernel = xrt::kernel(device, uuid, "attention_score_u55c_kernel");
    auto mask_kernel = xrt::kernel(device, uuid, "causal_mask_u55c_kernel");
    auto scale_kernel = xrt::kernel(device, uuid, "score_scale_u55c_kernel");
    auto softmax_kernel = xrt::kernel(device, uuid, "softmax_u55c_kernel");

    auto q_bo = xrt::bo(device, sizeof(std::int8_t) * q_tile.size(), score_kernel.group_id(0));
    auto k_bo = xrt::bo(device, sizeof(std::int8_t) * k_tile.size(), score_kernel.group_id(1));
    auto raw_score_bo =
        xrt::bo(device, sizeof(std::int32_t) * score_raw_expected.size(), score_kernel.group_id(2));
    auto masked_score_bo =
        xrt::bo(device, sizeof(std::int32_t) * score_masked_expected.size(), mask_kernel.group_id(1));
    auto scaled_score_bo =
        xrt::bo(device, sizeof(float) * score_scaled_expected.size(), scale_kernel.group_id(1));
    auto softmax_prob_bo =
        xrt::bo(device, sizeof(float) * score_softmax_expected.size(), softmax_kernel.group_id(1));

    q_bo.write(q_tile.data());
    k_bo.write(k_tile.data());
    q_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    k_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Running attention score kernel\n";
    auto score_run = score_kernel(
        q_bo,
        k_bo,
        raw_score_bo,
        meta.query_row_count,
        meta.key_col_count);
    score_run.wait();

    std::cout << "Running causal mask kernel\n";
    auto mask_run = mask_kernel(
        raw_score_bo,
        masked_score_bo,
        meta.query_pos_base,
        meta.key_pos_base,
        meta.query_row_count,
        meta.key_col_count);
    mask_run.wait();

    std::cout << "Running score scale kernel\n";
    auto scale_run = scale_kernel(masked_score_bo, scaled_score_bo, meta.total_scale);
    scale_run.wait();

    std::cout << "Running softmax kernel\n";
    auto softmax_run = softmax_kernel(scaled_score_bo, softmax_prob_bo, meta.query_row_count, meta.key_col_count);
    softmax_run.wait();

    raw_score_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    masked_score_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    scaled_score_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    softmax_prob_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::vector<std::int32_t> score_raw_got(score_raw_expected.size());
    std::vector<std::int32_t> score_masked_got(score_masked_expected.size());
    std::vector<float> score_scaled_got(score_scaled_expected.size());
    std::vector<float> score_softmax_got(score_softmax_expected.size());

    raw_score_bo.read(score_raw_got.data());
    masked_score_bo.read(score_masked_got.data());
    scaled_score_bo.read(score_scaled_got.data());
    softmax_prob_bo.read(score_softmax_got.data());

    compare_exact(score_raw_got, score_raw_expected, "score_raw");
    compare_exact(score_masked_got, score_masked_expected, "score_masked");
    compare_float(score_scaled_got, score_scaled_expected, "score_scaled", 1.0e-4f);
    compare_float(score_softmax_got, score_softmax_expected, "score_softmax", 1.0e-4f);

    std::cout << "XRT chain verification PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "XRT host failed: " << ex.what() << "\n";
    return 1;
  }
}
