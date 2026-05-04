#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hw|hw_emu> <platform.xpfm>"
  exit 1
fi

TARGET="$1"
PLATFORM="$2"
BUILD_DIR="attention_score_u55c/build"
VPP_COMMON_FLAGS=(--save-temps)

if ! command -v v++ >/dev/null 2>&1; then
  echo "v++ is not on PATH. Install/source full Vitis 2022.2 before building xclbin."
  exit 1
fi

if v++ --version 2>&1 | grep -q "v++ v (64-bit)"; then
  echo "Found a versionless v++ wrapper. This is usually the PetaLinux XSCT copy,"
  echo "not the full Vitis compiler. Install/source full Vitis 2022.2."
  exit 1
fi

mkdir -p "$BUILD_DIR"

v++ "${VPP_COMMON_FLAGS[@]}" -c -t "$TARGET" --platform "$PLATFORM" \
  -k score_mask_scale_u55c_kernel \
  -o "$BUILD_DIR/score_mask_scale_u55c_kernel.xo" \
  attention_score_u55c/hls/score_and_mask_scale/score_mask_scale_core_hls.cpp

v++ "${VPP_COMMON_FLAGS[@]}" -c -t "$TARGET" --platform "$PLATFORM" \
  -k softmax_u55c_kernel \
  -o "$BUILD_DIR/softmax_u55c_kernel.xo" \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp

v++ "${VPP_COMMON_FLAGS[@]}" -c -t "$TARGET" --platform "$PLATFORM" \
  -k softmax_full_row_u55c_kernel \
  -o "$BUILD_DIR/softmax_full_row_u55c_kernel.xo" \
  attention_score_u55c/hls/softmax_full_row/softmax_full_row_hls.cpp

v++ "${VPP_COMMON_FLAGS[@]}" -c -t "$TARGET" --platform "$PLATFORM" \
  -k v_weighted_sum_u55c_kernel \
  -o "$BUILD_DIR/v_weighted_sum_u55c_kernel.xo" \
  attention_score_u55c/hls/v_weighted_sum/v_weighted_sum_core_hls.cpp

v++ "${VPP_COMMON_FLAGS[@]}" -l -t "$TARGET" --platform "$PLATFORM" \
  --config attention_score_u55c/host/vpp_link.cfg \
  -o "$BUILD_DIR/attention_score_chain.xclbin" \
  "$BUILD_DIR/score_mask_scale_u55c_kernel.xo" \
  "$BUILD_DIR/softmax_u55c_kernel.xo" \
  "$BUILD_DIR/softmax_full_row_u55c_kernel.xo" \
  "$BUILD_DIR/v_weighted_sum_u55c_kernel.xo"

echo "Built $BUILD_DIR/attention_score_chain.xclbin"
