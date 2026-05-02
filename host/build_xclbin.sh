#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hw|hw_emu> <platform.xpfm>"
  exit 1
fi

TARGET="$1"
PLATFORM="$2"
BUILD_DIR="attention_score_u55c/build"

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

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k attention_score_u55c_kernel \
  -o "$BUILD_DIR/attention_score_u55c_kernel.xo" \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k mask_scale_u55c_kernel \
  -o "$BUILD_DIR/mask_scale_u55c_kernel.xo" \
  attention_score_u55c/hls/mask_and_scale/mask_scale_core_hls.cpp

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k softmax_u55c_kernel \
  -o "$BUILD_DIR/softmax_u55c_kernel.xo" \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp

v++ -l -t "$TARGET" --platform "$PLATFORM" \
  --config attention_score_u55c/host/vpp_link.cfg \
  -o "$BUILD_DIR/attention_score_chain.xclbin" \
  "$BUILD_DIR/attention_score_u55c_kernel.xo" \
  "$BUILD_DIR/mask_scale_u55c_kernel.xo" \
  "$BUILD_DIR/softmax_u55c_kernel.xo"

echo "Built $BUILD_DIR/attention_score_chain.xclbin"
