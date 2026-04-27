#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hw|hw_emu> <platform.xpfm>"
  exit 1
fi

TARGET="$1"
PLATFORM="$2"
BUILD_DIR="attention_score_u55c/build"

mkdir -p "$BUILD_DIR"

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k attention_score_u55c_kernel \
  -o "$BUILD_DIR/attention_score_u55c_kernel.xo" \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k causal_mask_u55c_kernel \
  -o "$BUILD_DIR/causal_mask_u55c_kernel.xo" \
  attention_score_u55c/hls/causal_mask/causal_mask_core_hls.cpp

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k score_scale_u55c_kernel \
  -o "$BUILD_DIR/score_scale_u55c_kernel.xo" \
  attention_score_u55c/hls/score_scale/score_scale_core_hls.cpp

v++ -c -t "$TARGET" --platform "$PLATFORM" \
  -k softmax_u55c_kernel \
  -o "$BUILD_DIR/softmax_u55c_kernel.xo" \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp

v++ -l -t "$TARGET" --platform "$PLATFORM" \
  --config attention_score_u55c/host/vpp_link.cfg \
  -o "$BUILD_DIR/attention_score_chain.xclbin" \
  "$BUILD_DIR/attention_score_u55c_kernel.xo" \
  "$BUILD_DIR/causal_mask_u55c_kernel.xo" \
  "$BUILD_DIR/score_scale_u55c_kernel.xo" \
  "$BUILD_DIR/softmax_u55c_kernel.xo"

echo "Built $BUILD_DIR/attention_score_chain.xclbin"
