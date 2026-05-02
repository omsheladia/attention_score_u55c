#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <platform.xpfm> [device_index]"
  exit 1
fi

PLATFORM="$1"
DEVICE_INDEX="${2:-0}"
BUILD_DIR="attention_score_u55c/build"

source "${XILINX_VITIS:-/tools/Xilinx/Vitis/2023.2}/settings64.sh" 2>/dev/null || true
source "${XILINX_XRT:-/opt/xilinx/xrt}/setup.sh" 2>/dev/null || true

python3 attention_score_u55c/model/export_attention_score_vectors.py
bash attention_score_u55c/host/build_xclbin.sh hw_emu "$PLATFORM"
bash attention_score_u55c/host/build_host.sh

if command -v emconfigutil >/dev/null 2>&1; then
  emconfigutil --platform "$PLATFORM" --nd 1
fi

export XCL_EMULATION_MODE=hw_emu

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin "$BUILD_DIR/attention_score_chain.xclbin" \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device "$DEVICE_INDEX"
