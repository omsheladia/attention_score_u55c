#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${XILINX_XRT:-}" ]]; then
  echo "XILINX_XRT is not set. Source the XRT setup script first."
  exit 1
fi

mkdir -p attention_score_u55c/build

g++ -O2 -std=c++17 \
  -I"$XILINX_XRT/include" \
  -L"$XILINX_XRT/lib" \
  -o attention_score_u55c/build/host_attention_score_chain \
  attention_score_u55c/host/attention_score_chain_xrt.cpp \
  -lxrt_coreutil -pthread

echo "Built attention_score_u55c/build/host_attention_score_chain"
