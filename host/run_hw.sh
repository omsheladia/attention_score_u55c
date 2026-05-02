#!/usr/bin/env bash
set -euo pipefail

DEVICE_INDEX="${1:-0}"
XCLBIN_PATH="${2:-attention_score_u55c/build/attention_score_chain.xclbin}"
VECTOR_DIR="${3:-attention_score_u55c/sim/attention_score_tile}"
HOST_BIN="attention_score_u55c/build/host_attention_score_chain"

if [[ ! -f "attention_score_u55c/host/setup_2022_2_env.sh" ]]; then
  echo "Run this script from the repository root: /home/advent/Desktop/RC19"
  exit 1
fi

source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

if [[ ! -x "$HOST_BIN" ]]; then
  echo "Missing host binary: $HOST_BIN"
  echo "Build it with: bash attention_score_u55c/host/build_host.sh"
  exit 1
fi

if [[ ! -f "$XCLBIN_PATH" ]]; then
  echo "Missing xclbin: $XCLBIN_PATH"
  echo "Build or restore the verified xclbin before running hardware."
  exit 1
fi

if [[ ! -d "$VECTOR_DIR" ]]; then
  echo "Missing vector directory: $VECTOR_DIR"
  echo "Regenerate vectors with: python3 attention_score_u55c/model/export_attention_score_vectors.py"
  exit 1
fi

"$HOST_BIN" \
  --xclbin "$XCLBIN_PATH" \
  --vectors "$VECTOR_DIR" \
  --device "$DEVICE_INDEX"
