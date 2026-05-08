#!/usr/bin/env bash
# Source this file before the demo to get repeatable build/run helpers.
#
# Usage:
#   cd /home/advent/Desktop/RC19
#   source attention_score_u55c/host/demo_env.sh
#   demo_help

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This file is meant to be sourced, not executed:"
  echo "  source attention_score_u55c/host/demo_env.sh"
  exit 1
fi

export ATTENTION_SCORE_U55C_ROOT="${ATTENTION_SCORE_U55C_ROOT:-/home/advent/Desktop/RC19/attention_score_u55c}"
export ATTENTION_SCORE_U55C_PLATFORM="${ATTENTION_SCORE_U55C_PLATFORM:-/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm}"
export ATTENTION_SCORE_U55C_DEVICE="${ATTENTION_SCORE_U55C_DEVICE:-0}"
export ATTENTION_SCORE_U55C_XCLBIN="${ATTENTION_SCORE_U55C_XCLBIN:-${ATTENTION_SCORE_U55C_ROOT}/build/attention_score_chain.xclbin}"
export ATTENTION_SCORE_U55C_HW_EMU_XCLBIN="${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN:-${ATTENTION_SCORE_U55C_ROOT}/build/attention_score_chain_hw_emu.xclbin}"
export ATTENTION_SCORE_U55C_HOST="${ATTENTION_SCORE_U55C_HOST:-${ATTENTION_SCORE_U55C_ROOT}/build/host_attention_score_chain}"
export ATTENTION_SCORE_U55C_LOG_DIR="${ATTENTION_SCORE_U55C_LOG_DIR:-${ATTENTION_SCORE_U55C_ROOT}/demo_logs}"

if [[ -f "${ATTENTION_SCORE_U55C_ROOT}/host/setup_2022_2_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ATTENTION_SCORE_U55C_ROOT}/host/setup_2022_2_env.sh"
else
  echo "Warning: ${ATTENTION_SCORE_U55C_ROOT}/host/setup_2022_2_env.sh not found"
fi

unset XCL_EMULATION_MODE

demo_help() {
  cat <<'EOF'
U55C attention-score demo helpers

Environment:
  demo_status          Show branch, tool, platform, xclbin, and card status
  demo_card            Show U55C card visibility with xbutil
  demo_xclbin_info     Print xclbin metadata

Build:
  demo_build_host      Build host_attention_score_chain
  demo_build_hw_emu    Build hardware-emulation xclbin
  demo_build_hw        Build real hardware xclbin

Run:
  demo_run_tile        Run saved synthetic tile vectors on real card
  demo_run_hw_emu      Run S=8 in hardware emulation
  demo_sweep           Run synthetic sequence sweep: 8,64,128,256,512
  demo_real_vectors    Run real TinyLlama-derived vector directories

Useful variables:
  ATTENTION_SCORE_U55C_ROOT
  ATTENTION_SCORE_U55C_PLATFORM
  ATTENTION_SCORE_U55C_DEVICE
  ATTENTION_SCORE_U55C_XCLBIN
  ATTENTION_SCORE_U55C_HW_EMU_XCLBIN
  ATTENTION_SCORE_U55C_LOG_DIR
EOF
}

demo_status() {
  echo "Root:      ${ATTENTION_SCORE_U55C_ROOT}"
  echo "Platform:  ${ATTENTION_SCORE_U55C_PLATFORM}"
  echo "Device:    ${ATTENTION_SCORE_U55C_DEVICE}"
  echo "XCLBIN:    ${ATTENTION_SCORE_U55C_XCLBIN}"
  echo "HW_EMU:    ${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}"
  echo "Host:      ${ATTENTION_SCORE_U55C_HOST}"
  echo "Log dir:   ${ATTENTION_SCORE_U55C_LOG_DIR}"
  echo
  git -C "${ATTENTION_SCORE_U55C_ROOT}" branch --show-current
  git -C "${ATTENTION_SCORE_U55C_ROOT}" log --oneline -1
  echo
  v++ --version | sed -n '1,4p'
  echo
  test -f "${ATTENTION_SCORE_U55C_PLATFORM}" && echo "Platform found" || echo "Platform missing"
  test -f "${ATTENTION_SCORE_U55C_XCLBIN}" && echo "XCLBIN found" || echo "XCLBIN missing"
  test -f "${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}" && echo "HW_EMU XCLBIN found" || echo "HW_EMU XCLBIN missing"
}

demo_card() {
  xbutil examine
}

demo_xclbin_info() {
  xclbinutil --info --input "${ATTENTION_SCORE_U55C_XCLBIN}" | sed -n '1,180p'
}

demo_build_host() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  (
    cd "$(dirname "${ATTENTION_SCORE_U55C_ROOT}")"
    bash attention_score_u55c/host/build_host.sh
  ) 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/build_host.log"
}

demo_build_hw_emu() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  unset XCL_EMULATION_MODE
  (
    cd "$(dirname "${ATTENTION_SCORE_U55C_ROOT}")"
    bash attention_score_u55c/host/build_xclbin.sh hw_emu "${ATTENTION_SCORE_U55C_PLATFORM}"
    cp attention_score_u55c/build/attention_score_chain.xclbin \
      "${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}"
  ) 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/build_hw_emu.log"
  echo "Saved hardware-emulation xclbin to ${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}"
}

demo_build_hw() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  unset XCL_EMULATION_MODE
  (
    cd "$(dirname "${ATTENTION_SCORE_U55C_ROOT}")"
    bash attention_score_u55c/host/build_xclbin.sh hw "${ATTENTION_SCORE_U55C_PLATFORM}"
  ) 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/build_hw.log"
}

demo_run_hw_emu() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  if [[ ! -f "${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}" ]]; then
    echo "Missing hardware-emulation xclbin: ${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}"
    echo "Run demo_build_hw_emu first. Do not use the real hardware bitstream for hw_emu."
    return 1
  fi
  export XCL_EMULATION_MODE=hw_emu
  (
    cd "$(dirname "${ATTENTION_SCORE_U55C_ROOT}")"
    "${ATTENTION_SCORE_U55C_HOST}" \
      --xclbin "${ATTENTION_SCORE_U55C_HW_EMU_XCLBIN}" \
      --seq-len 8 \
      --device "${ATTENTION_SCORE_U55C_DEVICE}"
  ) 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/run_hw_emu_s8.log"
  unset XCL_EMULATION_MODE
}

demo_run_tile() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  unset XCL_EMULATION_MODE
  "${ATTENTION_SCORE_U55C_HOST}" \
    --xclbin "${ATTENTION_SCORE_U55C_XCLBIN}" \
    --vectors "${ATTENTION_SCORE_U55C_ROOT}/sim/attention_score_tile" \
    --device "${ATTENTION_SCORE_U55C_DEVICE}" \
    2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/run_fpga_attention_tile.log"
}

demo_sweep() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  unset XCL_EMULATION_MODE
  for s in 8 64 128 256 512; do
    echo "===== S=${s} ====="
    "${ATTENTION_SCORE_U55C_HOST}" \
      --xclbin "${ATTENTION_SCORE_U55C_XCLBIN}" \
      --seq-len "${s}" \
      --device "${ATTENTION_SCORE_U55C_DEVICE}"
  done 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/run_fpga_seq_sweep.log"
}

demo_real_vectors() {
  mkdir -p "${ATTENTION_SCORE_U55C_LOG_DIR}"
  unset XCL_EMULATION_MODE
  for d in \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_tile" \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_s16" \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_s64" \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_s128" \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_s256" \
    "${ATTENTION_SCORE_U55C_ROOT}/sim/real_tinyllama_s512"
  do
    echo "===== ${d} ====="
    "${ATTENTION_SCORE_U55C_HOST}" \
      --xclbin "${ATTENTION_SCORE_U55C_XCLBIN}" \
      --vectors "${d}" \
      --device "${ATTENTION_SCORE_U55C_DEVICE}"
  done 2>&1 | tee "${ATTENTION_SCORE_U55C_LOG_DIR}/run_fpga_real_vectors.log"
}

echo "Loaded U55C attention-score demo helpers. Run: demo_help"
