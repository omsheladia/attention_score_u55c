#!/usr/bin/env bash

# Source this file before building or running the U55C attention-score demo.
# It keeps the toolchain aligned to the installed 2022.2/XRT stack.

source /opt/xilinx/xrt/setup.sh

if [[ -f /home/advent/Vivado/Vivado/2022.2/settings64.sh ]]; then
  source /home/advent/Vivado/Vivado/2022.2/settings64.sh
fi

if [[ -f /home/advent/Vivado/Vitis_HLS/2022.2/settings64.sh ]]; then
  source /home/advent/Vivado/Vitis_HLS/2022.2/settings64.sh
fi

if [[ -f /home/advent/Vivado/Vitis/2022.2/settings64.sh ]]; then
  source /home/advent/Vivado/Vitis/2022.2/settings64.sh
fi

export XILINX_VIVADO=/home/advent/Vivado/Vivado/2022.2
export XILINX_HLS=/home/advent/Vivado/Vitis_HLS/2022.2
export XILINX_VIVADO_HLS=/home/advent/Vivado/Vitis_HLS/2022.2

