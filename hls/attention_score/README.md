# Attention Score HLS Flow

This folder contains the isolated U55C-oriented score kernel:

- `attention_score_core_hls.cpp`
- `attention_score_core_hls.hpp`
- `tb_attention_score.cpp`
- `run_hls.tcl`

## What `run_hls.tcl` Does

It runs two useful first-pass steps in Vitis HLS:

- `csim_design`
- `csynth_design`

The target device is set to the Alveo U55C-class part:

- `xcu55c-fsvh2892-2L-e`

This is enough to validate:

- the file-driven testbench
- the kernel interface shape
- whether the score-tile loop nests synthesize cleanly
