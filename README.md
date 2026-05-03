# Attention Score U55C Workspace

This folder is a focused U55C mini-workspace for the attention-score block:

```text
Q_rot_int8, K_rot_int8
-> score_raw_int32
-> score_scaled_fp32
-> score_softmax_fp32
```

It mirrors the main repo layout on purpose, but keeps only the pieces needed to
understand, build, and verify an isolated attention-score offload on a Xilinx
Alveo U55C.

The current state is a verified tile-level FPGA demo: local C++ tests, HLS
`csim`/`csynth`, hardware emulation, and a real U55C hardware run have passed.
This is not yet a full TinyLlama runtime or tokens/sec benchmark.

## What Is In Scope

- Python reference math for one attention-score tile
- deterministic vector export for simulation and HLS C-sim
- U55C-oriented HLS kernels for:
  - INT8 score GEMM
  - merged causal mask + score scaling
  - row-wise softmax
- native XRT host app for the verified three-kernel chain
- Linux helpers for local C++ checks, HLS/Vitis builds, hardware emulation, and
  real-card execution
- HBM bank mapping for the chain buffers

## What Is Out Of Scope

- Q/K/V projection GEMMs
- RoPE generation
- weighted sum with V
- runtime controller integration
- full TinyLlama model execution
- multi-sequence-length benchmark sweeps
- model-level tokens/sec metrics

## Offload Boundary

The clean hardware boundary is:

1. Host or an upstream kernel produces RoPE-applied `Q` and `K`.
2. One U55C kernel consumes:
   - `q_tile`: shape `(query_rows, 64)` as `int8`
   - `k_tile`: shape `(key_cols, 64)` as `int8`
3. The kernel computes:
   - `score_raw_int32[row, col] = sum_d q_tile[row, d] * k_tile[col, d]`
4. Follow-on U55C kernels apply:
   - merged causal mask and scaling by `q_scale * k_scale * 1/sqrt(head_dim)`
   - softmax

That is the same split already suggested by the main repo's Python exporters:
raw score accumulation is a clean INT8 x INT8 -> INT32 block, while the current
runtime keeps mask+scale and softmax as separate kernels.

## Layout

- `docs/`: offload notes, mapping explanation, and exact FPGA run commands
- `model/`: isolated Python reference and vector exporter
- `hls/attention_score/`: INT8 score GEMM kernel and testbench
- `hls/mask_and_scale/`: current merged mask+scale kernel
- `hls/causal_mask/`: legacy standalone mask kernel
- `hls/score_scale/`: legacy standalone scale kernel
- `hls/softmax/`: current tile softmax kernel
- `host/`: native XRT host app and Linux build helpers
- `sim/`: generated vectors for the isolated score path
- `rtl/`: notes for later RTL lowering
- `backups/`: local preservation backups for known-good hardware runs

## Current Verification

The verified real-card command is:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

Or use the helper:

```bash
bash attention_score_u55c/host/run_hw.sh 0
```

The host app verifies FPGA outputs against the exported reference vectors:

- `score_raw.txt`: exact integer compare
- `score_scaled.txt`: float compare with `1.0e-4` tolerance
- `score_softmax.txt`: float compare with `1.0e-4` tolerance

Expected pass signal:

```text
XRT chain verification PASSED
```

The current verified real-card helper run for the three-kernel chain printed:

```text
attention_score_u55c_kernel 0.043 ms
mask_scale_u55c_kernel      0.025 ms
softmax_u55c_kernel         0.086 ms
total_chain                 0.159 ms
```

The preservation backup for the known-good hardware run is:

```text
backups/run_20260429_201240/
```

## Quick Start

From the parent repo root, generate vectors and run the local C++ checks:

```bash
bash attention_score_u55c/host/run_local_csim.sh
```

Build the XRT host:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
bash attention_score_u55c/host/build_host.sh
```

Build hardware emulation:

```bash
bash attention_score_u55c/host/build_xclbin.sh \
  hw_emu \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

Build real hardware:

```bash
bash attention_score_u55c/host/build_xclbin.sh \
  hw \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

For the full command runbook, see:

```text
docs/fpga_run_commands.md
```

## Why This Is A Good First U55C Cut

This block is easy to offload because:

- the interface is fixed-size and stream-friendly
- the math is dense MAC-heavy work
- accumulation is integer and deterministic
- mask+scale and softmax are clean separate stages
- it matches the way the existing repo already exports score tiles
- the chain already maps buffers across multiple HBM banks

## Next Work

- Add Python full-sequence tiling with `softmax_full_rows(logits)`
- Add a full-row softmax kernel/design for `S > 64`
- Add the matching XRT host tiling loop for `S = 8, 64, 128, 256, 512`
- Add multi-vector regression and CPU/GPU/FPGA timing tables
- Implement `softmax @ V`
- Integrate toward a real TinyLlama attention subgraph
