# Attention Score U55C Workspace

This folder is a focused U55C mini-workspace for the attention-score block:

```text
Q_rot_int8, K_rot_int8, V_fp32
-> score_raw_int32
-> score_scaled_fp32
-> score_softmax_fp32
-> attn_out_fp32          (Track B — pending lab PC verification)
```

It mirrors the main repo layout on purpose, but keeps only the pieces needed to
understand, build, and verify an isolated attention-score offload on a Xilinx
Alveo U55C.

The current state is a verified tile-level FPGA demo: local C++ tests, HLS
`csim`/`csynth`, hardware emulation, and a real U55C hardware run have passed.
This is not yet a full TinyLlama runtime or tokens/sec benchmark. Track A Step
3 Part A, the Python full-sequence tiled score/softmax reference, is now
implemented and verified for `S = 8, 64, 128, 256, 512`; the tiled XRT
`hw_emu` path is verified for `S = 8, 64, 128`, and the real U55C tiled sweep
is verified for `S = 8, 64, 128, 256, 512`.

## What Is In Scope

- Python reference math for one attention-score tile
- Python full-sequence tiled score/softmax reference checks
- deterministic vector export for simulation and HLS C-sim
- TinyLlama setup check for later real Q/K/V extraction
- U55C-oriented HLS kernels for:
  - INT8 score GEMM
  - merged causal mask + score scaling
  - row-wise softmax
  - full-row softmax up to `S = 512` for the next tiled host path
- native XRT host app for the verified three-kernel chain
- tiled XRT `hw_emu` path using full-row softmax for synthetic sequence lengths
- Linux helpers for local C++ checks, HLS/Vitis builds, hardware emulation, and
  real-card execution
- HBM bank mapping for the chain buffers

## What Is Out Of Scope

- Q/K/V projection GEMMs
- RoPE generation
- softmax @ V accumulation across full sequence (Track B code done, lab PC pending)
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
- `model/`: isolated Python reference, vector exporter, and TinyLlama setup checker
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

The verified synthetic-vector real-card command is:

```bash
source host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --vectors sim/attention_score_tile \
  --device 0
```

Or use the helper:

```bash
bash host/run_hw.sh 0
```

The host app verifies FPGA outputs against the exported reference vectors:

- `score_raw.txt`: exact integer compare
- `score_scaled.txt`: float compare with `1.0e-4` tolerance
- `score_softmax.txt`: float compare with `1.0e-4` tolerance

Expected pass signal:

```text
XRT chain verification PASSED
```

The current verified synthetic-vector helper run for the three-kernel chain
printed:

```text
attention_score_u55c_kernel 0.043 ms
mask_scale_u55c_kernel      0.025 ms
softmax_u55c_kernel         0.086 ms
total_chain                 0.159 ms
```

The real TinyLlama-derived single-tile vector directory has also been verified
on the real U55C:

```bash
bash host/run_hw.sh \
  0 \
  build/attention_score_chain.xclbin \
  sim/real_tinyllama_tile
```

That helper run printed:

```text
attention_score_u55c_kernel 0.062 ms
mask_scale_u55c_kernel      0.024 ms
softmax_u55c_kernel         0.028 ms
total_chain                 0.121 ms
XRT chain verification PASSED
```

The preservation backup for the known-good hardware run is:

```text
backups/run_20260429_201240/
```

## Quick Start

From the repo root, generate vectors and run the local C++ checks:

```bash
bash host/run_local_csim.sh
```

Run the full-sequence Python tiling check:

```bash
python model/attention_score_ref.py --check-full-tiling
```

Run the full-row softmax local C++ bench:

```bash
g++ -O2 -std=c++17 \
  hls/softmax_full_row/softmax_full_row_hls.cpp \
  hls/softmax_full_row/tb_softmax_full_row.cpp \
  -Ihls/common \
  -o sim/tb_softmax_full_row
sim/tb_softmax_full_row
```

Run the tiled synthetic hardware-emulation path:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
export XCL_EMULATION_MODE=hw_emu

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --seq-len 128 \
  --device 0
```

Run the real-card synthetic sequence sweep:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

for s in 8 64 128 256 512; do
  ./attention_score_u55c/build/host_attention_score_chain \
    --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
    --seq-len "$s" \
    --device 0
done
```

Latest real-card sweep after the Track A Step 4 double-buffered host pass:

| S | total ms | tiles/sec | scores/sec |
|---|----------|-----------|------------|
| 8 | 0.510 | 1,960.78 | 125,490.20 |
| 64 | 2.307 | 3,467.71 | 1,775,465.97 |
| 128 | 5.238 | 6,109.20 | 3,127,911.42 |
| 256 | 11.709 | 10,931.76 | 5,597,062.09 |
| 512 | 35.992 | 14,225.38 | 7,283,396.31 |

The Step 4 host path double-buffers the pass-1 score/mask-scale BO sets. These
are attention-score/softmax metrics, not model tokens/sec; tokens/sec still
requires Track B `softmax @ V` and decoder-loop integration.

Build the XRT host:

```bash
source host/setup_2022_2_env.sh
bash host/build_host.sh
```

Build hardware emulation:

```bash
bash host/build_xclbin.sh \
  hw_emu \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

Build real hardware:

```bash
bash host/build_xclbin.sh \
  hw \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

Check TinyLlama Python setup for Track C real-vector work:

```bash
python model/check_tinyllama_setup.py
```

Expected pass signal:

```text
TinyLlama forward pass OK
```

Export a real TinyLlama single-tile vector case for the current 3-kernel chain:

```bash
python model/export_real_vectors.py --local-files-only
```

This writes `sim/real_tinyllama_tile/`, which has been verified on the real U55C
with `--vectors sim/real_tinyllama_tile`.

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

- Add CPU/GPU/FPGA comparison tables
- Implement `softmax @ V`
- Extend real TinyLlama vectors beyond the current 8-token single-tile case
- Integrate toward a real TinyLlama attention subgraph
