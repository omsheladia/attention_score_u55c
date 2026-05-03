# Attention Score U55C Workspace

This folder is a focused U55C mini-workspace for the attention-score block:

```text
Q_rot_int8, K_rot_int8
-> score_raw_int32
-> score_scaled_fp32
-> score_softmax_fp32
-> attn_out_fp32 (`--seq-len` synthetic Track B path in hw_emu and real hw)
```

It mirrors the main repo layout on purpose, but keeps only the pieces needed to
understand, build, and verify an isolated attention-score offload on a Xilinx
Alveo U55C.

The current state is a verified tiled FPGA attention-score/softmax demo: local
C++ tests, HLS `csim`/`csynth`, hardware emulation, and real U55C hardware runs
have passed. Track A Steps 1-4 are complete for the staged design. Track B
Steps 1-3 are complete for synthetic `--seq-len` hardware emulation and real
U55C hardware: Python `softmax @ V` reference/export, standalone
`hls/v_weighted_sum/`, five-kernel xclbin integration, and final `attn_out`
verification all pass. Track C real TinyLlama vectors now run in both the legacy
single-tile path and the tiled full-sequence vector path at `S=16` and `S=64`.
This is still not a full TinyLlama runtime or model-level tokens/sec benchmark.

## What Is In Scope

- Python reference math for one attention-score tile
- Python full-sequence tiled score/softmax reference checks
- deterministic vector export for simulation and HLS C-sim
- TinyLlama setup, Q/K/V extraction, quantization, and real-vector export
- U55C-oriented HLS kernels for:
  - INT8 score GEMM
  - merged causal mask + score scaling
  - row-wise softmax
  - full-row softmax up to `S = 512`
  - Track B partial `softmax @ V` weighted-sum kernel
- native XRT host app for the verified score/softmax chain
- tiled XRT `hw_emu` path using full-row softmax and V weighted sum for
  synthetic sequence lengths
- Linux helpers for local C++ checks, HLS/Vitis builds, hardware emulation, and
  real-card execution
- HBM bank mapping for the chain buffers

## What Is Out Of Scope

- Q/K/V projection GEMMs
- RoPE generation
- runtime controller integration
- full TinyLlama model execution
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
   - `softmax @ V` weighted-sum accumulation for one attention head

That is the same split already suggested by the main repo's Python exporters:
raw score accumulation is a clean INT8 x INT8 -> INT32 block, while the current
runtime keeps mask+scale, softmax, and V weighted sum as staged kernels.

## Layout

- `docs/`: offload notes, mapping explanation, and exact FPGA run commands
- `model/`: isolated Python reference, vector exporter, and TinyLlama setup checker
- `hls/attention_score/`: INT8 score GEMM kernel and testbench
- `hls/mask_and_scale/`: current merged mask+scale kernel
- `hls/causal_mask/`: legacy standalone mask kernel
- `hls/score_scale/`: legacy standalone scale kernel
- `hls/softmax/`: current tile softmax kernel
- `hls/softmax_full_row/`: full-row softmax kernel for tiled `S <= 512`
- `hls/v_weighted_sum/`: Track B partial `softmax @ V` kernel
- `host/`: native XRT host app and Linux build helpers
- `sim/`: generated synthetic and real-vector cases for the staged attention path
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
- `attn_out.txt` or `attn_ref_float.txt` when `v_full.txt` is present

Expected pass signal:

```text
Attention output verification PASSED
XRT chain verification PASSED
```

The current verified synthetic-vector helper run for the five-kernel chain
printed:

```text
attention_score_u55c_kernel 0.066 ms
mask_scale_u55c_kernel      0.081 ms
softmax_u55c_kernel         0.117 ms
v_weighted_sum_u55c_kernel  0.041 ms
total_chain                 0.345 ms
Attention output verification PASSED
XRT chain verification PASSED
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
attention_score_u55c_kernel 0.053 ms
mask_scale_u55c_kernel      0.094 ms
softmax_u55c_kernel         0.029 ms
v_weighted_sum_u55c_kernel  0.041 ms
total_chain                 0.258 ms
Attention output verification PASSED
XRT chain verification PASSED
```

Real TinyLlama full-sequence vector directories have also been exported and
verified on the real U55C through the tiled five-kernel path:

| directory | S | q_chunks | k_chunks | total_chain ms | pass signal |
|---|---:|---:|---:|---:|---|
| `sim/real_tinyllama_s16` | 16 | 2 | 1 | 0.985 | `Attention output verification PASSED` |
| `sim/real_tinyllama_s64` | 64 | 8 | 1 | 2.510 | `Attention output verification PASSED` |

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

Latest Track B Step 3 `hw_emu` verification on 2026-05-03 used a five-kernel
xclbin containing `attention_score_u55c_kernel`, `mask_scale_u55c_kernel`,
`softmax_u55c_kernel`, `softmax_full_row_u55c_kernel`, and
`v_weighted_sum_u55c_kernel`. It passed final `attn_out` verification:

| S | q_chunks | k_chunks | total_chain ms | pass signal |
|---|----------|----------|----------------|-------------|
| 8 | 1 | 1 | 65,214.673 | `Attention output verification PASSED` |
| 64 | 8 | 1 | 580,737.286 | `Attention output verification PASSED` |
| 128 | 16 | 2 | 1,657,498.336 | `Attention output verification PASSED` |

These are simulator-dominated hardware-emulation timings, not hardware
performance numbers.

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

| S | total ms | pass signal |
|---|----------|-------------|
| 8 | 0.868 | `Attention output verification PASSED` |
| 64 | 2.328 | `Attention output verification PASSED` |
| 128 | 7.226 | `Attention output verification PASSED` |
| 256 | 21.269 | `Attention output verification PASSED` |
| 512 | 84.190 | `Attention output verification PASSED` |

This five-kernel run produces one-head `attn_out` for synthetic Q/K/V inputs.
It is still not model tokens/sec; tokens/sec requires real Q/K/V sequence
coverage and decoder-loop integration.

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

Export a real TinyLlama single-tile vector case for the legacy vector flow:

```bash
python model/export_real_vectors.py --local-files-only
```

This writes `sim/real_tinyllama_tile/`, which has been verified on the real U55C
with `--vectors sim/real_tinyllama_tile`.

Export real TinyLlama full-sequence vector cases for the tiled vector flow:

```bash
python model/export_real_vectors.py \
  --seq-len 16 \
  --output-dir sim/real_tinyllama_s16 \
  --text "In a small laboratory, engineers compare attention kernels across hardware targets. The experiment records tokens, latency, and numerical accuracy for each sequence length before the final report is written."

python model/export_real_vectors.py \
  --seq-len 64 \
  --output-dir sim/real_tinyllama_s64 \
  --text "In a small laboratory, engineers compare attention kernels across hardware targets. The experiment records tokens, latency, and numerical accuracy for each sequence length before the final report is written. A second paragraph adds enough context for a longer TinyLlama prompt, describing how query, key, and value tensors move through the FPGA pipeline while software baselines measure the same attention head for validation."
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

- Use `docs/track_d_results.md` for the current CPU/GPU/FPGA comparison
- Integrate toward a real TinyLlama attention subgraph
