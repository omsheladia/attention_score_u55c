# Attention Score U55C Workspace

This folder is a focused mini-workspace for one block only:

`Q_rot x K_rot^T -> attention score tile`

It mirrors the main repo layout on purpose, but keeps only the pieces needed to
understand and prototype attention-score offload onto a Xilinx Alveo U55C.

## What Is In Scope

- Python reference math for one attention-score tile
- deterministic vector export for simulation and HLS C-sim
- a U55C-oriented HLS kernel skeleton for the score GEMM
- separate HLS kernels for causal masking and score scaling
- notes on where masking and scaling fit

## What Is Out Of Scope

- Q/K/V projection GEMMs
- RoPE generation
- softmax
- weighted sum with V
- runtime controller integration

## Offload Boundary

The clean hardware boundary is:

1. Host or an upstream kernel produces RoPE-applied `Q` and `K`.
2. One U55C kernel consumes:
   - `q_tile`: shape `(query_rows, 64)` as `int8`
   - `k_tile`: shape `(key_cols, 64)` as `int8`
3. The kernel computes:
   - `score_raw_int32[row, col] = sum_d q_tile[row, d] * k_tile[col, d]`
4. A follow-on stage applies:
   - causal mask
   - scaling by `q_scale * k_scale * 1/sqrt(head_dim)`
   - softmax

That is the same split already suggested by the main repo's Python exporters:
raw score accumulation is a clean INT8 x INT8 -> INT32 block, while masking and
softmax stay separate.

## Layout

- `docs/`: offload notes and mapping explanation
- `model/`: isolated Python reference and vector exporter
- `hls/`: HLS score-kernel skeleton and C-sim testbench
- `hls/causal_mask/`: follow-on mask kernel
- `hls/score_scale/`: follow-on scale kernel
- `hls/softmax/`: stage-4 softmax kernel
- `host/`: native XRT host app and Linux build helpers
- `sim/`: generated vectors for the isolated score path
- `rtl/`: notes for later RTL lowering

## Quick Start

Generate deterministic vectors:

```powershell
python attention_score_u55c/model/export_attention_score_vectors.py
```

Compile and run the HLS-style C-sim testbench with a normal C++ compiler:

```powershell
g++ -std=c++17 attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp `
  attention_score_u55c/hls/attention_score/tb_attention_score.cpp `
  -Iattention_score_u55c/hls/common -o attention_score_u55c/sim/tb_attention_score.exe

.\attention_score_u55c\sim\tb_attention_score.exe
```

## Why This Is A Good First U55C Cut

This block is easy to offload because:

- the interface is fixed-size and stream-friendly
- the math is dense MAC-heavy work
- accumulation is integer and deterministic
- masking, scaling, and softmax can stay outside the first kernel
- it matches the way the existing repo already exports score tiles
