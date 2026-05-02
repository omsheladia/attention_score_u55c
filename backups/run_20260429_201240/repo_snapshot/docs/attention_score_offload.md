# Attention Score Offload Plan

## Goal

Offload only the attention-score accumulation onto the U55C:

```text
scores_raw = Q_rot_int8 @ K_rot_int8^T
```

This keeps the first FPGA kernel small and stable while preserving the natural
attention pipeline.

## Source Mapping

The main repo already expresses the same block in two useful ways:

- `model/tinyllama.py`
  - reference attention math:
    `scores = (q @ k.transpose(0, 2, 1)) * attn_scale`
- `model/export_fpga_vectors.py`
  - FPGA-shaped score accumulation:
    `score_active = q_rot_full[...] @ k_rot_full[...].T`

This workspace lifts only that second form into a small, self-contained flow.

## Recommended U55C Split

### Host Side

The host, or an earlier compute stage, should prepare:

- one query tile of RoPE-applied `Q` vectors
- one key tile of RoPE-applied `K` vectors
- metadata:
  - `query_row_count <= 8`
  - `key_col_count <= 64`
  - `head_dim = 64`

The host then DMA-transfers these tiles to HBM and launches the score kernel.

### Kernel Side

The kernel should:

1. Read `q_tile[8][64]` and `k_tile[64][64]`
2. Compute each dot product independently
3. Accumulate into `int32`
4. Write `score_raw[8][64]`

That is the full contract for the first kernel.

## Next Two Kernels

This workspace now includes the next two natural follow-on kernels:

1. `causal_mask_u55c_kernel`
   - input: `score_raw_int32`
   - output: `score_masked_int32`
2. `score_scale_u55c_kernel`
   - input: `score_masked_int32`
   - output: `score_scaled_fp32`

That gives you a clean three-stage chain:

```text
Q_rot_int8, K_rot_int8
-> score_raw_int32
-> score_masked_int32
-> score_scaled_fp32
```

## Why Not Push Softmax Into The Same First Kernel

Softmax is not the expensive part you want to validate first. The score GEMM is:

- larger arithmetic density
- easier to pipeline
- easier to verify
- easier to map to DSP-heavy fabric

Keeping softmax out of the first kernel also avoids mixing:

- integer accumulation
- causal masking rules
- floating-point or fixed-point exponentials

That separation reduces bring-up risk.

## Suggested Data Types

- `Q_rot`: `int8`
- `K_rot`: `int8`
- `score_raw`: `int32`
- optional later score scaling: `float32` or `Q16.16`

If `Q` and `K` were quantized with scales `s_q` and `s_k`, then:

```text
score_fp = score_raw * s_q * s_k * (1 / sqrt(64))
```

Since `sqrt(64) = 8`, the attention scaling term is `0.125`.

## U55C Kernel Shape

The U55C is a good fit because this kernel is:

- tile-based
- bandwidth-friendly
- MAC-dense
- free of long control dependencies

At a practical level, the kernel can be organized as:

- AXI master read of `q_tile`
- AXI master read of `k_tile`
- on-chip local buffers
- pipelined nested loops for dot products
- AXI master write of `score_raw`

## First Integration Milestone

The cleanest first milestone is:

1. Generate deterministic Python vectors for one score tile.
2. Match them in HLS C-sim.
3. Synthesize the kernel.
4. Later connect the kernel into the wider TinyLlama dataflow.

That is exactly what this new workspace is set up to support.
