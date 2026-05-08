# HLS Workspace

This HLS workspace contains the kernels for the current staged U55C one-head
attention runtime:

```text
score_mask_scale_u55c_kernel
-> softmax_full_row_u55c_kernel
-> v_weighted_sum_u55c_kernel
```

The legacy single-tile path still uses `softmax_u55c_kernel`; the tiled Track A
path uses `softmax_full_row_u55c_kernel` for S-wide row normalization.

On the `track-a-step5-fused-score-mask-scale` branch, the xclbin build is wired
for the fused Track A Step 5 pre-softmax kernel. This branch has local C++
verification, but Vitis HLS, `hw_emu`, and real U55C verification are still
pending. The last fully routed/real-card reports in
`docs/artifacts/u55c_fused/reports/PostRoute*.rpt` belong to the previous
staged design.

The runtime path uses:

- `score_and_mask_scale/`: fused INT8 score accumulation plus causal mask and
  scale to FP32 logits; removes the raw-score HBM round trip in the main path
- `attention_score/`: legacy standalone INT8 score accumulation reference
- `mask_and_scale/`: legacy merged causal mask and scale reference
- `softmax/`: row-wise softmax for one fixed `8 x 64` tile
- `softmax_full_row/`: Track A full-row softmax for `8 x S` rows with
  `S <= 512`; local C++ bench, Vitis HLS `csim/csynth`, `hw_emu`, and real
  U55C tiled runs pass
- `v_weighted_sum/`: Track B partial `softmax @ V` kernel for one `8 x 64`
  weights tile and one `64 x 64` V chunk; local bench, Vitis HLS, `hw_emu`,
  and real U55C tiled runs pass

The older `causal_mask/` and `score_scale/` folders are preserved as standalone
legacy stages and local references, but they are no longer part of the current
XRT chain.
