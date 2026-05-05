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

The current optimization branch also includes Track O2 resident full-buffer
variants:

```text
score_mask_scale_resident_u55c_kernel
-> softmax_full_row_resident_u55c_kernel
-> v_weighted_sum_resident_u55c_kernel
```

These resident kernels passed Vitis HLS `csim/csynth`, `hw_emu` smoke
verification at `S=8 --resident-debug`, and real U55C synthetic
`S=8,64,128,256,512` plus real-vector `S=16,64` validation. The resident path
reduced host/DMA/sync gap but regressed total runtime because resident
V accumulation performs slow HBM read-modify-write updates. See
`docs/track_d_results.md` and `docs/o2_*` reports for the current results.

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
  and real U55C tiled runs pass; this folder also contains the Track O2
  resident V kernel, whose current accumulation loop is the main performance
  bottleneck

The older `causal_mask/` and `score_scale/` folders are preserved as standalone
legacy stages and local references, but they are no longer part of the current
XRT chain.
