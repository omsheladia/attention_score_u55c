# HLS Workspace

This HLS workspace contains the kernels for the current single-tile U55C
attention-score runtime:

```text
attention_score_u55c_kernel
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

The verified runtime path uses:

- `attention_score/`: INT8 score accumulation,
  `score_raw = Q_rot_int8 @ K_rot_int8^T`
- `mask_and_scale/`: merged causal mask and scale to FP32
- `softmax/`: row-wise softmax for one fixed `8 x 64` tile
- `softmax_full_row/`: Track A full-row softmax for `8 x S` rows with
  `S <= 512`; local C++ bench and Vitis HLS `csim/csynth` pass, but it is not
  yet wired into the XRT chain

The older `causal_mask/` and `score_scale/` folders are preserved as standalone
legacy stages and local references, but they are no longer part of the current
XRT chain.
