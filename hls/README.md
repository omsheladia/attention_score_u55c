# HLS Workspace

This HLS workspace contains the kernels for the current U55C attention-score
runtime and the next Track B attention-output stage:

```text
attention_score_u55c_kernel
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

The tiled Track A path also uses `softmax_full_row_u55c_kernel` for `S > 64`.

The verified runtime path uses:

- `attention_score/`: INT8 score accumulation,
  `score_raw = Q_rot_int8 @ K_rot_int8^T`
- `mask_and_scale/`: merged causal mask and scale to FP32
- `softmax/`: row-wise softmax for one fixed `8 x 64` tile
- `softmax_full_row/`: Track A full-row softmax for `8 x S` rows with
  `S <= 512`; local C++ bench, Vitis HLS `csim/csynth`, `hw_emu`, and real
  U55C tiled runs pass
- `v_weighted_sum/`: Track B partial `softmax @ V` kernel for one `8 x 64`
  weights tile and one `64 x 64` V chunk

The older `causal_mask/` and `score_scale/` folders are preserved as standalone
legacy stages and local references, but they are no longer part of the current
XRT chain.
