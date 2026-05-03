# Mask And Scale HLS Flow

This kernel is the current runtime stage after raw score GEMM:

```text
score_raw_int32 -> score_scaled_fp32
```

It merges two operations that used to be separate kernels:

- apply the decoder causal mask rule `key_pos <= query_pos`
- multiply unmasked scores by `total_scale`

The scale term is supplied by the host:

```text
total_scale = q_scale * k_scale * (1 / sqrt(64))
```

The verified XRT chain now launches:

```text
attention_score_u55c_kernel
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

The legacy standalone `hls/causal_mask/` and `hls/score_scale/` kernels are
kept for reference and local testing, but they are not part of the current
hardware runtime chain.
