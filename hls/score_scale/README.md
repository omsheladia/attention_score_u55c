# Score Scale HLS Flow

This folder contains the legacy standalone score-scale kernel:

`score_masked_int32 -> score_scaled_fp32`

It applies one host-supplied `total_scale` term:

`total_scale = q_scale * k_scale * (1 / sqrt(64))`

The current verified XRT runtime does not launch this kernel separately. Its
behavior is merged with causal masking in:

```text
hls/mask_and_scale/
```
