# Causal Mask HLS Flow

This folder contains the legacy standalone causal-mask kernel:

`score_raw_int32 -> score_masked_int32`

It consumes one fixed `8 x 64` score tile plus tile-position metadata and
applies the standard decoder causal rule `key_pos <= query_pos`.

The current verified XRT runtime does not launch this kernel separately. Its
behavior is merged with score scaling in:

```text
hls/mask_and_scale/
```
