# Fused Score And Mask/Scale Kernel

`score_mask_scale_u55c_kernel` is the Track A Step 5 fused pre-softmax kernel.
It consumes the same packed INT8 Q/K tiles as `attention_score_u55c_kernel`, but
writes the scaled FP32 logits tile directly.

This replaces the staged path:

```text
attention_score_u55c_kernel -> raw-score HBM -> mask_scale_u55c_kernel
```

with:

```text
score_mask_scale_u55c_kernel
```

The raw INT32 score tile is no longer written to HBM in the main fused host
path. Full-row softmax and `softmax @ V` remain separate stages because they
need full-row or full-chunk data.

Local check:

```bash
g++ -O2 -std=c++17 \
  hls/score_and_mask_scale/score_mask_scale_core_hls.cpp \
  hls/score_and_mask_scale/tb_score_mask_scale.cpp \
  -Ihls/common \
  -o sim/tb_score_mask_scale
sim/tb_score_mask_scale sim/attention_score_tile
```

Vitis HLS:

```bash
vitis_hls -f attention_score_u55c/hls/score_and_mask_scale/run_hls.tcl
```
