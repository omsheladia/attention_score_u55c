# V Weighted Sum Kernel

Track B kernel for one partial `softmax @ V` contribution.

```text
weights_tile: 8 x 64 float32
v_tile:       64 x 64 float32
out_tile:      8 x 64 float32
```

The kernel computes one K/V chunk:

```text
out[row, dim] = sum_col weights[row, col] * v[col, dim]
```

The XRT host will accumulate these partial outputs across all K/V chunks to
produce final `attn_out` with shape `S x 64`.

Verified on 2026-05-03:

```bash
bash attention_score_u55c/host/run_local_csim.sh
source attention_score_u55c/host/setup_2022_2_env.sh
vitis_hls -f attention_score_u55c/hls/v_weighted_sum/run_hls.tcl
```

Results:

- local C++ bench PASS, max diff `5.96046e-08`
- Vitis HLS 2022.2 `csim PASS`
- Vitis HLS 2022.2 `csynth PASS`
- estimated Fmax: `342.47 MHz`
- latency: `5600 cycles`
- resources: `320 DSP`, `49 BRAM_18K`, `0 URAM`, `50454 FF`, `32358 LUT`
- loop constraints satisfied
