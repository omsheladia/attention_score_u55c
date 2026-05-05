# XRT Host Flow

This folder is the host-side step for running the isolated attention-score
chain on a U55C. The host supports both the fused tiled path and the Track O2
resident full-buffer path.

```text
Q_rot_int8, K_rot_int8
-> score_mask_scale_u55c_kernel
-> softmax_full_row_u55c_kernel
-> v_weighted_sum_u55c_kernel
-> attn_out_fp32
```

Resident Track O2 mode uses:

```text
full Q/K/V device buffers
-> score_mask_scale_resident_u55c_kernel
-> softmax_full_row_resident_u55c_kernel
-> v_weighted_sum_resident_u55c_kernel
-> resident attn_out_fp32
```

The legacy single-tile `--vectors` path still uses `softmax_u55c_kernel` and
can also run `v_weighted_sum_u55c_kernel` when `v_full.txt` and an attention
reference are present.

## Files

- `attention_score_chain_xrt.cpp`
  - native XRT C++ host app
  - loads one `.xclbin`
  - launches fused score+mask+scale, softmax, and optional V weighted-sum
    kernels for `--vectors <dir>`
  - launches tiled fused score+mask+scale, full-row softmax, and V weighted sum
    for synthetic `--seq-len <S>`
  - supports `--resident` and `--resident-debug` for full-buffer device-resident
    synthetic and full-sequence vector runs
  - compares device outputs against exported vectors or generated synthetic
    full-sequence references
- `build_host.sh`
  - Linux host compile helper
- `build_xclbin.sh`
  - Linux `v++` compile/link helper
- `run_hw.sh`
  - real-card run helper for the verified U55C/XRT 2022.2 flow
- `vpp_link.cfg`
  - example HBM bank placement so the intermediate score buffers can be shared

## What This Host App Assumes

- Linux machine
- XRT installed and sourced
- U55C platform installed
- one linked `.xclbin` containing the runtime kernels
- vectors already exported under `sim/attention_score_tile/` or
  real TinyLlama directories such as `sim/real_tinyllama_tile/`,
  `sim/real_tinyllama_s16/`, or `sim/real_tinyllama_s64/`

## Example Linux Flow

Local C++ simulation, no Vitis/XRT required:

```bash
bash host/run_local_csim.sh
```

Full U55C/XRT flow:

```bash
source host/setup_2022_2_env.sh

python3 model/export_attention_score_vectors.py

bash host/build_xclbin.sh hw_emu /path/to/u55c_platform.xpfm
bash host/build_host.sh

./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --vectors sim/attention_score_tile \
  --device 0
```

Tiled synthetic `hw_emu` flow with full-row softmax and V weighted sum:

```bash
export XCL_EMULATION_MODE=hw_emu

./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --seq-len 128 \
  --device 0
```

Verified five-kernel Track B `hw_emu` sequence lengths so far:

```text
S=8:  Attention output verification PASSED, total_chain 65214.673 ms
S=64: Attention output verification PASSED, total_chain 580737.286 ms
S=128: Attention output verification PASSED, total_chain 1657498.336 ms
```

For first bring-up, `hw_emu` is the right target before `hw`.

Resident real-card sweep:

```bash
unset XCL_EMULATION_MODE

for s in 8 64 128 256 512; do
  ./build/host_attention_score_chain \
    --xclbin build/attention_score_chain.xclbin \
    --seq-len "$s" \
    --resident \
    --device 0
done
```

Use `--resident-debug` for the first small run if you also want logits and
probability readback verification.

Latest Track O2 resident real-card synthetic totals:

```text
S=8   0.930 ms
S=64  4.131 ms
S=128 12.755 ms
S=256 45.809 ms
S=512 173.489 ms
```

The resident path is correct, but slower than the O1 fused tiled path because
resident V accumulation currently performs slow HBM read-modify-write updates.

Known-good real-card run after the U55C is on shell
`xilinx_u55c_gen3x16_xdma_base_3`:

```bash
bash host/run_hw.sh 0
```

The host prints per-kernel timing and total chain timing using host wall-clock
measurements from launch through `wait()`, then verifies `score_scaled`,
`score_softmax`, and, when present, `attn_out` / `attn_ref_float` against the
reference vectors in `--vectors` mode. The fused Step 5 path no longer writes
raw INT32 scores to HBM, so `score_raw` is not a device output in this branch.
In synthetic `--seq-len` mode it also verifies final `attn_out` from the V
weighted-sum stage.

Latest verified synthetic-vector helper timing for the five-kernel hardware
xclbin:

```text
attention_score_u55c_kernel 0.066 ms
mask_scale_u55c_kernel      0.081 ms
softmax_u55c_kernel         0.117 ms
v_weighted_sum_u55c_kernel  0.041 ms
total_chain                 0.345 ms
Attention output verification PASSED
XRT chain verification PASSED
```

The same xclbin and host were also verified with the real TinyLlama-derived
single-tile vectors:

```bash
bash host/run_hw.sh \
  0 \
  build/attention_score_chain.xclbin \
  sim/real_tinyllama_tile
```

That helper run printed:

```text
attention_score_u55c_kernel 0.053 ms
mask_scale_u55c_kernel      0.094 ms
softmax_u55c_kernel         0.029 ms
v_weighted_sum_u55c_kernel  0.041 ms
total_chain                 0.258 ms
Attention output verification PASSED
XRT chain verification PASSED
```

## Deployable Meaning

There are two useful meanings of "deployable" here:

1. **Tile-demo deployable**
   - enough to run the isolated staged attention path on the card
   - this now includes score, mask/scale, softmax, and optional V weighted sum
2. **Model deployable**
   - enough to run a meaningful end-to-end attention path inside the larger
     TinyLlama accelerator
   - that still needs more blocks and system integration

Right now this workspace has reached the first meaning for a single tile and
for synthetic tiled sequence lengths through final one-head `attn_out`. It also
passes real TinyLlama-derived vector mode for the legacy single-tile case and
full-sequence `S=16` / `S=64` directories. Track A Step 4 added
double-buffered pass-1 BO sets in the tiled host path. Track B Step 3 real
hardware sequence sweeps pass for `S = 8, 64, 128, 256, 512`; the latest
five-kernel S=512 run completed in `84.190 ms`.
