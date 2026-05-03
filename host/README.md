# XRT Host Flow

This folder is the host-side step for running the isolated attention-score
chain on a U55C. The proven real-card path is still the single-tile three-kernel
flow, while the current synthetic `hw_emu` path has advanced to a five-kernel
attention-output flow.

```text
Q_rot_int8, K_rot_int8
-> attention_score_u55c_kernel
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

## Files

- `attention_score_chain_xrt.cpp`
  - native XRT C++ host app
  - loads one `.xclbin`
  - launches score, mask/scale, softmax, and optional V weighted-sum kernels
    for `--vectors <dir>`
  - launches tiled score+mask+scale, full-row softmax, and V weighted sum for
    synthetic `--seq-len <S>`
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
  `sim/real_tinyllama_tile/`

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

Known-good real-card run after the U55C is on shell
`xilinx_u55c_gen3x16_xdma_base_3`:

```bash
bash host/run_hw.sh 0
```

The host prints per-kernel timing and total chain timing using host wall-clock
measurements from launch through `wait()`, then verifies `score_raw`,
`score_scaled`, `score_softmax`, and, when present, `attn_out` /
`attn_ref_float` against the reference vectors in `--vectors` mode. In
synthetic `--seq-len` mode it also verifies final `attn_out` from the V
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
   - enough to run this isolated three-kernel score path on the card
   - this host app is meant for that stage
2. **Model deployable**
   - enough to run a meaningful end-to-end attention path inside the larger
     TinyLlama accelerator
   - that still needs more blocks and system integration

Right now this workspace has reached the first meaning for a single tile and
for synthetic tiled sequence lengths through final one-head `attn_out`. Track A
Step 4 also added double-buffered pass-1 BO sets in the tiled host path. Track B
Step 3 real hardware sequence sweeps pass for `S = 8, 64, 128, 256, 512`; the
latest five-kernel S=512 run completed in `84.190 ms`.
