# XRT Host Flow

This folder is the first real host-side step for running the isolated
four-kernel chain on a U55C:

```text
Q_rot_int8, K_rot_int8
-> attention_score_u55c_kernel
-> causal_mask_u55c_kernel
-> score_scale_u55c_kernel
-> softmax_u55c_kernel
```

## Files

- `attention_score_chain_xrt.cpp`
  - native XRT C++ host app
  - loads one `.xclbin`
  - launches the four kernels in sequence
  - compares device outputs against the exported reference vectors
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
- one linked `.xclbin` containing all four kernels
- vectors already exported under `attention_score_u55c/sim/attention_score_tile/`

## Example Linux Flow

Local C++ simulation, no Vitis/XRT required:

```bash
bash attention_score_u55c/host/run_local_csim.sh
```

Full U55C/XRT flow:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh

python3 attention_score_u55c/model/export_attention_score_vectors.py

bash attention_score_u55c/host/build_xclbin.sh hw_emu /path/to/u55c_platform.xpfm
bash attention_score_u55c/host/build_host.sh

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

For first bring-up, `hw_emu` is the right target before `hw`.

Known-good real-card run after the U55C is on shell
`xilinx_u55c_gen3x16_xdma_base_3`:

```bash
bash attention_score_u55c/host/run_hw.sh 0
```

The host prints per-kernel timing and total chain timing using host wall-clock
measurements from launch through `wait()`, then verifies all intermediate
outputs against the reference vectors.

## Deployable Meaning

There are two useful meanings of "deployable" here:

1. **Tile-demo deployable**
   - enough to run this isolated four-kernel score path on the card
   - this host app is meant for that stage
2. **Model deployable**
   - enough to run a meaningful end-to-end attention path inside the larger
     TinyLlama accelerator
   - that still needs more blocks and system integration

Right now this workspace is very close to the first meaning, but not the second.
