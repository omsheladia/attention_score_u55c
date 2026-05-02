# Linux Bring-Up Checklist

This checklist is for the isolated `attention_score_u55c` demo only.

Goal:

```text
Q_rot_int8, K_rot_int8
-> attention_score_u55c_kernel
-> causal_mask_u55c_kernel
-> score_scale_u55c_kernel
-> softmax_u55c_kernel
```

Expected outcome:

- build one `.xclbin` containing all 4 kernels
- build one XRT host app
- run `hw_emu`
- compare device outputs against the exported reference vectors

## 1. Use Linux

Do the real Alveo/XRT flow on Linux, not Windows.

You need:

- Vitis 2023.2
- XRT installed
- U55C platform installed
- a visible U55C card for `hw`

For first bring-up, `hw_emu` is enough.

## 2. Open A Fresh Shell And Source The Tools

Example:

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

Then confirm:

```bash
v++ --version
xbutil --version || xrt-smi --version
```

Pass signal:

- both commands print normal version info

## 3. Confirm XRT And Platform Visibility

Check the card:

```bash
xbutil examine || xrt-smi examine
```

Check the platform:

```bash
platforminfo -p /path/to/u55c_platform.xpfm
```

Pass signal:

- the U55C is visible
- the `.xpfm` resolves successfully

If you do not know the installed platform path yet:

```bash
find /opt/xilinx -name "*.xpfm" 2>/dev/null | grep -i u55c
find /tools/Xilinx -name "*.xpfm" 2>/dev/null | grep -i u55c
```

## 4. Export The Reference Vectors

From the repo root:

```bash
python3 attention_score_u55c/model/export_attention_score_vectors.py
```

This should generate:

- `attention_score_u55c/sim/attention_score_tile/q_tile.txt`
- `attention_score_u55c/sim/attention_score_tile/k_tile.txt`
- `attention_score_u55c/sim/attention_score_tile/score_raw.txt`
- `attention_score_u55c/sim/attention_score_tile/score_masked.txt`
- `attention_score_u55c/sim/attention_score_tile/score_scaled.txt`
- `attention_score_u55c/sim/attention_score_tile/score_softmax.txt`

Pass signal:

- the script prints `Wrote deterministic attention-score vectors ...`

## 5. Build The `.xclbin` In Hardware Emulation

From the repo root:

```bash
bash attention_score_u55c/host/build_xclbin.sh hw_emu /path/to/u55c_platform.xpfm
```

This compiles:

- `attention_score_u55c_kernel`
- `causal_mask_u55c_kernel`
- `score_scale_u55c_kernel`
- `softmax_u55c_kernel`

And links:

- `attention_score_u55c/build/attention_score_chain.xclbin`

Pass signal:

- no `v++` error
- `attention_score_u55c/build/attention_score_chain.xclbin` exists

## 6. Build The XRT Host App

From the repo root:

```bash
bash attention_score_u55c/host/build_host.sh
```

Pass signal:

- no compiler/linker error
- `attention_score_u55c/build/host_attention_score_chain` exists

If this step fails because XRT headers or libs are missing, check:

```bash
echo "$XILINX_XRT"
ls "$XILINX_XRT/include"
ls "$XILINX_XRT/lib"
```

The host build expects:

- XRT headers under `$XILINX_XRT/include`
- XRT libraries under `$XILINX_XRT/lib`

## 7. Generate Emulation Config

If `emconfigutil` is available:

```bash
emconfigutil --platform /path/to/u55c_platform.xpfm --nd 1
```

Pass signal:

- `emconfig.json` is created in the current directory

## 8. Run Hardware Emulation

From the repo root:

```bash
export XCL_EMULATION_MODE=hw_emu

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

Or use the helper:

```bash
bash attention_score_u55c/host/run_hw_emu.sh /path/to/u55c_platform.xpfm 0
```

Pass signal:

- the host prints:
  - `Opening device 0`
  - `Running attention score kernel`
  - `Running causal mask kernel`
  - `Running score scale kernel`
  - `Running softmax kernel`
  - `XRT chain verification PASSED`

If the final line is missing, the host should also print which stage mismatched.

## 9. Move From `hw_emu` To Real `hw`

Once `hw_emu` passes:

```bash
bash attention_score_u55c/host/build_xclbin.sh hw /path/to/u55c_platform.xpfm

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

Pass signal:

- same `XRT chain verification PASSED`
- no XRT runtime error

## 10. What “Working” Means Here

If the above passes, the design is deployable as an isolated FPGA demo for the
attention-score path.

That means:

- the 4-kernel tile chain runs on the U55C
- device output matches the reference vectors
- host/XRT/kernel linkage is functioning

It does **not** yet mean full model deployment. Full TinyLlama attention still
needs:

- upstream Q/K production inside the model path
- V-path weighted sum
- wider runtime integration

## Common Failure Cases

### `v++` cannot find the platform

Use the full `.xpfm` path and verify it with:

```bash
platforminfo -p /full/path/to/platform.xpfm
```

### host build fails on XRT includes

Check:

```bash
echo "$XILINX_XRT"
find "$XILINX_XRT/include" -name "xrt*.h*"
```

### host runs but kernel open fails

Usually one of:

- wrong kernel name in the `.xclbin`
- broken link step
- wrong device/platform target

### runtime mismatch at the end

That means:

- kernels executed
- but one stage disagrees with the exported reference

In that case, the host already narrows the problem to one of:

- `score_raw`
- `score_masked`
- `score_scaled`
- `score_softmax`

## Recommended First Real Milestone

The first real success target is:

1. `hw_emu` pass
2. one real `hw` pass

After that, the next engineering task should be integrating real Q/K producers,
not expanding this isolated demo much further.
