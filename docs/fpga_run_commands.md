# U55C FPGA Run Commands

This is the command runbook for the isolated `attention_score_u55c` FPGA demo.
It captures the steps and parameters used to build, emulate, and run the
four-kernel attention-score chain on the real U55C.

## Verified Hardware State

- XRT: `2.14.354` / `2022.2`
- Vitis/Vivado: `2022.2`
- Card shell: `xilinx_u55c_gen3x16_xdma_base_3`
- Platform UUID: `97088961-FEAE-DA91-52A2-1D9DFD63CCEF`
- Build platform: `xilinx_u55c_gen3x16_xdma_3_202210_1`
- Platform file:
  `/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`
- Working directory for all commands below: `/home/advent/Desktop/RC19`

## Kernel Chain

```text
Q_rot_int8, K_rot_int8
-> attention_score_u55c_kernel
-> causal_mask_u55c_kernel
-> score_scale_u55c_kernel
-> softmax_u55c_kernel
```

## HBM Bank Mapping

The link config used for the `.xclbin` is
`attention_score_u55c/host/vpp_link.cfg`:

```text
q_tile      -> HBM[0]
k_tile      -> HBM[1]
raw score   -> HBM[2]
masked      -> HBM[3]
scaled      -> HBM[4]
prob out    -> HBM[5]
```

## 1. Source The 2022.2 Environment

```bash
cd /home/advent/Desktop/RC19
source attention_score_u55c/host/setup_2022_2_env.sh
```

The helper sources XRT and the local 2022.2 Xilinx installs:

```bash
source /opt/xilinx/xrt/setup.sh
source /home/advent/Vivado/Vivado/2022.2/settings64.sh
source /home/advent/Vivado/Vitis_HLS/2022.2/settings64.sh
source /home/advent/Vivado/Vitis/2022.2/settings64.sh
```

## 2. Confirm Tool And Card Visibility

```bash
v++ --version
xbutil --version
xbutil examine
xbmgmt examine
platforminfo -p /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The real-card pass used device index `0`.

## 3. Export The Reference Vectors

Default verified case:

```bash
python3 attention_score_u55c/model/export_attention_score_vectors.py \
  --output-dir attention_score_u55c/sim/attention_score_tile \
  --query-rows 4 \
  --key-cols 10 \
  --query-pos-base 6 \
  --key-pos-base 0 \
  --q-scale 0.03125 \
  --k-scale 0.02734375
```

The generated files are:

```text
attention_score_u55c/sim/attention_score_tile/q_tile.txt
attention_score_u55c/sim/attention_score_tile/k_tile.txt
attention_score_u55c/sim/attention_score_tile/score_raw.txt
attention_score_u55c/sim/attention_score_tile/score_masked.txt
attention_score_u55c/sim/attention_score_tile/score_scaled.txt
attention_score_u55c/sim/attention_score_tile/score_softmax.txt
attention_score_u55c/sim/attention_score_tile/kernel_meta.txt
attention_score_u55c/sim/attention_score_tile/metadata.json
```

## 4. Run Local C++ Verification

This regenerates vectors, builds the native C++ testbenches, and runs all four
kernel testbenches against the vector directory:

```bash
bash attention_score_u55c/host/run_local_csim.sh
```

Expanded commands used by that helper:

```bash
python3 attention_score_u55c/model/export_attention_score_vectors.py \
  --output-dir attention_score_u55c/sim/attention_score_tile

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp \
  attention_score_u55c/hls/attention_score/tb_attention_score.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_attention_score

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/causal_mask/causal_mask_core_hls.cpp \
  attention_score_u55c/hls/causal_mask/tb_causal_mask.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_causal_mask

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/score_scale/score_scale_core_hls.cpp \
  attention_score_u55c/hls/score_scale/tb_score_scale.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_score_scale

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp \
  attention_score_u55c/hls/softmax/tb_softmax.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_softmax

attention_score_u55c/sim/tb_attention_score attention_score_u55c/sim/attention_score_tile
attention_score_u55c/sim/tb_causal_mask attention_score_u55c/sim/attention_score_tile
attention_score_u55c/sim/tb_score_scale attention_score_u55c/sim/attention_score_tile
attention_score_u55c/sim/tb_softmax attention_score_u55c/sim/attention_score_tile
```

Expected pass signal:

```text
Local attention-score C++ simulation chain PASSED
```

## 5. Run Vitis HLS C Simulation And Synthesis

Each HLS Tcl script runs `csim_design` and `csynth_design` for one kernel.

```bash
vitis_hls -f attention_score_u55c/hls/attention_score/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/causal_mask/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/score_scale/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/softmax/run_hls.tcl
```

The scripts use:

```text
part: xcu55c-fsvh2892-2L-e
clock: 4 ns
vector_dir: attention_score_u55c/sim/attention_score_tile
```

## 6. Build The XRT Host App

```bash
bash attention_score_u55c/host/build_host.sh
```

Expanded compile command:

```bash
g++ -O2 -std=c++17 \
  -I"$XILINX_XRT/include" \
  -L"$XILINX_XRT/lib" \
  -o attention_score_u55c/build/host_attention_score_chain \
  attention_score_u55c/host/attention_score_chain_xrt.cpp \
  -lxrt_coreutil -pthread
```

Expected output:

```text
Built attention_score_u55c/build/host_attention_score_chain
```

## 7. Build The Hardware Emulation `.xclbin`

```bash
bash attention_score_u55c/host/build_xclbin.sh \
  hw_emu \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The helper compiles these objects:

```bash
v++ -c -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  -k attention_score_u55c_kernel \
  -o attention_score_u55c/build/attention_score_u55c_kernel.xo \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp

v++ -c -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  -k causal_mask_u55c_kernel \
  -o attention_score_u55c/build/causal_mask_u55c_kernel.xo \
  attention_score_u55c/hls/causal_mask/causal_mask_core_hls.cpp

v++ -c -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  -k score_scale_u55c_kernel \
  -o attention_score_u55c/build/score_scale_u55c_kernel.xo \
  attention_score_u55c/hls/score_scale/score_scale_core_hls.cpp

v++ -c -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  -k softmax_u55c_kernel \
  -o attention_score_u55c/build/softmax_u55c_kernel.xo \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp
```

Then links:

```bash
v++ -l -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  --config attention_score_u55c/host/vpp_link.cfg \
  -o attention_score_u55c/build/attention_score_chain.xclbin \
  attention_score_u55c/build/attention_score_u55c_kernel.xo \
  attention_score_u55c/build/causal_mask_u55c_kernel.xo \
  attention_score_u55c/build/score_scale_u55c_kernel.xo \
  attention_score_u55c/build/softmax_u55c_kernel.xo
```

## 8. Run Hardware Emulation

One-command helper:

```bash
bash attention_score_u55c/host/run_hw_emu.sh \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  0
```

Expanded run commands:

```bash
emconfigutil \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  --nd 1

export XCL_EMULATION_MODE=hw_emu

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

Expected pass signal:

```text
XRT chain verification PASSED
```

## 9. Build The Real Hardware `.xclbin`

```bash
unset XCL_EMULATION_MODE

bash attention_score_u55c/host/build_xclbin.sh \
  hw \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The hardware build uses the same four `v++ -c` compile commands and final
`v++ -l` link command as hardware emulation, with `-t hw` instead of
`-t hw_emu`.

Output:

```text
attention_score_u55c/build/attention_score_chain.xclbin
```

## 10. Run On The Real U55C

Verified direct command:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/attention_score_tile \
  --device 0
```

Equivalent helper:

```bash
bash attention_score_u55c/host/run_hw.sh \
  0 \
  attention_score_u55c/build/attention_score_chain.xclbin \
  attention_score_u55c/sim/attention_score_tile
```

The helper defaults to the same values, so this is also valid:

```bash
bash attention_score_u55c/host/run_hw.sh 0
```

Expected output includes:

```text
Opening device 0
Running attention score kernel
Running causal mask kernel
Running score scale kernel
Running softmax kernel
Kernel timing summary (host wall-clock, launch through wait):
XRT chain verification PASSED
```

One verified real-card run printed:

```text
attention_score_u55c_kernel 0.110 ms
causal_mask_u55c_kernel     0.029 ms
score_scale_u55c_kernel     0.021 ms
softmax_u55c_kernel         0.026 ms
total_chain                 0.193 ms
```

## 11. Inspect The Built XCLBIN

```bash
xclbinutil --info \
  --input attention_score_u55c/build/attention_score_chain.xclbin
```

Expected kernels:

```text
attention_score_u55c_kernel
causal_mask_u55c_kernel
score_scale_u55c_kernel
softmax_u55c_kernel
```

## 12. Preserve A Known-Good Run

The known-good preservation backup was created at:

```text
attention_score_u55c/backups/run_20260429_201240/
```

Its manifest is:

```bash
sed -n '1,220p' attention_score_u55c/backups/run_20260429_201240/MANIFEST.md
```

The backup includes the verified `.xclbin`, host binary, vectors, logs, reports,
device state, checksums, and the real hardware verification log.

## Current Benchmark Scope

The current FPGA run is a verified single attention-score tile. It is not yet a
multi-sequence-length benchmark or full tokens/sec measurement.

Current active vector parameters:

```text
query_rows: 4
key_cols: 10
query_pos_base: 6
key_pos_base: 0
q_scale: 0.03125
k_scale: 0.02734375
```

The next benchmarking step is to generate multiple vector directories with
different `query_rows` and `key_cols`, run each through the host app, and report
latency, tiles/sec, effective scores/sec, and HBM traffic estimates.
