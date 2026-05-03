# U55C FPGA Run Commands

This is the command runbook for the isolated `attention_score_u55c` FPGA demo.
It captures the steps and parameters used to build, emulate, and run the current
three-kernel attention-score chain on the real U55C.

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
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

The current XRT host verifies:

- `score_raw.txt`
- `score_scaled.txt`
- `score_softmax.txt`

The real TinyLlama vector directory also contains `v_full.txt` and
`attn_ref_float.txt`, but those are for later `softmax @ V` work and are not
consumed by the current three-kernel xclbin.

## HBM Bank Mapping

The link config used for the `.xclbin` is
`attention_score_u55c/host/vpp_link.cfg`:

```text
q_tile       -> HBM[0]
k_tile       -> HBM[1]
raw score    -> HBM[2]
scaled score -> HBM[3]
prob out     -> HBM[4]
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

The real-card passes used device index `0`.

## 3. Export The Synthetic Reference Vectors

Default verified synthetic case:

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

## 4. Export The Real TinyLlama-Derived Vectors

The current real-vector directory has already been exported at:

```text
attention_score_u55c/sim/real_tinyllama_tile/
```

To regenerate it:

```bash
python3 attention_score_u55c/model/export_real_vectors.py \
  --output-dir attention_score_u55c/sim/real_tinyllama_tile
```

Default real-vector metadata:

```text
prompt: The cat sat on the mat.
query_rows: 8
key_cols: 8
query_pos_base: 0
key_pos_base: 0
q_scale: 0.0489841662347317
k_scale: 0.0174317248165607
total_scale: 0.0001067348132716
```

The exported real-vector files include:

```text
q_tile.txt
k_tile.txt
kernel_meta.txt
score_raw.txt
score_masked.txt
score_scaled.txt
score_softmax.txt
score_packed.txt
metadata.json
q_float.txt
k_float.txt
v_full.txt
attn_ref_float.txt
```

## 5. Run Local C++ Verification

This regenerates the default synthetic vectors, builds the native C++
testbenches, and runs the local checks:

```bash
bash attention_score_u55c/host/run_local_csim.sh
```

For the current runtime path, the important local benches are:

```bash
g++ -O2 -std=c++17 \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp \
  attention_score_u55c/hls/attention_score/tb_attention_score.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_attention_score

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/mask_and_scale/mask_scale_core_hls.cpp \
  attention_score_u55c/hls/mask_and_scale/tb_mask_scale.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_mask_scale

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp \
  attention_score_u55c/hls/softmax/tb_softmax.cpp \
  -Iattention_score_u55c/hls/common \
  -o attention_score_u55c/sim/tb_softmax

attention_score_u55c/sim/tb_attention_score attention_score_u55c/sim/attention_score_tile
attention_score_u55c/sim/tb_mask_scale attention_score_u55c/sim/attention_score_tile
attention_score_u55c/sim/tb_softmax attention_score_u55c/sim/attention_score_tile
```

The same benches can be pointed at the real TinyLlama vectors:

```bash
attention_score_u55c/sim/tb_attention_score attention_score_u55c/sim/real_tinyllama_tile
attention_score_u55c/sim/tb_mask_scale attention_score_u55c/sim/real_tinyllama_tile
attention_score_u55c/sim/tb_softmax attention_score_u55c/sim/real_tinyllama_tile
```

Expected pass signal from the helper:

```text
Local attention-score C++ simulation chain PASSED
```

## 6. Run Vitis HLS C Simulation And Synthesis

Each HLS Tcl script runs `csim_design` and `csynth_design` for one kernel.

```bash
vitis_hls -f attention_score_u55c/hls/attention_score/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/mask_and_scale/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/softmax/run_hls.tcl
```

Legacy split-kernel HLS projects still exist for reference:

```bash
vitis_hls -f attention_score_u55c/hls/causal_mask/run_hls.tcl
vitis_hls -f attention_score_u55c/hls/score_scale/run_hls.tcl
```

The scripts use:

```text
part: xcu55c-fsvh2892-2L-e
clock: 4 ns
vector_dir: attention_score_u55c/sim/attention_score_tile
```

## 7. Build The XRT Host App

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

## 8. Build The Hardware Emulation `.xclbin`

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
  -k mask_scale_u55c_kernel \
  -o attention_score_u55c/build/mask_scale_u55c_kernel.xo \
  attention_score_u55c/hls/mask_and_scale/mask_scale_core_hls.cpp

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
  attention_score_u55c/build/mask_scale_u55c_kernel.xo \
  attention_score_u55c/build/softmax_u55c_kernel.xo
```

## 9. Run Hardware Emulation

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

## 10. Build The Real Hardware `.xclbin`

```bash
unset XCL_EMULATION_MODE

bash attention_score_u55c/host/build_xclbin.sh \
  hw \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The hardware build uses the same three `v++ -c` compile commands and final
`v++ -l` link command as hardware emulation, with `-t hw` instead of
`-t hw_emu`.

Output:

```text
attention_score_u55c/build/attention_score_chain.xclbin
```

## 11. Run Synthetic Vectors On The Real U55C

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
Running mask+scale kernel
Running softmax kernel
Kernel timing summary (host wall-clock, launch through wait):
XRT chain verification PASSED
```

One verified real-card helper run printed:

```text
attention_score_u55c_kernel 0.043 ms
mask_scale_u55c_kernel      0.025 ms
softmax_u55c_kernel         0.086 ms
total_chain                 0.159 ms
```

## 12. Run Real TinyLlama Vectors On The Real U55C

Verified direct command:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --vectors attention_score_u55c/sim/real_tinyllama_tile \
  --device 0
```

Verified helper command:

```bash
bash attention_score_u55c/host/run_hw.sh \
  0 \
  attention_score_u55c/build/attention_score_chain.xclbin \
  attention_score_u55c/sim/real_tinyllama_tile
```

Direct real-card run result on 2026-05-02:

```text
Opening device 0
Running attention score kernel
Running mask+scale kernel
Running softmax kernel
Kernel timing summary (host wall-clock, launch through wait):
  attention_score_u55c_kernel 0.059 ms
  mask_scale_u55c_kernel      0.029 ms
  softmax_u55c_kernel         0.028 ms
  total_chain                 0.121 ms
XRT chain verification PASSED
```

Helper real-card run result on 2026-05-02:

```text
Opening device 0
Running attention score kernel
Running mask+scale kernel
Running softmax kernel
Kernel timing summary (host wall-clock, launch through wait):
  attention_score_u55c_kernel 0.062 ms
  mask_scale_u55c_kernel      0.024 ms
  softmax_u55c_kernel         0.028 ms
  total_chain                 0.121 ms
XRT chain verification PASSED
```

## 13. Inspect The Built XCLBIN

```bash
xclbinutil --info \
  --input attention_score_u55c/build/attention_score_chain.xclbin
```

Expected xclbin details for the current real hardware build:

```text
Content: Bitstream
UUID: 06fc7f72-fc9f-b542-28d3-aac2d65918ef
Kernels:
  attention_score_u55c_kernel
  mask_scale_u55c_kernel
  softmax_u55c_kernel
```

## 14. Preserve A Known-Good Run

The older known-good preservation backup was created at:

```text
attention_score_u55c/backups/run_20260429_201240/
```

Its manifest is:

```bash
sed -n '1,220p' attention_score_u55c/backups/run_20260429_201240/MANIFEST.md
```

The backup includes the earlier verified `.xclbin`, host binary, vectors, logs,
reports, device state, checksums, and the real hardware verification log.

## Current Benchmark Scope

The current FPGA run is a verified single attention-score tile. It is not yet a
multi-sequence-length benchmark or full tokens/sec measurement.

Verified synthetic vector parameters:

```text
query_rows: 4
key_cols: 10
query_pos_base: 6
key_pos_base: 0
q_scale: 0.03125
k_scale: 0.02734375
```

Verified real TinyLlama vector parameters:

```text
query_rows: 8
key_cols: 8
query_pos_base: 0
key_pos_base: 0
q_scale: 0.0489841662347317
k_scale: 0.0174317248165607
total_scale: 0.0001067348132716
```

The next benchmarking step is to generate multiple vector directories with
different `query_rows` and `key_cols`, run each through the host app, and report
latency, tiles/sec, effective scores/sec, and HBM traffic estimates. Tokens/sec
is not meaningful yet because this design stops at attention probabilities and
does not run `softmax @ V` or full TinyLlama decoding.
