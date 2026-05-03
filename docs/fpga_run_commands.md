# U55C FPGA Run Commands

This is the command runbook for the isolated `attention_score_u55c` FPGA demo.
It captures the steps and parameters used to build, emulate, and run the
attention-score chain on the U55C.

## Verified Hardware State

- XRT: `2.14.354` / `2022.2`
- Vitis/Vivado: `2022.2`
- Card shell: `xilinx_u55c_gen3x16_xdma_base_3`
- Platform UUID: `97088961-FEAE-DA91-52A2-1D9DFD63CCEF`
- Build platform: `xilinx_u55c_gen3x16_xdma_3_202210_1`
- Platform file:
  `/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`
- Working directory for all commands below: `/home/advent/Desktop/RC19`

## Kernel Chains

Verified real-card single-tile path:

```text
Q_rot_int8, K_rot_int8
-> attention_score_u55c_kernel
-> mask_scale_u55c_kernel
-> softmax_u55c_kernel
```

Current tiled `hw_emu` path:

```text
Q_rot_int8, K_rot_int8
-> tiled attention_score_u55c_kernel launches
-> tiled mask_scale_u55c_kernel launches
-> softmax_full_row_u55c_kernel per Q chunk
```

The XRT host verifies:

- `score_raw.txt`
- `score_scaled.txt`
- `score_softmax.txt`

for `--vectors <dir>` single-tile mode, and generated full-sequence raw,
scaled-logit, and softmax references for `--seq-len <S>` tiled synthetic mode.

The real TinyLlama vector directory also contains `v_full.txt` and
`attn_ref_float.txt`, but those are for later `softmax @ V` work and are not
consumed by the current xclbin.

## HBM Bank Mapping

The link config used for the `.xclbin` is
`attention_score_u55c/host/vpp_link.cfg`:

```text
q_tile       -> HBM[0]
k_tile       -> HBM[1]
raw score    -> HBM[2]
tile scaled  -> HBM[3]
tile prob    -> HBM[4]
full logits  -> HBM[4]
full prob    -> HBM[5]
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
vitis_hls -f attention_score_u55c/hls/softmax_full_row/run_hls.tcl
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

v++ -c -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  -k softmax_full_row_u55c_kernel \
  -o attention_score_u55c/build/softmax_full_row_u55c_kernel.xo \
  attention_score_u55c/hls/softmax_full_row/softmax_full_row_hls.cpp
```

Then links:

```bash
v++ -l -t hw_emu \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  --config attention_score_u55c/host/vpp_link.cfg \
  -o attention_score_u55c/build/attention_score_chain.xclbin \
  attention_score_u55c/build/attention_score_u55c_kernel.xo \
  attention_score_u55c/build/mask_scale_u55c_kernel.xo \
  attention_score_u55c/build/softmax_u55c_kernel.xo \
  attention_score_u55c/build/softmax_full_row_u55c_kernel.xo
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

Tiled synthetic hardware-emulation commands:

```bash
export XCL_EMULATION_MODE=hw_emu

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --seq-len 8 \
  --device 0

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --seq-len 64 \
  --device 0

./attention_score_u55c/build/host_attention_score_chain \
  --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
  --seq-len 128 \
  --device 0
```

Verified `hw_emu` tiled results on 2026-05-02:

```text
S=8:   q_chunks=1,  k_chunks=1, Tiled sequence verification PASSED
S=64:  q_chunks=8,  k_chunks=1, Tiled sequence verification PASSED
S=128: q_chunks=16, k_chunks=2, Tiled sequence verification PASSED
```

The current `build/attention_score_chain.xclbin` after this build is a
hardware-emulation xclbin:

```text
Content: HW Emulation Binary
UUID: 1f6b30da-9112-3c47-7e52-a4a149705c10
Kernels:
  attention_score_u55c_kernel
  mask_scale_u55c_kernel
  softmax_u55c_kernel
  softmax_full_row_u55c_kernel
```

## 10. Build The Real Hardware `.xclbin`

```bash
unset XCL_EMULATION_MODE

bash attention_score_u55c/host/build_xclbin.sh \
  hw \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The hardware build uses the same four `v++ -c` compile commands and final
`v++ -l` link command as hardware emulation, with `-t hw` instead of
`-t hw_emu`.

The four-kernel tiled design has now been rebuilt and run on the real U55C card.

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

Expected xclbin details for the latest verified three-kernel real hardware build:

```text
Content: Bitstream
UUID: 06fc7f72-fc9f-b542-28d3-aac2d65918ef
Kernels:
  attention_score_u55c_kernel
  mask_scale_u55c_kernel
  softmax_u55c_kernel
```

Expected xclbin details for the latest four-kernel `hw_emu` build:

```text
Content: HW Emulation Binary
UUID: 1f6b30da-9112-3c47-7e52-a4a149705c10
Kernels:
  attention_score_u55c_kernel
  mask_scale_u55c_kernel
  softmax_u55c_kernel
  softmax_full_row_u55c_kernel
```

Expected xclbin details for the latest four-kernel real hardware build:

```text
Content: Bitstream
UUID: d0099c1c-0481-4332-4772-0a89999bc1f8
Kernels:
  attention_score_u55c_kernel
  mask_scale_u55c_kernel
  softmax_u55c_kernel
  softmax_full_row_u55c_kernel
HBM banks used:
  HBM[0] through HBM[5]
```

## 14. Run Tiled Synthetic Sequences On The Real U55C

Verified sweep command:

```bash
source attention_score_u55c/host/setup_2022_2_env.sh
unset XCL_EMULATION_MODE

for s in 8 64 128 256 512; do
  ./attention_score_u55c/build/host_attention_score_chain \
    --xclbin attention_score_u55c/build/attention_score_chain.xclbin \
    --seq-len "$s" \
    --device 0
done
```

Verified real-card results on 2026-05-02 before Track A Step 4 host
double-buffering:

| S | q_chunks | k_chunks | tiles | total ms | tiles/sec | scores/sec |
|---|----------|----------|-------|----------|-----------|------------|
| 8 | 1 | 1 | 1 | 3.321 | 301.11 | 19,271.30 |
| 64 | 8 | 1 | 8 | 1.931 | 4,142.93 | 2,121,180.74 |
| 128 | 16 | 2 | 32 | 4.622 | 6,923.41 | 3,544,785.81 |
| 256 | 32 | 4 | 128 | 13.353 | 9,585.86 | 4,907,960.76 |
| 512 | 64 | 8 | 512 | 49.444 | 10,355.15 | 5,301,836.42 |

All five runs printed:

```text
Tiled sequence verification PASSED
XRT chain verification PASSED
```

Approximate kernel timing summary from that sweep:

| S | attention score ms | mask+scale ms | full-row softmax ms | total ms |
|---|--------------------|---------------|---------------------|----------|
| 8 | 2.939 | 0.094 | 0.129 | 3.321 |
| 64 | 0.538 | 0.316 | 0.634 | 1.931 |
| 128 | 0.875 | 1.086 | 1.354 | 4.622 |
| 256 | 2.817 | 2.912 | 3.377 | 13.353 |
| 512 | 12.969 | 11.928 | 8.090 | 49.444 |

Verified real-card results on 2026-05-03 after the Track A Step 4
double-buffered tiled host pass:

| S | q_chunks | k_chunks | tiles | total ms | tiles/sec | scores/sec |
|---|----------|----------|-------|----------|-----------|------------|
| 8 | 1 | 1 | 1 | 0.510 | 1,960.78 | 125,490.20 |
| 64 | 8 | 1 | 8 | 2.307 | 3,467.71 | 1,775,465.97 |
| 128 | 16 | 2 | 32 | 5.238 | 6,109.20 | 3,127,911.42 |
| 256 | 32 | 4 | 128 | 11.709 | 10,931.76 | 5,597,062.09 |
| 512 | 64 | 8 | 512 | 35.992 | 14,225.38 | 7,283,396.31 |

All five runs printed:

```text
Tiled sequence verification PASSED
XRT chain verification PASSED
```

The Step 4 host pass uses two BO sets for Q/K/raw-score/scaled-score in the
tiled score/mask-scale loop. Full-row softmax remains after all K chunks for a
Q chunk, because it needs the complete S-wide logit row. Per-kernel mask/scale
timing is now an overlapped host-observed window; `total_chain` is the primary
comparison metric.

Approximate HBM traffic estimate for the current staged implementation,
recomputed with the 2026-05-03 Step 4 totals:

| S | estimated HBM bytes | estimated HBM GB/s |
|---|---------------------|--------------------|
| 8 | 11,264 | 0.0221 |
| 64 | 118,784 | 0.0515 |
| 128 | 475,136 | 0.0907 |
| 256 | 1,900,544 | 0.1623 |
| 512 | 7,602,176 | 0.2112 |

These are attention-score probabilities/sec metrics, not model tokens/sec.
Tokens/sec still requires Track B `softmax @ V` and decode-loop integration.

## 15. Preserve A Known-Good Run

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

The current real-card FPGA run now includes a synthetic tiled attention-score
sweep through `S = 512`. This is not yet a full tokens/sec measurement.

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

The next benchmarking step is to compare these FPGA score-probability timings
against the CPU/GPU baselines, then add Track B `softmax @ V`. Tokens/sec is not
meaningful yet because this design stops at attention probabilities and does not
run full TinyLlama decoding.
