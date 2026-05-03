# CLAUDE.md

## Project Overview

Isolated FPGA attention-score pipeline targeting the Xilinx Alveo U55C.
Carved from a TinyLlama inference repo; the current verified runtime chain is a
3-kernel single-tile pipeline.

**Offload boundary:**
```
Q_rot_int8, K_rot_int8
  -> score_raw_int32      (INT8×INT8 GEMM)
  -> score_scaled_fp32    (merged causal mask + scale)
  -> score_softmax_fp32   (row-wise softmax)
```

**Currently out of scope:** Q/K/V projections, RoPE generation, decoder integration.
**Planned (Track B):** softmax@V to complete the full attention block for one head.

---

## Repo Layout

```
model/      Python reference math + deterministic vector export
hls/
  attention_score/   stage 1 — INT8 score GEMM
  mask_and_scale/    stage 2 — merged causal mask + FP32 scale
  causal_mask/       legacy standalone causal mask
  score_scale/       legacy standalone FP32 scale
  softmax/           stage 3 — row-wise softmax
  common/            shared fixed-point types (fixed_types.hpp)
host/       XRT host app, build scripts, vpp_link.cfg, bring-up guide
sim/
  attention_score_tile/   exported reference vectors (txt + json)
rtl/        placeholder for future RTL lowering
docs/       offload design notes
```

---

## Key Files

| Purpose | Path |
|---|---|
| Python reference | [model/attention_score_ref.py](model/attention_score_ref.py) |
| Vector export | [model/export_attention_score_vectors.py](model/export_attention_score_vectors.py) |
| TinyLlama setup check | [model/check_tinyllama_setup.py](model/check_tinyllama_setup.py) |
| Q/K/V extraction + INT8 quant | [model/extract_tinyllama_qkv.py](model/extract_tinyllama_qkv.py) |
| Real vector export (single-tile) | [model/export_real_vectors.py](model/export_real_vectors.py) |
| CPU attention baseline | [model/benchmark_cpu.py](model/benchmark_cpu.py) |
| GPU attention baseline | [model/benchmark_gpu.py](model/benchmark_gpu.py) |
| Score GEMM HLS | [hls/attention_score/attention_score_core_hls.cpp](hls/attention_score/attention_score_core_hls.cpp) |
| Mask+scale HLS | [hls/mask_and_scale/mask_scale_core_hls.cpp](hls/mask_and_scale/mask_scale_core_hls.cpp) |
| Causal mask HLS | [hls/causal_mask/causal_mask_core_hls.cpp](hls/causal_mask/causal_mask_core_hls.cpp) |
| Score scale HLS | [hls/score_scale/score_scale_core_hls.cpp](hls/score_scale/score_scale_core_hls.cpp) |
| Softmax HLS | [hls/softmax/softmax_core_hls.cpp](hls/softmax/softmax_core_hls.cpp) |
| XRT host app | [host/attention_score_chain_xrt.cpp](host/attention_score_chain_xrt.cpp) |
| xclbin build script | [host/build_xclbin.sh](host/build_xclbin.sh) |
| Host build script | [host/build_host.sh](host/build_host.sh) |
| hw_emu run script | [host/run_hw_emu.sh](host/run_hw_emu.sh) |
| Linux bring-up guide | [host/LINUX_BRINGUP.md](host/LINUX_BRINGUP.md) |

---

## Verification Status

| Stage | Local bench | Vitis csim | Vitis csynth |
|---|---|---|---|
| score GEMM | PASS | PASS | PASS (~342 MHz, 32 DSP, ~1312 cycles after 64-bit packed I/O) |
| mask+scale | PASS | PASS | PASS (~331 MHz, 3 DSP) |
| causal mask | PASS | PASS | PASS (~331 MHz, 0 DSP) |
| score scale | PASS | PASS | PASS (~342 MHz, 3 DSP) |
| softmax | PASS | PASS | PASS (~316 MHz, 9 DSP) — minor timing warning remains |

XRT deployment status for the current 3-kernel chain:

| Target | Result | Notes |
|---|---|---|
| host compile | PASS | `bash host/build_host.sh` |
| `hw_emu` xclbin | PASS | Vitis 2022.2, U55C platform `xilinx_u55c_gen3x16_xdma_3_202210_1` |
| `hw_emu` run | PASS | `XRT chain verification PASSED`; emulation timing is simulator dominated |
| real `hw` xclbin | PASS | hardware link took about 43 minutes |
| real U55C run | PASS | device 0, shell `xilinx_u55c_gen3x16_xdma_base_3`; synthetic and real TinyLlama single-tile vectors pass |

> Re-read HLS reports under `hls/build/.../syn/report/` before quoting numbers — the table above is a snapshot.

---

## Current Track A WIP

- Step 1 code change is in place:
  - `hls/attention_score/attention_score_core_hls.cpp` now uses
    `#pragma HLS UNROLL factor=16`
  - matching local array partitioning is now 16-way
  - local score-kernel bench still passes
  - Vitis HLS 2023.2 `csim` and `csynth` now pass again at
    `~342.47 MHz`, `32 DSP`, `7 BRAM_18K`
  - important result: this source change did not move the synthesized score
    kernel off the earlier ~`32 DSP` point under the current tool/loop shape
  - a follow-on contained optimization then widened the score-kernel external
    Q/K/score memory traffic to 64-bit packed words without changing the Track A
    checklist ordering
  - after that packed-I/O change, the score kernel still synthesized at
    `~342.47 MHz` and `32 DSP`, but latency improved from about `5153 cycles`
    to about `1312 cycles`
  - the improvement came from memory traffic:
    - Q load about `515 -> 67` cycles
    - K load about `4099 -> 515` cycles
    - score store about `516 -> 259` cycles
- Step 2 code change is in place:
  - new merged kernel under `hls/mask_and_scale/`
  - `host/attention_score_chain_xrt.cpp`, `host/vpp_link.cfg`, and
    `host/build_xclbin.sh` now target `attention_score -> mask_scale -> softmax`
  - local merged-kernel bench passes against `score_scaled.txt`
  - Vitis HLS 2023.2 `csim` and `csynth` now pass for the merged kernel at
    `~330.91 MHz`, `3 DSP`, `6 BRAM_18K`
- Still pending for this WIP:
  - CPU/GPU/FPGA baseline tables using the real U55C sequence sweep
  - Track B `softmax @ V`
  - full multi-sequence real TinyLlama vector export

### 2026-05-02 Track A Step 3 Part A

The Python full-sequence tiled score/softmax reference is implemented in
`model/attention_score_ref.py`:

- `compute_full_attention_score(q_full, k_full, ...)`
- `softmax_full_rows(logits)`
- `brute_force_full_attention_score(...)`
- CLI validation with `python3 model/attention_score_ref.py --check-full-tiling`

The verified run covered `S = 8, 64, 128, 256, 512` and reported zero max
difference for raw scores, scaled logits, and full-row softmax probabilities.
This started as Python reference work; the full-row HLS softmax kernel and XRT
host tiling loop were added later on 2026-05-02.

### 2026-05-02 Track A Full-Row Softmax Kernel

`hls/softmax_full_row/` was added as the correctness-first full-row softmax
kernel for `8 x S` rows with `S <= 512`. The original `hls/softmax/` tile
kernel is unchanged for the current single-tile runtime.

Verification:

- local C++ bench passed for `S = 64, 128, 256, 512`, partial rows, and masked logits
- Vitis HLS 2022.2 `csim PASS`
- Vitis HLS 2022.2 `csynth PASS`
- estimated Fmax: `315.96 MHz`
- resources: `9 DSP`, `1 BRAM_18K`, `2 URAM`, `3693 FF`, `5772 LUT`
- loop constraint status: all loop constraints satisfied

This kernel was then connected to `host/attention_score_chain_xrt.cpp`,
`host/build_xclbin.sh`, and `host/vpp_link.cfg` for the four-kernel tiled path.

### 2026-05-02 Four-Kernel Tiled XRT HW Emulation

The tiled XRT host path is now implemented for synthetic sequence lengths:

- legacy `--vectors <dir>` mode still runs the existing single-tile path
- new `--seq-len <S>` mode generates deterministic Q/K, runs tiled
  score+mask+scale, assembles full-row logits, and runs
  `softmax_full_row_u55c_kernel`
- `host/build_xclbin.sh` compiles and links four kernels
- `host/vpp_link.cfg` maps full-row logits to `HBM[4]` and full-row
  probabilities to `HBM[5]`

Fresh `hw_emu` xclbin build passed. `xclbinutil --info` reports content
`HW Emulation Binary`, UUID `1f6b30da-9112-3c47-7e52-a4a149705c10`, and kernels:

```text
attention_score_u55c_kernel
mask_scale_u55c_kernel
softmax_u55c_kernel
softmax_full_row_u55c_kernel
```

Verified `hw_emu` runs:

```text
S=8:   q_chunks=1,  k_chunks=1, Tiled sequence verification PASSED
S=64:  q_chunks=8,  k_chunks=1, Tiled sequence verification PASSED
S=128: q_chunks=16, k_chunks=2, Tiled sequence verification PASSED
```

The original single-tile `--vectors attention_score_u55c/sim/attention_score_tile`
path also passed under the new four-kernel `hw_emu` xclbin. Real hardware
rebuild/run for this tiled design was completed next.

### 2026-05-02 Four-Kernel Real U55C Sweep

The four-kernel real hardware xclbin build passed. `xclbinutil --info` reports:

```text
Content: Bitstream
UUID: d0099c1c-0481-4332-4772-0a89999bc1f8
Kernels: attention_score_u55c_kernel, mask_scale_u55c_kernel,
         softmax_u55c_kernel, softmax_full_row_u55c_kernel
HBM banks used: HBM[0] through HBM[5]
```

Verified real-card tiled synthetic sequence sweep:

| S | q_chunks | k_chunks | tiles | total ms | tiles/sec | scores/sec |
|---|----------|----------|-------|----------|-----------|------------|
| 8 | 1 | 1 | 1 | 3.321 | 301.11 | 19,271.30 |
| 64 | 8 | 1 | 8 | 1.931 | 4,142.93 | 2,121,180.74 |
| 128 | 16 | 2 | 32 | 4.622 | 6,923.41 | 3,544,785.81 |
| 256 | 32 | 4 | 128 | 13.353 | 9,585.86 | 4,907,960.76 |
| 512 | 64 | 8 | 512 | 49.444 | 10,355.15 | 5,301,836.42 |

All runs printed `Tiled sequence verification PASSED` and
`XRT chain verification PASSED`. The old single-tile synthetic and real
TinyLlama vector modes also pass on this four-kernel bitstream.

### 2026-05-03 Track A Step 4 Double-Buffered Host Pass

The tiled XRT host pass now uses two BO sets for pass 1:

```text
q_bo[2], k_bo[2], raw_score_bo[2], scaled_score_bo[2]
```

The host alternates K tiles through those sets so the next score tile can be
prepared/launched while the previous mask+scale tile is still in flight. The
full-row softmax pass is intentionally not double-buffered here because it must
wait for all K chunks in a Q chunk to assemble the complete S-wide logit row.

Verification on the real U55C with the existing four-kernel bitstream:

| S | q_chunks | k_chunks | tiles | total ms | tiles/sec | scores/sec |
|---|----------|----------|-------|----------|-----------|------------|
| 8 | 1 | 1 | 1 | 0.510 | 1,960.78 | 125,490.20 |
| 64 | 8 | 1 | 8 | 2.307 | 3,467.71 | 1,775,465.97 |
| 128 | 16 | 2 | 32 | 5.238 | 6,109.20 | 3,127,911.42 |
| 256 | 32 | 4 | 128 | 11.709 | 10,931.76 | 5,597,062.09 |
| 512 | 64 | 8 | 512 | 35.992 | 14,225.38 | 7,283,396.31 |

All runs printed `Tiled sequence verification PASSED` and
`XRT chain verification PASSED`. Legacy `--vectors` mode also still passes for
both synthetic vectors and `sim/real_tinyllama_tile`. Per-kernel mask+scale
timing is now an overlapped host-observed window; use `total_chain` as the
primary Step 4 metric.

### 2026-05-01 3-Kernel XRT Results

Platform used:

```text
/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The rebuilt xclbin contains:

```text
attention_score_u55c_kernel
mask_scale_u55c_kernel
softmax_u55c_kernel
```

`xclbinutil --info` for the real hardware xclbin reports:

```text
Content: Bitstream
UUID: 06fc7f72-fc9f-b542-28d3-aac2d65918ef
HBM banks used: HBM[0] through HBM[4]
Clocks: hbm_aclk 450 MHz, KERNEL_CLK 500 MHz, DATA_CLK 300 MHz
```

Hardware emulation passed:

```text
attention_score_u55c_kernel 1000.153 ms
mask_scale_u55c_kernel      1000.076 ms
softmax_u55c_kernel         1000.158 ms
total_chain                 3000.421 ms
XRT chain verification PASSED
```

The real U55C run passed. The first post-program run showed cold-launch timing:

```text
attention_score_u55c_kernel 6.998 ms
mask_scale_u55c_kernel      0.085 ms
softmax_u55c_kernel         0.200 ms
total_chain                 7.302 ms
XRT chain verification PASSED
```

Repeat direct run:

```text
attention_score_u55c_kernel 0.137 ms
mask_scale_u55c_kernel      0.123 ms
softmax_u55c_kernel         0.099 ms
total_chain                 0.367 ms
XRT chain verification PASSED
```

Verified `host/run_hw.sh` helper run:

```text
attention_score_u55c_kernel 0.043 ms
mask_scale_u55c_kernel      0.025 ms
softmax_u55c_kernel         0.086 ms
total_chain                 0.159 ms
XRT chain verification PASSED
```

Verified real TinyLlama-derived vector run using `sim/real_tinyllama_tile/`:

```text
attention_score_u55c_kernel 0.062 ms
mask_scale_u55c_kernel      0.024 ms
softmax_u55c_kernel         0.028 ms
total_chain                 0.121 ms
XRT chain verification PASSED
```

Deployment notes:

- `hw_emu` printed `Unable to find emconfig.json. Using default device ...`
  despite `emconfigutil` creating `emconfig.json`; the run still passed.
- Vitis 2022.2 `v++` softmax compile still reports one unsatisfied loop
  constraint, but the 3-kernel `hw_emu` and real hardware runs both passed.

---

## Target Device

`xcu55c-fsvh2892-2L-e`

---

## Quick Start (local desktop, no FPGA needed)

**Generate reference vectors:**
```bash
python3 model/export_attention_score_vectors.py
```

**Run a local C-sim testbench (any machine with g++):**
```bash
g++ -std=c++17 hls/attention_score/attention_score_core_hls.cpp \
    hls/attention_score/tb_attention_score.cpp \
    -Ihls/common -o sim/tb_attention_score
./sim/tb_attention_score
```

Repeat the pattern for `causal_mask`, `score_scale`, and `softmax` testbenches.

---

## FPGA Build & Run (Linux only)

Full bring-up requires Linux with Vitis/XRT + U55C platform. The current
3-kernel xclbin was built and run with Vitis/XRT 2022.2 and the local U55C
platform path below.
Follow [host/LINUX_BRINGUP.md](host/LINUX_BRINGUP.md) step by step.

**Short version:**
```bash
source host/setup_2022_2_env.sh

# build xclbin (hw_emu first)
bash host/build_xclbin.sh hw_emu \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm

# build host app
bash host/build_host.sh

# generate emconfig and run
emconfigutil \
  --platform /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
  --nd 1
export XCL_EMULATION_MODE=hw_emu
./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --vectors sim/attention_score_tile \
  --device 0
```

Pass signal: `XRT chain verification PASSED`

---

## Platform Notes

- Deploy artifact is an `.xclbin`, not a raw bitfile.
- The build machine does **not** need a physically attached U55C.
- The machine running the host app **must** have XRT installed.
- XRT is not available on the current large build machine; target Linux machine attached to the U55C is the practical all-in-one path.

---

## What "Correct" Means Here

Correctness is scoped to: the 3-kernel runtime chain matches the exported reference
vectors for the chosen tile format. It does **not** mean full TinyLlama attention
or full model execution is verified.

---

## Input Data — Synthetic vs Real

The current Python exporter (`model/export_attention_score_vectors.py`) generates
**synthetic deterministic Q/K vectors** using a fixed mathematical pattern. These
are not real language model inputs — they exist purely to verify hardware
correctness in a reproducible way.

Four implementation tracks are planned in priority order:

**Track A — Synthetic inputs (primary)**
Use the synthetic exporter with the host tiling loop to test correctness and
measure performance at multiple sequence lengths: S = 8, 64, 128, 256, 512.
No TinyLlama installation required.

**Track B — Complete attention block**
Add `softmax @ V` as a 5th kernel so the FPGA produces a complete attention
output `(S, 64)` rather than stopping at softmax weights. V stays float32 —
the FPGA already runs float32 in stages 3 and 4.
Key constraint: the existing softmax kernel is fixed at 64 columns; a new
full-row softmax kernel (up to 512 columns) is required before the tiling loop
works correctly for S > 64.

**Track C — Real TinyLlama inputs (extension)**
Run TinyLlama inference via PyTorch, hook into one attention layer to extract
real `Q_rot` (RoPE-rotated), `K_rot` (RoPE-rotated), and `V` (projected,
not RoPE-rotated) tensors. Quantize Q/K to INT8; V stays float32. Export in
the same file format the host app already reads.

Steps 1–4 are implemented for the current single-tile design:
`model/check_tinyllama_setup.py` verifies the TinyLlama environment;
`model/extract_tinyllama_qkv.py` extracts Q/K/V via a forward-hook, applies
INT8 symmetric quantization to Q and K, and validates against PyTorch SDPA
output; `model/export_real_vectors.py` exports real TinyLlama Q/K/V vectors
to `sim/real_tinyllama_tile/` in the same file format the host app reads.
The current real-vector export is still limited to one 8-token tile, while the
synthetic Track A tiled host path now supports `S = 8, 64, 128, 256, 512`.
The current single-tile real-vector directory has passed on the real U55C with
`XRT chain verification PASSED` and a 0.121 ms helper total-chain timing.
Remaining work is extending Step 4/5 to the full tiling loop, multiple sequence
lengths, and later `attn_out` comparison once Track B exists.

**Track D — CPU/GPU baseline and benchmarking**
Measure FPGA `attn_out` latency against CPU/GPU at S = 8, 64, 128, 256, 512.
CPU and GPU baselines run on Windows. FPGA timing requires Linux + XRT.

Steps 1–2 are implemented: `model/benchmark_cpu.py` runs the CPU baseline for
synthetic S = 8, 64, 128, 256, 512 and real-vector input, validated with zero
tiled-vs-brute difference for softmax and `softmax @ V`; `model/benchmark_gpu.py`
runs the CUDA baseline with the same inputs, verified on an RTX 3050 Laptop GPU
(`torch 2.11.0+cu128`). Final speedup numbers require Track A full-sequence
FPGA tiling (Step 3) and Track B `softmax @ V` on FPGA before a meaningful
comparison can be made.

See [docs/implementation_checklist.md](docs/implementation_checklist.md) for the
full step-by-step plan for all four tracks.

---

## How the Host Communicates with the U55C

The physical connection is PCIe. The software interface is XRT (Xilinx Runtime).

```
Host CPU                                  U55C FPGA
────────────────────────────────────      ─────────────────────────────
device.load_xclbin()               ─────► loads bitstream onto FPGA
xrt::bo — allocate HBM buffer      ─────► allocates buffer in HBM
bo.sync(TO_DEVICE)                 ─────► DMA over PCIe → HBM
kernel(q_bo, k_bo, ...)            ─────► kernel runs, reads/writes HBM
run.wait()                         ◄─────  kernel signals done
bo.sync(FROM_DEVICE)               ◄─────  DMA result over PCIe → host RAM
```

HBM (High Bandwidth Memory) is the 16 GB memory physically on the U55C die.
All kernel inputs and outputs pass through HBM. Intermediate results between
the 3 runtime kernel stages also pass through HBM in the current design.

---

## Tile Size and Sequence Length

The hardware is fixed at one tile size:

```
Q tile:     8 rows  × 64 cols  (8 query tokens,  INT8)
K tile:    64 rows  × 64 cols  (64 key tokens,   INT8)
Score tile: 8 rows  × 64 cols  (output block,    INT32 → FP32)
```

To process a full sequence of length S, the host loops over tiles:

```
tiles per head = ceil(S/8) × ceil(S/64)
```

| S | Tiles/head |
|---|---|
| 8 | 1 |
| 64 | 8 |
| 128 | 32 |
| 256 | 128 |
| 512 | 512 |

The synthetic tiled host loop exists in `--seq-len <S>` mode and has passed on
the real U55C for `S = 8, 64, 128, 256, 512`.

---

## Parallelism in the Current Design

**Exists:**
- 16-way MAC parallelism in the current Track A WIP GEMM kernel
- Loop pipelining at II=1 in all kernels
- 16-bank array partitioning for parallel SRAM reads in the current Track A WIP GEMM kernel

**Does not exist:**
- The 3 runtime kernel stages run sequentially (no dataflow streaming between them)
- Tiled pass 1 uses two BO sets for score/mask-scale double buffering; full-row
  softmax still runs once per Q chunk after the complete S-wide row is ready
- All 32 attention heads are processed sequentially (one kernel instance)

---

## Recommended Next Steps

**Track A — tiled real-card path done; remaining benchmarks and next-stage work:**
- Steps 1–2 complete: UNROLL factor=16, merged mask+scale kernel, 3-kernel chain verified on hw_emu and real U55C hardware.
- Step 3 Part A complete: Python full-sequence tiling with `softmax_full_rows()` reference verified at S = 8, 64, 128, 256, 512.
- Full-row softmax HLS kernel complete: `hls/softmax_full_row/` passes local bench and Vitis HLS.
- Step 3 Part B complete in `hw_emu`: XRT host tiling loop and full-row kernel pass at S = 8, 64, 128.
- Four-kernel real hardware xclbin complete: real U55C sweep passes at S = 8, 64, 128, 256, 512.
- Step 4 complete for the current staged host: double-buffered pass-1 BO sets pass on real U55C at S = 8, 64, 128, 256, 512.
- Step 5 remains a future dataflow/fusion milestone.
- Next: start Track B `softmax @ V` and add CPU/GPU/FPGA comparison tables.
- If more pre-softmax speed is needed, prefer wider packing/on-chip fusion before chasing more GEMM unroll

**Track B — Complete attention block:**
1. Add `softmax @ V` Python reference and synthetic V export
2. Implement V weighted-sum HLS kernel (`hls/v_weighted_sum/`)
3. Update host app with three-pass tiling loop

**Track C — Steps 1–5 done for the current single-tile design; remaining full-tiling work:**
- Steps 1–4 complete: TinyLlama loads on CPU, Q/K/V extraction via hook verified, INT8 Q/K quantization verified, `model/export_real_vectors.py` exports real vectors to `sim/real_tinyllama_tile/` (seq_len ≤ 8).
- Step 5 current single-tile FPGA run complete: `sim/real_tinyllama_tile/` passed on the real U55C with `XRT chain verification PASSED` and 0.121 ms total-chain helper timing.
- Step 4 tiling extension: unblocked by Track A host tiling; still needs exporter work for multi-sequence real TinyLlama vector directories.
- Step 5 full-coverage extension: run multiple real-vector sequence lengths; compare softmax output now, or `attn_out` once Track B is done, against PyTorch reference

**Track D — Steps 1–2 done; remaining:**
- Steps 1–2 complete: `model/benchmark_cpu.py` verified for synthetic S = 8, 64, 128, 256, 512 and real-vector input; `model/benchmark_gpu.py` verified on RTX 3050 Laptop GPU.
- Step 3: FPGA timing instrumentation for attention-score/softmax is now available from the real U55C Track A sweeps.
- Step 4: Comparison table and speedup analysis can be done for score/softmax now; full-attention comparison still needs Track B `softmax @ V`.

**Future (post-hardware confirmation):**
1. Merge pre-softmax stages into one dataflow kernel
2. Connect to full TinyLlama attention subgraph
