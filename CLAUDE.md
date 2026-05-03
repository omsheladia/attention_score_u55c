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
  - full-sequence tiling and full-row softmax for `S > 64`
  - multi-vector regression and CPU/GPU/FPGA baseline tables

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
to `sim/real_tinyllama_tile/` in the same file format the host app reads
(currently limited to seq_len ≤ 8 — one Q tile — pending Track A Step 3).
The current single-tile real-vector directory has passed on the real U55C with
`XRT chain verification PASSED` and a 0.121 ms helper total-chain timing.
Remaining work is extending Step 4/5 to the full tiling loop, multiple sequence
lengths, and later `attn_out` comparison once Track B exists.

**Track D — CPU/GPU baseline and benchmarking**
Measure FPGA `attn_out` latency against CPU/GPU at S = 8, 64, 128, 256, 512.
CPU and GPU baselines run on Windows. FPGA timing requires Linux + XRT.

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

The tiling loop does not yet exist in the host app — it is Track A Step 3
in the implementation checklist.

---

## Parallelism in the Current Design

**Exists:**
- 16-way MAC parallelism in the current Track A WIP GEMM kernel
- Loop pipelining at II=1 in all kernels
- 16-bank array partitioning for parallel SRAM reads in the current Track A WIP GEMM kernel

**Does not exist:**
- The 3 runtime kernel stages run sequentially (no dataflow streaming between them)
- Tiles are processed one at a time (no double buffering)
- All 32 attention heads are processed sequentially (one kernel instance)

---

## Recommended Next Steps

**Track A — Steps 1–2 done; remaining:**
- Steps 1–2 complete: UNROLL factor=16, merged mask+scale kernel, 3-kernel chain verified on hw_emu and real U55C hardware.
- Step 3: Add Python full-sequence tiling with `softmax_full_rows()` reference
- Step 4: Implement full-row softmax HLS kernel (`hls/softmax_full_row/`) — required for S > 64
- Step 5: Add matching XRT host tiling loop for S = 8, 64, 128, 256, 512
- Step 6: Add multi-vector regression and CPU/GPU/FPGA timing tables
- If more pre-softmax speed is needed, prefer wider packing/on-chip fusion before chasing more GEMM unroll

**Track B — Complete attention block:**
1. Add `softmax @ V` Python reference and synthetic V export
2. Implement V weighted-sum HLS kernel (`hls/v_weighted_sum/`)
3. Update host app with three-pass tiling loop

**Track C — Steps 1–5 done for the current single-tile design; remaining full-tiling work:**
- Steps 1–4 complete: TinyLlama loads on CPU, Q/K/V extraction via hook verified, INT8 Q/K quantization verified, `model/export_real_vectors.py` exports real vectors to `sim/real_tinyllama_tile/` (seq_len ≤ 8).
- Step 5 current single-tile FPGA run complete: `sim/real_tinyllama_tile/` passed on the real U55C with `XRT chain verification PASSED` and 0.121 ms total-chain helper timing.
- Step 4 tiling extension: blocked on Track A Step 3 (full-row softmax + host tiling loop)
- Step 5 full-coverage extension: run multiple real-vector sequence lengths; compare softmax output now, or `attn_out` once Track B is done, against PyTorch reference

**Track D — Benchmarking:**
1. CPU baseline in Python (runnable on Windows now)
2. GPU baseline in PyTorch (runnable on Windows if CUDA available)
3. FPGA timing instrumentation (requires Linux + XRT)
4. Comparison table and analysis

**Future (post-hardware confirmation):**
1. Double-buffer DMA transfers
2. Merge pre-softmax stages into one dataflow kernel
3. Connect to full TinyLlama attention subgraph
