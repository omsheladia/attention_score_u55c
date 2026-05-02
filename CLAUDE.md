# CLAUDE.md

## Project Overview

Isolated FPGA attention-score pipeline targeting the Xilinx Alveo U55C.
Carved from a TinyLlama inference repo; implements one 4-stage tile chain only.

**Offload boundary:**
```
Q_rot_int8, K_rot_int8
  -> score_raw_int32      (INT8×INT8 GEMM)
  -> score_masked_int32   (causal mask)
  -> score_scaled_fp32    (scale by q_scale * k_scale * 1/sqrt(head_dim))
  -> score_softmax_fp32   (row-wise softmax)
```

**Out of scope:** Q/K/V projections, RoPE generation, softmax@V, decoder integration.

---

## Repo Layout

```
model/      Python reference math + deterministic vector export
hls/
  attention_score/   stage 1 — INT8 score GEMM
  causal_mask/       stage 2 — causal mask
  score_scale/       stage 3 — FP32 scale
  softmax/           stage 4 — row-wise softmax
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
| Score GEMM HLS | [hls/attention_score/attention_score_core_hls.cpp](hls/attention_score/attention_score_core_hls.cpp) |
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
| score GEMM | PASS | PASS | PASS (~342 MHz, 32 DSP) |
| causal mask | PASS | PASS | PASS (~331 MHz, 0 DSP) |
| score scale | PASS | PASS | PASS (~342 MHz, 3 DSP) |
| softmax | PASS | PASS | PASS (~316 MHz, 9 DSP) — minor timing warning remains |

Not yet run: `hw_emu`, real `hw` on U55C, full XRT host compile.

> Re-read HLS reports under `hls/build/.../syn/report/` before quoting numbers — the table above is a snapshot.

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

Full bring-up requires Linux with Vitis 2023.2 + XRT + U55C platform.
Follow [host/LINUX_BRINGUP.md](host/LINUX_BRINGUP.md) step by step.

**Short version:**
```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

# build xclbin (hw_emu first)
bash host/build_xclbin.sh hw_emu /path/to/u55c_platform.xpfm

# build host app
bash host/build_host.sh

# generate emconfig and run
emconfigutil --platform /path/to/u55c_platform.xpfm --nd 1
bash host/run_hw_emu.sh /path/to/u55c_platform.xpfm 0
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

Correctness is scoped to: the 4-stage tile chain matches the exported reference
vectors for the chosen tile format. It does **not** mean full TinyLlama attention
or full model execution is verified.

---

## Input Data — Synthetic vs Real

The current Python exporter (`model/export_attention_score_vectors.py`) generates
**synthetic deterministic Q/K vectors** using a fixed mathematical pattern. These
are not real language model inputs — they exist purely to verify hardware
correctness in a reproducible way.

Two testing tracks are planned:

**Track A — Synthetic inputs (primary)**
Use the synthetic exporter with the host tiling loop to test correctness and
measure performance at multiple sequence lengths: S = 8, 64, 128, 256, 512.
No TinyLlama installation required.

**Track B — Real TinyLlama inputs (extension)**
Run TinyLlama inference via PyTorch, hook into one attention layer to extract
real `Q_rot` and `K_rot` tensors, quantize to INT8, and export in the same file
format the host app already reads. The FPGA pipeline and host app do not change —
only the input files change.

See [docs/implementation_checklist.md](docs/implementation_checklist.md) for the
full step-by-step plan for both tracks.

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
the 4 kernel stages also pass through HBM in the current design.

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
- 8-way MAC parallelism inside the GEMM kernel (`UNROLL factor=8`)
- Loop pipelining at II=1 in all kernels
- 8-bank array partitioning for parallel SRAM reads in the GEMM

**Does not exist:**
- The 4 kernel stages run sequentially (no dataflow streaming between them)
- Tiles are processed one at a time (no double buffering)
- All 32 attention heads are processed sequentially (one kernel instance)

---

## Recommended Next Steps

**Immediate (Track A):**
1. Move to Linux (Vitis 2023.2 + XRT + U55C platform)
2. Follow [host/LINUX_BRINGUP.md](host/LINUX_BRINGUP.md) — get `hw_emu` passing
3. Increase GEMM UNROLL factor (1 line, re-synthesize)
4. Add host tiling loop — test at S = 8, 64, 128, 256, 512
5. Merge causal mask + score scale into one kernel

**Extension (Track B):**
1. Extract real Q/K from TinyLlama via PyTorch hooks
2. Quantize to INT8 and export in existing file format
3. Run real vectors through the FPGA pipeline
4. Compare FPGA softmax output against PyTorch reference

**Future (post-hardware confirmation):**
1. Double-buffer DMA transfers
2. Merge all stages into one dataflow kernel
3. Add V weighted-sum stage
4. Connect to full TinyLlama attention subgraph
