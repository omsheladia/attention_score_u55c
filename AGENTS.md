# AGENTS.md

This file is a persistent handoff summary for AI agents working in this repo.

Future agents should read this file first, use it as the default project-state
context for the current FPGA attention-score effort, and update it after every
meaningful change.

## Update Rule

Update this file whenever you do something meaningful, including:

- create or modify architecture/code for the isolated `attention_score_u55c` flow
- add or change verification flows
- run synthesis or runtime validation and learn something important
- discover environment/platform constraints that affect bring-up
- change the recommended next steps

Keep updates short, factual, and high signal. Do not turn this into a full log.

## Original User Request

The user asked:

- inspect the Python files in `model/`
- ignore the rest of the codebase
- focus only on the attention-score portion from one of the Python scripts
- explain how to offload attention score onto the U55C FPGA
- create a new separate folder, organized similarly to the repo

The user explicitly wanted explanation first, then implementation after approval.
Approval was later given, and work proceeded.

## Scope Decision That Was Made

The implemented scope is intentionally narrower than full TinyLlama attention.

Chosen offload boundary:

```text
Q_rot_int8, K_rot_int8
-> score_raw_int32
-> score_masked_int32
-> score_scaled_fp32
-> score_softmax_fp32
```

This is based on the attention-score path in the repo, mainly:

- `model/tinyllama.py`
- `model/export_fpga_vectors.py`

What is intentionally **not** implemented in this isolated flow:

- full Q/K/V projection path on FPGA
- live RoPE generation in the full runtime path
- `softmax @ V`
- decoder-layer integration
- full TinyLlama inference

## Key Interpretation Of "Correctness"

Current correctness means:

- the isolated 4-stage attention-score chain matches the local TinyLlama-derived
  software reference vectors for the chosen tile format

Current correctness does **not** mean:

- full TinyLlama attention is verified on FPGA
- full decoder/model execution is verified

## Main Workspace Added

Created:

- `attention_score_u55c/`

Key subfolders:

- `attention_score_u55c/model/`
- `attention_score_u55c/hls/attention_score/`
- `attention_score_u55c/hls/causal_mask/`
- `attention_score_u55c/hls/score_scale/`
- `attention_score_u55c/hls/softmax/`
- `attention_score_u55c/host/`
- `attention_score_u55c/docs/`
- `attention_score_u55c/sim/`

## Important Files

### Reference / Export

- `attention_score_u55c/model/attention_score_ref.py`
- `attention_score_u55c/model/export_attention_score_vectors.py`

These export deterministic vectors under:

- `attention_score_u55c/sim/attention_score_tile/`

Current exported files include:

- `q_tile.txt`
- `k_tile.txt`
- `score_raw.txt`
- `score_masked.txt`
- `score_scaled.txt`
- `score_softmax.txt`
- `kernel_meta.txt`
- `metadata.json`

### HLS Kernels

- score GEMM:
  - `attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp`
- causal mask:
  - `attention_score_u55c/hls/causal_mask/causal_mask_core_hls.cpp`
- score scale:
  - `attention_score_u55c/hls/score_scale/score_scale_core_hls.cpp`
- softmax:
  - `attention_score_u55c/hls/softmax/softmax_core_hls.cpp`

### Host / XRT

- `attention_score_u55c/host/attention_score_chain_xrt.cpp`
- `attention_score_u55c/host/build_xclbin.sh`
- `attention_score_u55c/host/build_host.sh`
- `attention_score_u55c/host/vpp_link.cfg`
- `attention_score_u55c/host/run_hw_emu.sh`
- `attention_score_u55c/host/LINUX_BRINGUP.md`

## Verification Status

### Local / Desktop Verification

All file-driven local benches passed:

- score
- causal mask
- score scale
- softmax

### Vitis HLS Verification

`csim` and `csynth` were run successfully for:

- score kernel
- causal mask kernel
- score scale kernel
- softmax kernel

Target device used:

- `xcu55c-fsvh2892-2L-e`

### Softmax Note

The softmax kernel was later optimized to remove the earlier "loop constraints
not satisfied" issue. After optimization:

- local softmax bench still passed
- Vitis `csim` still passed
- Vitis `csynth` still passed
- loop constraint status became satisfied

There is still a small timing-budget warning in the softmax report, but the HLS
status is much cleaner than before optimization.

## Approximate HLS Snapshot

These are rough working numbers, not contractual final performance claims.

- score GEMM:
  - about 342 MHz
  - 32 DSP
- causal mask:
  - about 331 MHz
  - 0 DSP
- score scale:
  - about 342 MHz
  - 3 DSP
- softmax:
  - about 316 MHz
  - 9 DSP

Agents should re-read the current reports before repeating these numbers.

Reports live under:

- `attention_score_u55c/hls/build/.../syn/report/`

## Platform / Deployment Understanding

Important deployment conclusion:

- for U55C, the practical deploy artifact is an `.xclbin`, not just a raw `.bit`
  file in the way users often mean it

Recommended machine split:

- big Linux machine:
  - build `.xo` / `.xclbin`
  - optionally build host app
- Linux machine attached to real U55C:
  - run XRT
  - load `.xclbin`
  - execute host app

The current repo guidance and work both point toward Linux for real FPGA bring-up.

## What Has NOT Been Confirmed Yet

Not yet confirmed:

- XRT host app compile on a real Linux+XRT install
- full `.xclbin` link and run in `hw_emu`
- real on-card `hw` run on the U55C
- full TinyLlama attention path
- full TinyLlama model execution

So the project is close to an **isolated FPGA demo**, but not yet a fully proven
hardware runtime.

## Best Next Step

Best next practical step:

1. move to Linux
2. follow `attention_score_u55c/host/LINUX_BRINGUP.md`
3. get `hw_emu` passing
4. then get a real `hw` pass on the U55C

If those pass, the isolated attention-score chain can be considered deployable
as a tile-level FPGA demo.

## After That

If the isolated demo works on hardware, the next major engineering steps are:

1. integrate real Q/K producers
2. add the V weighted-sum stage
3. connect to a real TinyLlama attention subgraph
4. add KV-cache-aware decode flow
5. eventually integrate into a decoder-layer path

## Current Repo State Relevant To This Effort

At the time of writing:

- `attention_score_u55c/` is newly added and uncommitted

Agents should avoid redoing exploration that this file already captures unless
something materially changed.
