# AGENTS.md

This file is a persistent handoff summary for AI agents working in this repo.

Future agents should read this file first, use it as the default project-state
context for the current FPGA attention-score effort, and update it after every
meaningful change.

## Update Rule

Update this file whenever you do something meaningful, including:

- create or modify architecture/code for the isolated U55C attention-score flow
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
- `softmax @ V` in the current verified four-kernel flow
- decoder-layer integration
- full TinyLlama inference

## Key Interpretation Of "Correctness"

Current correctness means:

- the isolated 4-stage attention-score chain matches the local TinyLlama-derived
  software reference vectors for the chosen tile format

Current correctness does **not** mean:

- full TinyLlama attention is verified on FPGA
- full decoder/model execution is verified

## Current Required Scope

The current required scope is the four-track plan in
`docs/implementation_checklist.md`. This is still narrower than full TinyLlama,
but it has expanded beyond the original 4-stage score-weight demo.

Priority order:

1. Track A: speed up the existing synthetic-input pipeline and make it work
   across real sequence lengths
2. Track B: add `softmax @ V` as a 5th stage so the FPGA produces complete
   one-head attention output, not just attention weights
3. Track C: replace synthetic vectors with real TinyLlama Q/K/V inputs
4. Track D: collect CPU/GPU/FPGA baseline and speedup numbers

Track A priority work:

1. increase the GEMM unroll factor in `hls/attention_score/`
2. merge causal mask and score scale into one kernel
3. add full-sequence tiling in the Python reference and XRT host app
4. add full-row softmax support for `S > 64`

Important ordering conclusion:

- full-row softmax is required before sequence lengths with multiple K chunks
  can be correct
- `softmax @ V` depends on Track A tiling because it accumulates partial
  weighted-sum results across K/V chunks
- real TinyLlama Q/K/V input export is useful after the synthetic full attention
  block is working

## Tile And Sequence-Length Model

The hardware tile shape remains fixed:

```text
Q tile:      8 x 64   INT8
K tile:     64 x 64   INT8
Score tile:  8 x 64   INT32 -> FP32
```

For a sequence length `S`, the host must cover the full `S x S` score matrix
with:

```text
tiles per head = ceil(S / 8) * ceil(S / 64)
```

Recommended synthetic test lengths are `S = 8, 64, 128, 256, 512`. The current
docs state that the host tiling loop does not yet exist, so the proven flow is
still a single tile of one head unless later work changes this file.

Softmax correctness note:

- running softmax independently per 8 x 64 score tile is only correct for
  `S <= 64`
- for `S > 64`, softmax must normalize each query row across all `S` keys
- the existing `hls/softmax/` kernel is fixed at 8 x 64, so Track A now calls
  for a new `hls/softmax_full_row/` kernel or an equivalent online/tiled
  full-row softmax design

Track B data note:

- the V weighted-sum kernel should consume 8 x 64 softmax-weight tiles and
  64 x 64 V chunks
- each V-kernel call produces a partial 8 x 64 contribution; the host
  accumulates across K/V chunks into final `attn_out` with shape `(S, 64)`
- multi-chunk sequence tests should use full V data such as `v_full.txt`; a
  `v_tile.txt` file is only a single-chunk convenience

## Current Repo Layout

The isolated attention-score workspace is now the repo root, not a nested
`attention_score_u55c/` folder.

Key subfolders:

- `model/`
- `hls/attention_score/`
- `hls/causal_mask/`
- `hls/score_scale/`
- `hls/softmax/`
- `hls/common/`
- `host/`
- `docs/`
- `sim/`
- `rtl/`

## Important Files

### Reference / Export

- `model/attention_score_ref.py`
- `model/export_attention_score_vectors.py`

These export deterministic vectors under:

- `sim/attention_score_tile/`

Current exported files include:

- `q_tile.txt`
- `k_tile.txt`
- `score_raw.txt`
- `score_masked.txt`
- `score_scaled.txt`
- `score_softmax.txt`
- `kernel_meta.txt`
- `metadata.json`

Planned vector/export additions from `docs/implementation_checklist.md`:

- `softmax_full_rows(logits)` in `model/attention_score_ref.py`
- full-sequence Q/K reference exports for `S = 8, 64, 128, 256, 512`
- `v_full.txt` for complete `(S, 64)` V data
- `attn_out.txt` for final `(S, 64)` attention output after `softmax @ V`

### HLS Kernels

- score GEMM:
  - `hls/attention_score/attention_score_core_hls.cpp`
- causal mask:
  - `hls/causal_mask/causal_mask_core_hls.cpp`
- score scale:
  - `hls/score_scale/score_scale_core_hls.cpp`
- softmax:
  - `hls/softmax/softmax_core_hls.cpp`

Planned HLS additions from `docs/implementation_checklist.md`:

- `hls/mask_and_scale/` to merge causal mask and score scale
- `hls/softmax_full_row/` for full-row softmax up to `S = 512`
- `hls/v_weighted_sum/` for the `softmax @ V` partial weighted-sum kernel
- `hls/score_and_mask_scale/` as a later pre-softmax dataflow merge

### Host / XRT

- `host/attention_score_chain_xrt.cpp`
- `host/build_xclbin.sh`
- `host/build_host.sh`
- `host/vpp_link.cfg`
- `host/run_hw.sh`
- `host/run_hw_emu.sh`
- `host/LINUX_BRINGUP.md`
- `docs/fpga_run_commands.md`

## Verification Status

### Real Hardware Verification

On 2026-04-28, the U55C shell was updated from
`xilinx_u55c_gen3x16_xdma_base_2` to `xilinx_u55c_gen3x16_xdma_base_3`.
After cold power cycle, `xbutil examine` reported the user function ready on
`[0000:01:00.1]` with platform UUID `97088961-FEAE-DA91-52A2-1D9DFD63CCEF`.

The real hardware `build/attention_score_chain.xclbin` was loaded and run on
device 0 with the XRT host app. All four kernels ran and the host printed
`XRT chain verification PASSED`.

On 2026-04-29, a preservation backup was created at
`backups/run_20260429_201240/`. It contains a repo snapshot, hardware xclbin,
xo files, host binary, vectors, HLS/Vitis reports/logs, XRT device state,
checksums, and a fresh real-hardware verification log.

On 2026-04-30, `host/run_hw.sh` was added as the clean real-card runner for the
verified flow. It sources `host/setup_2022_2_env.sh`, unsets
`XCL_EMULATION_MODE`, validates the host binary/xclbin/vector paths, and runs
device 0 by default. The XRT host app now prints host wall-clock timing for each
kernel launch/wait and total four-kernel chain runtime before reporting
verification. Running `bash host/run_hw.sh 0` on the real
card passed and printed:

- attention score: 0.110 ms
- causal mask: 0.029 ms
- score scale: 0.021 ms
- softmax: 0.026 ms
- total chain: 0.193 ms

On 2026-04-30, `docs/fpga_run_commands.md` was added as the concise command
runbook for the verified FPGA flow. It records the exact environment setup,
platform path, vector export parameters, local C++ checks, HLS csim/csynth
commands, hw_emu build/run commands, real hardware build/run commands, HBM bank
mapping, and current benchmark scope.

On 2026-05-01, the top-level `README.md` was refreshed to describe the current
verified four-kernel U55C demo instead of the earlier score-kernel prototype.
It now points to `docs/fpga_run_commands.md`, documents the real-card run helper
and verification scope, and lists benchmark/HBM follow-up work.

### HW Emulation Verification

On 2026-04-28, after full Vitis 2022.2 was added, `host/build_xclbin.sh hw_emu`
successfully produced `build/attention_score_chain.xclbin` for
`xilinx_u55c_gen3x16_xdma_3_202210_1`. `xclbinutil --info` identifies it as a
`HW Emulation Binary` containing all four kernels:

- `attention_score_u55c_kernel`
- `causal_mask_u55c_kernel`
- `score_scale_u55c_kernel`
- `softmax_u55c_kernel`

The XRT host app was rebuilt and run with `XCL_EMULATION_MODE=hw_emu`; the full
four-kernel chain completed and printed `XRT chain verification PASSED`.

### Historical Bring-Up Notes

On 2026-04-28, this device had `g++`, Python, XRT 2022.2 under
`/opt/xilinx/xrt`, Vivado/Vitis HLS 2022.2 under `/home/advent/Vivado`, and
the U55C platform file under
`/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/`. It did not have
full Vitis 2022.2 installed/discoverable. A `v++` exists under
`/home/advent/petalinux/tools/xsct/bin`, but it is a versionless XSCT wrapper
and is not sufficient for this platform; it failed with:
`Platform 'xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm' (version 2022.1) is not
supported by the current tool version (.)`. `.xclbin` packaging/linking is
therefore still blocked until full Vitis 2022.2 is added.

Outside the earlier sandbox, XRT sees the card and reports it ready:
`xbutil examine` shows `[0000:01:00.1]` as
`xilinx_u55c_gen3x16_xdma_base_2`, and `xbmgmt examine` shows `[0000:01:00.0]`
as the matching management function. `lspci` sees Xilinx devices `505c` and
`505d` at `01:00.0` and `01:00.1`; `xclmgmt` and `xocl` are loaded and bound.

The development platform in use is `xdma_3_202210_1`, while the card currently
reports deployment shell `xdma_base_2`. Matching local base_3 deployment .debs
exist under `/home/advent/Downloads`, but installing them requires sudo
password. Flashing/updating the shell may also require a reboot or cold power
cycle.

A minimal Vitis 2022.2 add-on config was saved as
`host/vitis_2022_2_add_config.txt`. The installer is available locally at
`/home/advent/Downloads/Xilinx_Unified_2022.2_1014_8888_Lin64.bin`, but batch
install requires an AMD/Xilinx auth token generated with
`/home/advent/Vivado/.xinstall/Vivado_2022.2/xsetup -b AuthTokenGen`; that
prompt requires the user's AMD/Xilinx credentials.

A local native C++ validation helper was added at `host/run_local_csim.sh`; it
regenerates vectors, builds the four C++ testbenches, and runs score, causal
mask, score scale, and softmax. All four local benches passed on this device.
After sourcing `/opt/xilinx/xrt/setup.sh`, the XRT host app also compiled
successfully to `build/host_attention_score_chain`.

An environment helper was added at `host/setup_2022_2_env.sh`. Source it before
building/running this demo. `host/build_xclbin.sh` now rejects missing `v++` and
the versionless PetaLinux XSCT `v++` wrapper with a clearer error.

The HLS Tcl scripts were made portable for this Vitis HLS install by opening
simple project names from inside `hls/build/` instead of passing slash-containing
paths to `open_project`. Running Vitis HLS outside the sandbox was required
because `csim_design` opens a local dispatch-server port.

Vitis HLS 2022.2 `csim` and `csynth` passed again on this device for all four
kernels:

- attention score: estimated Fmax 342.47 MHz, 32 DSP, 57 BRAM_18K
- causal mask: estimated Fmax 342.47 MHz, 0 DSP, 2 BRAM_18K
- score scale: estimated Fmax 342.47 MHz, 3 DSP, 2 BRAM_18K
- softmax: estimated Fmax 315.96 MHz, 9 DSP, 2 BRAM_18K; timing-budget warning
  remains around the `sum_exp` floating-point add path, but loop constraints
  were satisfied

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

- `hls/build/.../syn/report/`

## Platform / Deployment Understanding

Important deployment conclusion:

- for U55C, the practical deploy artifact is an `.xclbin`, not just a raw `.bit`
  file in the way users often mean it

Recommended machine split:

- big Linux machine:
  - build `.xo` / `.xclbin`
  - optionally build host app if XRT development headers/libs are installed
- Linux machine attached to real U55C:
  - run XRT
  - load `.xclbin`
  - execute host app

The current repo guidance and work both point toward Linux for real FPGA bring-up.

Build-machine-specific note:

- the build machine does not need a physically attached U55C card to compile the
  kernel objects or link the final `.xclbin`
- it does need Vitis plus the matching U55C platform `.xpfm`
- it only needs XRT if the host app will also be compiled there
- the target machine attached to the card needs XRT even if Vitis is absent

Current pulled-state note:

- earlier planning assumed the target Linux machine attached to the U55C would
  be needed for full bring-up
- the pulled `AGENTS.md` history now records successful `hw_emu` and real-card
  verification of the single-tile four-kernel chain
- future `.xclbin` builds and hardware runs still require Linux, XRT, Vitis, and
  a matching U55C platform

## What Has NOT Been Confirmed Yet

Not yet confirmed:

- full-sequence host tiling over `S = 8, 64, 128, 256, 512`
- synthetic Track A regression through multiple vector directories
- full-row softmax kernel/design for `S > 64`
- merged mask-and-scale kernel
- `softmax @ V` / V weighted-sum stage
- real TinyLlama `Q_rot` / `K_rot` extraction, V extraction, and INT8 Q/K export
- CPU/GPU/FPGA baseline comparison for acceleration claims
- full TinyLlama attention path
- full TinyLlama model execution

So the project is now a proven **single-tile isolated FPGA demo**, but not yet a
full sequence-length attention accelerator or a full TinyLlama hardware runtime.

## Best Next Step

Best next practical steps are Track A from
`docs/implementation_checklist.md`:

1. increase the GEMM unroll factor and re-synthesize
2. merge causal mask and score scale into one kernel and verify locally/HLS
3. add Python full-sequence tiling with `softmax_full_rows(logits)`
4. add the matching XRT host tiling loop for `S = 8, 64, 128, 256, 512`
5. add a full-row softmax kernel/design for `S > 64`

## After That

After Track A works across sequence lengths, the next major engineering steps
are:

1. Track B: add `softmax @ V` with a V weighted-sum HLS kernel
2. export and verify `v_full.txt` and `attn_out.txt`
3. Track C: extract real TinyLlama Q/K/V with PyTorch hooks
4. quantize/export real Q/K and keep V as float32
5. run real vectors through the same FPGA pipeline and compare `attn_out` to
   PyTorch attention output
6. Track D: collect CPU/GPU/FPGA timing baselines and speedup tables
7. add double-buffering or merge kernels for performance
8. connect to a real TinyLlama attention subgraph
9. add KV-cache-aware decode flow
10. eventually integrate into a decoder-layer path

## Current Repo State Relevant To This Effort

At the time of writing:

- the isolated workspace lives at the repo root (`model/`, `hls/`, `host/`,
  `docs/`, `sim/`, `rtl/`)
- `CLAUDE.md`, `docs/attention_concepts.md`, and
  `docs/implementation_checklist.md` define the current required scope:
  Track A first, then Track B, Track C, and Track D
- `docs/implementation_checklist.md` now treats full-row softmax as required
  for `S > 64`, adds `softmax @ V` as Track B, upgrades real inputs to Q/K/V,
  and adds CPU/GPU/FPGA baseline work
- `AGENTS.md` has been updated to reconcile the pulled real-hardware history
  with those current scope docs

Agents should avoid redoing exploration that this file already captures unless
something materially changed.
