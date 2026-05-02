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

### Current Device Verification

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

- `attention_score_u55c/hls/build/.../syn/report/`

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

Current user environment note:

- the user later found that XRT likely cannot be installed on the large build
  machine
- as a result, the practical next path is to do the full build and run flow on
  the target Linux machine attached to the U55C, assuming it has enough local
  tool support and disk space

## What Has NOT Been Confirmed Yet

Not yet confirmed:

- full TinyLlama attention path
- full TinyLlama model execution

So the project is now a proven **isolated FPGA demo**, but not yet a full
TinyLlama hardware runtime.

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
