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
- `softmax @ V` in the current verified runtime flow
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

On 2026-05-01, Track A Steps 1-2 were started in code:

- `hls/attention_score/attention_score_core_hls.cpp` now uses
  `#pragma HLS UNROLL factor=16` and matching 16-way local array partitioning
- a new merged kernel lives under `hls/mask_and_scale/`
- `host/attention_score_chain_xrt.cpp`, `host/vpp_link.cfg`, and
  `host/build_xclbin.sh` now target a 3-kernel chain:
  `attention_score -> mask_scale -> softmax`
- local g++ file-driven benches passed for:
  - `tb_attention_score.cpp`
  - `tb_mask_scale.cpp`
- Vitis HLS 2023.2 reruns now also pass on this Windows machine for:
  - `hls/attention_score/run_hls.tcl`
  - `hls/mask_and_scale/run_hls.tcl`
- fresh HLS results from those reruns:
  - `attention_score_u55c_kernel`: `csim PASS`, `csynth PASS`,
    estimated `342.47 MHz`, `32 DSP`, `7 BRAM_18K`
  - `mask_scale_u55c_kernel`: `csim PASS`, `csynth PASS`,
    estimated `330.91 MHz`, `3 DSP`, `6 BRAM_18K`
- important synthesis note:
  - under Vitis HLS 2023.2, the Step 1 source change to
    `#pragma HLS UNROLL factor=16` did **not** change the synthesized score
    kernel DSP count from the earlier ~`32 DSP` snapshot
  - the HLS log reports the inner dot-product loop as effectively completely
    unrolled under the surrounding pipeline, so the checklist's simple
    expected-DSP table does not directly describe the current synthesized result
- on 2026-05-01, a contained score-kernel bandwidth optimization was then added
  without changing the Track A task ordering:
  - `attention_score_u55c_kernel` now uses 64-bit packed external memory words
    for Q, K, and raw-score traffic while keeping the same functional tile math
  - this does **not** start Track A Step 3 tiling, Track A Step 4
    double-buffering, or Track A Step 5 full-kernel dataflow fusion
- fresh HLS results after the 64-bit packed-I/O score-kernel change:
  - `attention_score_u55c_kernel`: `csim PASS`, `csynth PASS`,
    estimated `342.47 MHz`, `32 DSP`, `13 BRAM_18K`
  - kernel latency improved from about `5153 cycles` to about `1312 cycles`
  - stage-level latency moved from roughly:
    - Q load: `515 -> 67`
    - K load: `4099 -> 515`
    - score store: `516 -> 259`
  - this confirms the main bottleneck was external memory beat width, not GEMM
    MAC count
- `v++` exists locally, but no U55C `.xpfm` platform file was found under the
  checked local Vitis 2023.2 platform directories, so rebuilt `xclbin` / XRT
  verification for the new 3-kernel chain remains pending

On 2026-05-01, the current 3-kernel chain was built and verified with Vitis/XRT
2022.2 on Linux using platform:
`/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`.

- `bash attention_score_u55c/host/build_xclbin.sh hw_emu <xpfm>` passed and
  produced a HW Emulation Binary containing:
  - `attention_score_u55c_kernel`
  - `mask_scale_u55c_kernel`
  - `softmax_u55c_kernel`
- `bash attention_score_u55c/host/build_host.sh` passed.
- `hw_emu` run passed with `XRT chain verification PASSED`.
  Host wall-clock emulation timing was approximately:
  - attention score: 1000.153 ms
  - mask+scale: 1000.076 ms
  - softmax: 1000.158 ms
  - total chain: 3000.421 ms
- `bash attention_score_u55c/host/build_xclbin.sh hw <xpfm>` passed and
  produced a real hardware bitstream xclbin. Hardware link took about 43
  minutes. `xclbinutil --info` reports content `Bitstream`, UUID
  `06fc7f72-fc9f-b542-28d3-aac2d65918ef`, kernels
  `attention_score_u55c_kernel`, `mask_scale_u55c_kernel`,
  `softmax_u55c_kernel`, HBM banks `[0]` through `[4]` used, and clocks:
  HBM 450 MHz, kernel 500 MHz, data 300 MHz.
- Direct real-card run on device 0 passed with `XRT chain verification PASSED`.
  The first post-program run showed a cold-looking total of 7.302 ms, dominated
  by the first attention-score launch at 6.998 ms.
- A repeat direct real-card run passed with timings:
  - attention score: 0.137 ms
  - mask+scale: 0.123 ms
  - softmax: 0.099 ms
  - total chain: 0.367 ms
- The `host/run_hw.sh` helper was also verified on the real card and passed with
  timings:
  - attention score: 0.043 ms
  - mask+scale: 0.025 ms
  - softmax: 0.086 ms
  - total chain: 0.159 ms
- Deployment notes:
  - `emconfigutil` created `emconfig.json`, but XRT printed
    `Unable to find emconfig.json. Using default device ...` during `hw_emu`;
    the emulation run still completed successfully.
  - Vitis 2022.2 `v++` compile for the softmax kernel still reports
    `Loop Constraint Status: All loop constraints were NOT satisfied` because
    one loop reached II=4 vs target II=3, but the xclbin linked and both
    `hw_emu` and hardware verification passed.

On 2026-05-02, the live README files were refreshed for the verified 3-kernel
runtime. Updated files include `README.md`, `host/README.md`, `hls/README.md`,
`model/README.md`, `hls/mask_and_scale/README.md`,
`hls/causal_mask/README.md`, `hls/score_scale/README.md`,
`hls/softmax/README.md`, and `sim/README.md`. The historical READMEs under
`backups/run_20260429_201240/repo_snapshot/` were intentionally left unchanged.

On 2026-05-02, Track C Step 1 was started with
`model/check_tinyllama_setup.py`. The script checks for `torch`,
`transformers`, and `sentencepiece`, loads
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` or a local model path, and runs one short
forward pass. Local verification performed here was limited to `py_compile` and
`--help`; the 2.2 GB model download/load and forward pass still need to be run
in an environment with the dependencies and enough disk/RAM/GPU capacity. The
current Python environment has `torch` and `transformers` discoverable but is
missing `sentencepiece`.

Later on 2026-05-02, the user installed the missing dependency and ran:
`python model/check_tinyllama_setup.py`. TinyLlama loaded on CPU with
`torch.float32`, the weights loaded successfully, the input shape was `(1, 8)`,
the logits shape was `(1, 8, 32000)`, and the script printed
`TinyLlama forward pass OK`. Track C Step 1 is therefore implemented and
verified.

The same day, `README.md`, `model/README.md`, and `host/README.md` were updated
to mention the Track C TinyLlama setup checker and to use current repo-root
paths (`host/...`, `model/...`, `sim/...`) instead of stale nested
`attention_score_u55c/...` command paths.

Also on 2026-05-02, Track C Step 2 was implemented with
`model/extract_tinyllama_qkv.py`. It registers a pre-hook on one TinyLlama
attention layer, mirrors the installed Transformers Llama attention projection
and RoPE code, captures `Q_rot`, `K_rot`, and projected `V`, maps Q heads to KV
heads for grouped-query attention, and validates one selected head against
PyTorch scaled-dot-product attention. The verified local run used the cached
TinyLlama model on CPU with the default 8-token prompt and printed:
`Q_rot (1, 32, 8, 64)`, `K_rot (1, 4, 8, 64)`, `V (1, 4, 8, 64)`,
`attn_out (8, 64)`, `SDPA max diff 3.72529030e-09`, and
`TinyLlama Q/K/V extraction OK`.

Track C Step 3 was then added to the same script. It applies per-tensor
symmetric INT8 quantization to selected `q_head` and `k_head`, leaves `v_head`
as float32, reports Q/K scales, reconstruction errors, and dequantized score
error. The verified default run printed `q_scale 4.898416623473e-02`,
`k_scale 1.743172481656e-02`, Q reconstruction max error
`2.44865417e-02`, K reconstruction max error `8.71065259e-03`,
score dequant max error `2.65718549e-02`, score dequant mean error
`5.51600056e-03`, and `TinyLlama Q/K quantization OK`.

Track C Step 4 was implemented for the current single-tile design with
`model/export_real_vectors.py`. It exports a real TinyLlama-derived vector
directory compatible with the current 3-kernel host flow:
`q_tile.txt`, `k_tile.txt`, `kernel_meta.txt`, `score_raw.txt`,
`score_masked.txt`, `score_scaled.txt`, `score_softmax.txt`,
`score_packed.txt`, `metadata.json`, plus inspection/later-stage files
`q_float.txt`, `k_float.txt`, `v_full.txt`, and `attn_ref_float.txt`. The
verified run used the default 8-token prompt and wrote
`sim/real_tinyllama_tile/`. Local C++ benches passed against that directory for
`attention_score`, `mask_scale`, and `softmax`. This does not implement full
sequence tiling or `softmax @ V`; it provides real Q/K inputs for the current
single-tile 3-kernel design.

On 2026-05-02, `sim/real_tinyllama_tile/` was run on the real U55C with the
current 3-kernel XRT host and existing hardware bitstream UUID
`06fc7f72-fc9f-b542-28d3-aac2d65918ef`. Both the direct host invocation and
`host/run_hw.sh` passed with `XRT chain verification PASSED`. The direct run
timed attention score at 0.059 ms, mask+scale at 0.029 ms, softmax at 0.028 ms,
and total chain at 0.121 ms. The helper run timed attention score at 0.062 ms,
mask+scale at 0.024 ms, softmax at 0.028 ms, and total chain at 0.121 ms. This
verifies real TinyLlama-derived Q/K through score, mask+scale, and softmax; the
exported `v_full.txt` and `attn_ref_float.txt` remain for later Track B work and
are not consumed by the current xclbin.

The same day, `README.md`, `host/README.md`, `model/README.md`,
`docs/implementation_checklist.md`, and `CLAUDE.md` were refreshed to reflect
that real-card `sim/real_tinyllama_tile/` result. Track C Step 5 is now marked
complete for the current single-tile 3-kernel design only; multi-length real
vectors still require Track A tiling and full-row softmax.

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
- `hls/mask_and_scale/`
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
- `model/check_tinyllama_setup.py`
- `model/extract_tinyllama_qkv.py`
- `model/export_real_vectors.py`

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
- merged mask + scale:
  - `hls/mask_and_scale/mask_scale_core_hls.cpp`
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
- the pulled `AGENTS.md` history records successful `hw_emu` and real-card
  verification of the earlier single-tile four-kernel chain and the current
  single-tile three-kernel chain
- future `.xclbin` builds and hardware runs still require Linux, XRT, Vitis, and
  a matching U55C platform

## What Has NOT Been Confirmed Yet

Not yet confirmed:

- full-sequence host tiling over `S = 8, 64, 128, 256, 512`
- synthetic Track A regression through multiple vector directories
- full-row softmax kernel/design for `S > 64`
- `softmax @ V` / V weighted-sum stage
- XRT `hw_emu` run using `sim/real_tinyllama_tile/`
- CPU/GPU/FPGA baseline comparison for acceleration claims
- full TinyLlama attention path
- full TinyLlama model execution

So the project is now a proven **single-tile isolated FPGA demo**, but not yet a
full sequence-length attention accelerator or a full TinyLlama hardware runtime.
Track C Steps 1-3 are implemented and verified locally; Track C Step 4 is
implemented for the current single-tile 3-kernel design and verified with local
C++ benches plus a real-card XRT run.

## Best Next Step

Best next practical steps are Track A from
`docs/implementation_checklist.md`:

1. add Python full-sequence tiling with `softmax_full_rows(logits)`
2. add a full-row softmax kernel/design for `S > 64`
3. add the matching XRT host tiling loop for `S = 8, 64, 128, 256, 512`
4. add multi-vector regression and CPU/GPU/FPGA baseline tables
5. if more score-kernel speed is needed after that, prefer wider packing or
   on-chip fusion before chasing higher GEMM unroll, because the 64-bit packed
   interface produced the first material latency drop

## After That

After Track A works across sequence lengths, the next major engineering steps
are:

1. Track B: add `softmax @ V` with a V weighted-sum HLS kernel
2. export and verify `v_full.txt` and `attn_out.txt`
3. extend the existing Track C real-vector export beyond the current single-tile
   path once full-sequence tiling exists
4. run real vectors through the same FPGA pipeline and compare `attn_out` to
   PyTorch attention output
5. Track D: collect CPU/GPU/FPGA timing baselines and speedup tables
6. add double-buffering or merge kernels for performance
7. connect to a real TinyLlama attention subgraph
8. add KV-cache-aware decode flow
9. eventually integrate into a decoder-layer path

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
- Track C Steps 1-3 are already implemented in `model/check_tinyllama_setup.py`
  and `model/extract_tinyllama_qkv.py`
- Track C Step 4 has a current-design exporter in `model/export_real_vectors.py`
  that writes `sim/real_tinyllama_tile/` and has passed local C++ benches plus
  a real-card XRT run for attention score, mask+scale, and softmax

Agents should avoid redoing exploration that this file already captures unless
something materially changed.
