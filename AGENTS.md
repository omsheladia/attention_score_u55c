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
-> attn_out_fp32 in synthetic `--seq-len` mode and saved-vector `--vectors`
   mode when V/reference files are present
```

This is based on the attention-score path in the repo, mainly:

- `model/tinyllama.py`
- `model/export_fpga_vectors.py`

What is intentionally **not** implemented in this isolated flow:

- full Q/K/V projection path on FPGA
- live RoPE generation in the full runtime path
- multi-sequence real TinyLlama V-vector export
- decoder-layer integration
- full TinyLlama inference

## Key Interpretation Of "Correctness"

Current correctness means:

- the isolated 4-stage attention-score chain matches the local TinyLlama-derived
  software reference vectors in `--vectors <dir>` mode, including final
  `attn_out` when V/reference files are present
- the synthetic five-kernel `--seq-len` hardware-emulation path matches CPU
  references through final one-head `attn_out`

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
- historical note: `v++` existed locally, but no U55C `.xpfm` platform file was
  found under the checked local Vitis 2023.2 platform directories at that time.
  The platform path was later found under `/opt/xilinx/platforms/` and the
  design was rebuilt and verified with Vitis/XRT 2022.2.

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
directory compatible with the current host flow:
`q_tile.txt`, `k_tile.txt`, `kernel_meta.txt`, `score_raw.txt`,
`score_masked.txt`, `score_scaled.txt`, `score_softmax.txt`,
`score_packed.txt`, `metadata.json`, plus inspection/Track B files
`q_float.txt`, `k_float.txt`, `v_full.txt`, and `attn_ref_float.txt`. The
verified run used the default 8-token prompt and wrote
`sim/real_tinyllama_tile/`. Local C++ benches passed against that directory for
`attention_score`, `mask_scale`, and `softmax`. Track B later added FPGA
`softmax @ V` consumption of `v_full.txt` and comparison against
`attn_ref_float.txt`; full multi-sequence real-vector tiling is still future
work.

On 2026-05-02, `sim/real_tinyllama_tile/` was run on the real U55C with the
current 3-kernel XRT host and existing hardware bitstream UUID
`06fc7f72-fc9f-b542-28d3-aac2d65918ef`. Both the direct host invocation and
`host/run_hw.sh` passed with `XRT chain verification PASSED`. The direct run
timed attention score at 0.059 ms, mask+scale at 0.029 ms, softmax at 0.028 ms,
and total chain at 0.121 ms. The helper run timed attention score at 0.062 ms,
mask+scale at 0.024 ms, softmax at 0.028 ms, and total chain at 0.121 ms. This
was the earlier pre-Track-B verification of real TinyLlama-derived Q/K through
score, mask+scale, and softmax. The current five-kernel xclbin also consumes
`v_full.txt` and verifies final `attn_out` against `attn_ref_float.txt`.

The same day, `README.md`, `host/README.md`, `model/README.md`,
`docs/implementation_checklist.md`, and `CLAUDE.md` were refreshed to reflect
that real-card `sim/real_tinyllama_tile/` result. Track C Step 5 is now marked
complete for the current single-tile 3-kernel design only; multi-length real
vectors still require Track A host tiling with the full-row softmax kernel.

On 2026-05-03, Track D Steps 1-2 benchmark infrastructure was added:

- `model/benchmark_common.py` provides shared synthetic input generation,
  exported-vector loading, tiled CPU math, brute-force CPU math, validation,
  and timing helpers
- `model/benchmark_cpu.py` implements the CPU baseline for synthetic
  `S = 8, 64, 128, 256, 512` and `--vectors sim/real_tinyllama_tile`
- `model/benchmark_gpu.py` implements the CUDA baseline with the same inputs
  and exits cleanly when CUDA is unavailable
- verified CPU runs:
  - synthetic default run passed with zero tiled-vs-brute differences for
    softmax probabilities and `softmax @ V`
  - real-vector run matched `sim/real_tinyllama_tile/score_softmax.txt` with
    max difference `2.98023224e-08`
- after installing CUDA-enabled PyTorch (`torch 2.11.0+cu128`,
  `torch.version.cuda 12.8`), GPU timing was verified on an NVIDIA GeForce RTX
  3050 Laptop GPU
- verified GPU runs:
  - synthetic `S = 8, 64, 128, 256, 512` validated against CPU with max
    softmax diff `2.98023224e-07` and max `attn_out` diff `1.90734863e-06`
  - real-vector run matched `sim/real_tinyllama_tile/score_softmax.txt` with
    max difference `2.98023224e-08`
- these are software baselines only; final speedup claims should now compare
  them against the Track B five-kernel real U55C runs

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
host supports the legacy single-tile `--vectors <dir>` path and the tiled
synthetic `--seq-len <S>` path for these lengths.

Softmax correctness note:

- running softmax independently per 8 x 64 score tile is only correct for
  `S <= 64`
- for `S > 64`, softmax must normalize each query row across all `S` keys
- the existing `hls/softmax/` kernel is fixed at 8 x 64
- `hls/softmax_full_row/` now provides the standalone full-row kernel for
  `S <= 512`, and the XRT host tiled `--seq-len` path uses it

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
- `model/benchmark_common.py`
- `model/benchmark_cpu.py`
- `model/benchmark_gpu.py`

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

Current and planned vector/export additions from `docs/implementation_checklist.md`:

- `softmax_full_rows(logits)` in `model/attention_score_ref.py` is implemented
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
- full-row softmax:
  - `hls/softmax_full_row/softmax_full_row_hls.cpp`

Current and planned HLS additions from `docs/implementation_checklist.md`:

- `hls/mask_and_scale/` to merge causal mask and score scale is implemented
- `hls/softmax_full_row/` for full-row softmax up to `S = 512` is implemented
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

- automated synthetic Track A regression through multiple vector directories
- XRT `hw_emu` run using `sim/real_tinyllama_tile/`
- CPU/GPU/FPGA baseline comparison for final acceleration claims
- full TinyLlama attention path
- full TinyLlama model execution

So the project is now a proven **tiled one-head attention FPGA demo** for
synthetic sequence lengths up to `S=512`, with saved-vector real TinyLlama
`attn_out` verification in both single-tile and tiled full-sequence modes, but
not yet a full TinyLlama hardware runtime.
Track C Steps 1-5 are now implemented for the current one-head staged design.

On 2026-05-02, Track A Step 3 Part A was implemented in
`model/attention_score_ref.py`. New helpers include
`compute_full_attention_score(q_full, k_full, ...)`,
`softmax_full_rows(logits)`, and `brute_force_full_attention_score(...)`.
Running `python3 model/attention_score_ref.py --check-full-tiling` passed for
`S = 8, 64, 128, 256, 512` with zero max difference for raw scores, scaled
logits, and full-row softmax probabilities. This started as Python reference
work; the full-row HLS softmax kernel and XRT host tiling loop were added later
on 2026-05-02.

Later on 2026-05-02, `hls/softmax_full_row/` was added as the full-row softmax
kernel for `8 x S` rows with `S <= 512`. The original `hls/softmax/` tile
kernel remains unchanged. Local C++ bench passed for `S = 64, 128, 256, 512`,
partial rows, and masked logits. Vitis HLS 2022.2 `csim` and `csynth` passed
for `softmax_full_row_u55c_kernel` with estimated `315.96 MHz`, `9 DSP`,
`1 BRAM_18K`, `2 URAM`, `3693 FF`, `5772 LUT`, and all loop constraints
satisfied in standalone HLS. This kernel was then wired into the XRT host,
xclbin build script, and link config.

Later still on 2026-05-02, the XRT host tiled path was implemented and verified
in `hw_emu`. `host/attention_score_chain_xrt.cpp` now keeps the legacy
single-tile `--vectors` mode and adds `--seq-len <S>` for synthetic
full-sequence runs. `host/build_xclbin.sh` and `host/vpp_link.cfg` now include
`softmax_full_row_u55c_kernel`, mapped with full-row logits on `HBM[4]` and
full-row probabilities on `HBM[5]`. A fresh four-kernel `hw_emu` xclbin was
built successfully; `xclbinutil --info` reports content `HW Emulation Binary`,
UUID `1f6b30da-9112-3c47-7e52-a4a149705c10`, and kernels
`attention_score_u55c_kernel`, `mask_scale_u55c_kernel`,
`softmax_u55c_kernel`, and `softmax_full_row_u55c_kernel`. The host compile
passed. `hw_emu` passed for `--seq-len 8`, `--seq-len 64`, and `--seq-len 128`
with `Tiled sequence verification PASSED` and `XRT chain verification PASSED`;
the `S=128` run is the first verified two-K-chunk case. The original
single-tile `--vectors attention_score_u55c/sim/attention_score_tile` path also
passed under the new four-kernel `hw_emu` xclbin. Real hardware rebuild/run for
this four-kernel tiled design was completed next.

The four-kernel real hardware xclbin was then built successfully. Hardware link
took about 45 minutes and produced `build/attention_score_chain.xclbin` with
content `Bitstream`, UUID `d0099c1c-0481-4332-4772-0a89999bc1f8`, kernels
`attention_score_u55c_kernel`, `mask_scale_u55c_kernel`,
`softmax_u55c_kernel`, and `softmax_full_row_u55c_kernel`, and HBM banks
`HBM[0]` through `HBM[5]` used. The real U55C tiled synthetic sweep passed for
`S = 8, 64, 128, 256, 512`, each printing `Tiled sequence verification PASSED`
and `XRT chain verification PASSED`. Timings were:

- `S=8`: total 3.321 ms, 1 tile, 301.11 tiles/sec, 19,271.30 scores/sec
- `S=64`: total 1.931 ms, 8 tiles, 4,142.93 tiles/sec, 2,121,180.74 scores/sec
- `S=128`: total 4.622 ms, 32 tiles, 6,923.41 tiles/sec, 3,544,785.81 scores/sec
- `S=256`: total 13.353 ms, 128 tiles, 9,585.86 tiles/sec, 4,907,960.76 scores/sec
- `S=512`: total 49.444 ms, 512 tiles, 10,355.15 tiles/sec, 5,301,836.42 scores/sec

Approximate staged-kernel HBM traffic estimates were 11,264 bytes for `S=8`,
118,784 bytes for `S=64`, 475,136 bytes for `S=128`, 1,900,544 bytes for
`S=256`, and 7,602,176 bytes for `S=512`. The original single-tile synthetic
and real TinyLlama vector modes also passed on the new four-kernel bitstream.

On 2026-05-03, Track A Step 4 was implemented in the XRT host for the tiled
synthetic path. `host/attention_score_chain_xrt.cpp` now uses two BO sets for
the pass-1 score/mask-scale loop (`q`, `k`, raw score, and scaled score) and
alternates K tiles through those sets. Full-row softmax remains after all K
chunks for a Q chunk, because it needs the complete S-wide row for correctness.
The host binary rebuilt successfully, the Python full-tiling check still
passed, and the existing four-kernel real U55C bitstream passed:

- `S=8`: total 0.510 ms, 1 tile, 1,960.78 tiles/sec, 125,490.20 scores/sec
- `S=64`: total 2.307 ms, 8 tiles, 3,467.71 tiles/sec, 1,775,465.97 scores/sec
- `S=128`: total 5.238 ms, 32 tiles, 6,109.20 tiles/sec, 3,127,911.42 scores/sec
- `S=256`: total 11.709 ms, 128 tiles, 10,931.76 tiles/sec, 5,597,062.09 scores/sec
- `S=512`: total 35.992 ms, 512 tiles, 14,225.38 tiles/sec, 7,283,396.31 scores/sec

All Step 4 real-card runs printed `Tiled sequence verification PASSED` and
`XRT chain verification PASSED`. The legacy single-tile synthetic vector mode
and real TinyLlama vector mode also still passed. Per-kernel mask/scale timing
is now an overlapped host-observed window; use `total_chain` for Track A Step 4
comparisons.

Later on 2026-05-03, Track B Steps 1-2 were implemented and verified locally:

- `model/attention_score_ref.py` now includes:
  - `deterministic_v_matrix(row_count)`
  - `compute_v_weighted_sum_partial(weights_tile, v_tile)`
  - `compute_v_weighted_sum(softmax_weights, v_full)`
  - `brute_force_v_weighted_sum(softmax_weights, v_full)`
  - `compute_full_attention(q_full, k_full, v_full, ...)`
- `model/export_attention_score_vectors.py` now writes Track B files:
  - `v_full.txt`
  - `v_tile.txt`
  - `v_partial_expected.txt`
  - `attn_out.txt`
  - optional full-sequence export via `--seq-len <S>`
- Python verification with `python3 model/attention_score_ref.py --check-full-tiling`
  passed for `S = 8, 64, 128, 256, 512`; max `attn_out` diff was
  `2.77555756e-17`.
- `hls/v_weighted_sum/` was added as the Track B partial weighted-sum kernel.
  It consumes one `8 x 64` softmax-weight tile and one `64 x 64` V chunk, and
  produces one partial `8 x 64` contribution.
- `bash attention_score_u55c/host/run_local_csim.sh` passed, including
  `v_weighted_sum test PASSED, max diff 5.96046e-08`.
- Vitis HLS 2022.2 for `v_weighted_sum_u55c_kernel` passed `csim` and
  `csynth`: estimated `342.47 MHz`, latency `5600 cycles`, resources
  `320 DSP`, `49 BRAM_18K`, `0 URAM`, `50454 FF`, `32358 LUT`, and all loop
  constraints satisfied.

Track B Step 3 was then integrated for synthetic `--seq-len` XRT hardware
emulation:

- `host/attention_score_chain_xrt.cpp` now opens
  `v_weighted_sum_u55c_kernel`, generates deterministic synthetic V, slices
  per-K V chunks, runs a three-pass tiled flow, accumulates partial V outputs on
  the host, and verifies final `attn_out` against a CPU `softmax @ V`
  reference.
- `host/build_xclbin.sh` now compiles/links `v_weighted_sum_u55c_kernel`.
- `host/vpp_link.cfg` maps `v_weighted_sum_u55c_kernel_1.weights_tile` to
  `HBM[5]`, `v_tile` to `HBM[6]`, and `out_tile` to `HBM[7]`; the five-kernel
  `hw_emu` xclbin uses HBM banks `[0]` through `[7]`.
- The five-kernel `hw_emu` xclbin build passed with the U55C platform
  `/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`.
- The XRT host build passed.
- `hw_emu` passed for `--seq-len 8`: q_chunks=1, k_chunks=1,
  `total_chain 65214.673 ms`, `Attention output verification PASSED`, and
  `XRT chain verification PASSED`.
- `hw_emu` passed for `--seq-len 64`: q_chunks=8, k_chunks=1,
  `total_chain 580737.286 ms`, `Attention output verification PASSED`, and
  `XRT chain verification PASSED`.
- `hw_emu` passed for `--seq-len 128`: q_chunks=16, k_chunks=2,
  `total_chain 1657498.336 ms`, `Attention output verification PASSED`, and
  `XRT chain verification PASSED`.
- `hw_emu` printed `Unable to find emconfig.json. Using default device ...`;
  this did not block the passing runs.
- The five-kernel real hardware xclbin build passed. Hardware link took
  `0h 59m 42s`. `xclbinutil --info` reports content `Bitstream`, UUID
  `45a627bf-33e6-b364-b9ef-2d17b4f5e0e1`, clocks `hbm_aclk 450 MHz`,
  `KERNEL_CLK 500 MHz`, `DATA_CLK 300 MHz`, and HBM banks `[0]` through `[7]`
  used.
- Real U55C synthetic `--seq-len` runs passed:
  - `S=8`: q_chunks=1, k_chunks=1, `total_chain 0.868 ms`
  - `S=64`: q_chunks=8, k_chunks=1, `total_chain 2.328 ms`
  - `S=128`: q_chunks=16, k_chunks=2, `total_chain 7.226 ms`
  - `S=256`: q_chunks=32, k_chunks=4, `total_chain 21.269 ms`
  - `S=512`: q_chunks=64, k_chunks=8, `total_chain 84.190 ms`
  All printed `Tiled sequence verification PASSED`,
  `Attention output verification PASSED`, and `XRT chain verification PASSED`.
- `--vectors <dir>` mode now runs the V weighted-sum stage when the vector
  directory contains `v_full.txt` and either `attn_out.txt` or
  `attn_ref_float.txt`. Because tile softmax writes HBM[4] and the V kernel
  weights port is mapped to HBM[5], the host copies the tile-softmax output
  into the V weights BO before launching `v_weighted_sum_u55c_kernel`.
- Real U55C saved-vector runs passed:
  - `sim/attention_score_tile`: `attn_out.txt`, `total_chain 0.345 ms`,
    `Attention output verification PASSED`, `XRT chain verification PASSED`
  - `sim/real_tinyllama_tile`: `attn_ref_float.txt`, `total_chain 0.258 ms`,
    `Attention output verification PASSED`, `XRT chain verification PASSED`

On 2026-05-03, Track C was extended beyond the legacy single-tile
`sim/real_tinyllama_tile/` directory:

- `model/export_real_vectors.py` now supports full-sequence tiled export when
  `--seq-len <S>` is provided, up to the current full-row softmax limit
  `S <= 512`.
- full-sequence real-vector directories include `q_full.txt`, `k_full.txt`,
  `v_full.txt`, full `score_raw.txt`, full `score_scaled.txt`, full
  `score_softmax.txt`, quantized-pipeline `attn_out.txt`, full-float
  TinyLlama `attn_ref_float.txt`, and first-tile compatibility files.
- The XRT host now detects `q_full.txt`/`k_full.txt` in `--vectors <dir>` mode
  and runs the tiled full-sequence path instead of the legacy single-tile path.
- `model/check_tinyllama_setup.py`, `model/extract_tinyllama_qkv.py`, and
  `model/export_real_vectors.py` now pass `torch_dtype=...` to Transformers
  for compatibility with the 4.x stack used in this environment.
- Generated and verified real TinyLlama full-sequence vector directories:
  - `sim/real_tinyllama_s16`: `S=16`, q_chunks=2, k_chunks=1,
    quantized-vs-float `attn_out` max error `2.67604024e-04`
  - `sim/real_tinyllama_s64`: `S=64`, q_chunks=8, k_chunks=1,
    quantized-vs-float `attn_out` max error `1.62767614e-04`
- Real U55C runs passed using the existing five-kernel hardware xclbin:
  - `--vectors sim/real_tinyllama_s16`: total `0.985 ms`,
    `Tiled sequence verification PASSED`, `Attention output verification PASSED`,
    `XRT chain verification PASSED`
  - `--vectors sim/real_tinyllama_s64`: total `2.510 ms`,
    `Tiled sequence verification PASSED`, `Attention output verification PASSED`,
    `XRT chain verification PASSED`
- Environment note: default `/usr/bin/python3` on this shared machine did not
  have `torch`, `transformers`, or `sentencepiece`. Verification used the
  existing `/home/advent/kmhatre/DT/bin/python` for `torch` plus temporary
  `/tmp/attention_score_trackc_pydeps` for `transformers`/`sentencepiece`, and
  `/tmp/attention_score_hf_cache` for the model cache.

## Best Next Step

Track A Steps 1-4 are complete for the current staged design. Track A Step 5
remains a future fusion/dataflow milestone, not required before starting
dependent work. Best next practical steps are:

1. add/update CPU/GPU/FPGA comparison tables using the real U55C sequence sweep
2. add Track D comparison tables for synthetic and real-vector FPGA runs
3. if more score-kernel speed is needed after that, prefer wider packing or
   on-chip fusion before chasing higher GEMM unroll, because the 64-bit packed
   interface produced the first material latency drop

## After That

With Track A Steps 1-4 working across synthetic sequence lengths, the next
major engineering steps are:

1. Track D: collect CPU/GPU/FPGA timing baselines and speedup tables
2. run larger real-vector lengths such as `S=128`, `S=256`, and `S=512`
   if the demo needs a broader real-input sweep
3. compare real-vector FPGA `attn_out` timings with CPU/GPU baselines
4. merge kernels for performance if the staged HBM path becomes the bottleneck
5. connect to a real TinyLlama attention subgraph
6. add KV-cache-aware decode flow
7. eventually integrate into a decoder-layer path

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
- Track D Step 1 CPU baseline is implemented and locally verified in
  `model/benchmark_cpu.py`
- Track D Step 2 CUDA baseline is implemented and verified on the local RTX
  3050 Laptop GPU in `model/benchmark_gpu.py`

Agents should avoid redoing exploration that this file already captures unless
something materially changed.
