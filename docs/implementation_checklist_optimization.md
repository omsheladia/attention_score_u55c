# FPGA Optimization Checklist

This checklist captures the next engineering work after the required
`implementation_checklist.md` scope. The current project already has a correct
one-head FPGA attention block:

```text
Q_int8 @ K_int8.T -> causal mask + scale -> full-row softmax -> softmax @ V
```

The remaining goal is performance: make the FPGA path meaningfully faster than
the one-head CPU baseline.

Current fused Step 5 result from `docs/track_d_results.md`:

| S | CPU full attn ms | fused FPGA total ms | CPU/fused FPGA |
|---:|---:|---:|---:|
| 8 | 0.0322 | 0.296 | 0.11x |
| 64 | 0.2920 | 2.364 | 0.12x |
| 128 | 2.5009 | 5.484 | 0.46x |
| 256 | 3.9523 | 17.307 | 0.23x |
| 512 | 12.3290 | 60.328 | 0.20x |

At `S=512`, the FPGA needs about a 5x end-to-end speedup to beat the current
CPU baseline. Small HLS tuning alone is unlikely to achieve that. The main
problem is architecture: too many kernel launches and too much host-visible HBM
staging between pipeline stages.

---

# Priority Summary

Recommended order:

1. **Track O1:** preserve a clean measurement baseline
2. **Track O2:** keep full Q/K/V/intermediates resident on device
3. **Track O3:** reduce kernel launch count with larger-grain kernels
4. **Track O4:** implement online softmax fused with V accumulation
5. **Track O5:** exploit parallelism across Q blocks and/or heads
6. **Track O6:** tune HLS math, memory, and clocks after the architecture is fixed
7. **Track O7:** run larger real TinyLlama vector sweeps and update report docs

Do not start with clock-frequency nudges or minor guidance-report cleanup. Those
may help later, but they will not remove the dominant launch/staging overhead.

---

# Current Bottleneck Model

For `S=512`, the current fused tiled path has roughly:

```text
512 score_mask_scale launches
 64 full_row_softmax launches
512 v_weighted_sum launches
= 1088 kernel launches
```

Current double buffering exists only for pass 1, the fused
`score_mask_scale_u55c_kernel` loop. Softmax and V weighted-sum still run as
separate staged passes with host-visible reads/writes between them.

Current HBM story:

```text
Q tile                 HBM[0]
K tile                 HBM[1]
fused scaled logits    HBM[3]
tile/full-row logits   HBM[4]
full-row probs/weights HBM[5]
V chunk                HBM[6]
V partial output       HBM[7]
```

The fused Step 5 branch removed the raw-score HBM round trip, but the design
still materializes logits/probabilities and replays V chunks through host-driven
tile loops.

---

# Track O1 - Measurement Baseline And Profiling

**Goal:** make every optimization measurable and comparable.

**Effort:** 0.5-1 day  
**Requires:** Linux + XRT + U55C for final numbers  
**Risk:** Low

## What to do

- [x] Freeze the current fused xclbin UUID, commit hash, and command lines used
      for timing.
- [x] Add a short baseline table for:
  - `S = 8, 64, 128, 256, 512`
  - total time
  - kernel launch/wait sum
  - host/DMA/sync gap
  - launch count by kernel
- [x] Enable XRT profiling for at least one `S=512` run.
- [x] Capture whether time is dominated by:
  - kernel launch overhead
  - BO sync/write/read traffic
  - score compute
  - softmax
  - V weighted sum
- [x] Record results in a new section of `docs/track_d_results.md`.
- [x] Keep the same CPU/GPU baseline scripts so speedup claims stay apples to
      apples.

## Expected result

- A clean "before optimization" reference.
- Clear evidence for which optimization should be attempted first.

---

# Track O2 - Device-Resident Full-Buffer Flow

**Goal:** stop moving every tile back through host memory between stages.

**Effort:** 1-2 days  
**Requires:** XRT host changes and xclbin rebuild  
**Risk:** Medium

## What to do

### Host layout

- [x] Allocate full-sequence device buffers instead of only tile buffers:
  - `q_full_bo` for `(S, 64)` INT8
  - `k_full_bo` for `(S, 64)` INT8
  - `v_full_bo` for `(S, 64)` FP32
  - `logits_bo` for `(S, S)` FP32, if still materializing logits
  - `probs_bo` for `(S, S)` FP32, if still materializing probabilities
  - `attn_out_bo` for `(S, 64)` FP32
- [x] Copy Q/K/V to the device once per sequence.
- [x] Read back only final `attn_out` for verification.
- [x] Keep optional debug mode that can read back logits/probabilities for
      comparison during bring-up.

### Kernel interface changes

- [x] Update `score_mask_scale_u55c_kernel` or add a new variant that accepts:
  - full Q buffer
  - full K buffer
  - full logits output buffer
  - `seq_len`
  - `query_base`
  - `key_base`
  - scale
- [x] Kernel reads the proper tile from full Q/K using offsets.
- [x] Kernel writes the scaled logit tile directly into the correct offset of
      the full `S x S` logits buffer.
- [x] Update `softmax_full_row_u55c_kernel` to read/write row offsets in full
      logits/probability buffers.
- [x] Update `v_weighted_sum_u55c_kernel` to read weights and V from full
      buffers using offsets, and to write partial/final output on device.

### 2026-05-04 Windows implementation status

- [x] Added `--resident` host mode for synthetic `--seq-len` and full-sequence
      `--vectors <dir>` inputs.
- [x] Added `--resident-debug` to optionally read back and compare full logits
      and full softmax probabilities during bring-up.
- [x] Added resident HLS top functions:
  - `score_mask_scale_resident_u55c_kernel`
  - `softmax_full_row_resident_u55c_kernel`
  - `v_weighted_sum_resident_u55c_kernel`
- [x] Updated `host/build_xclbin.sh` and `host/vpp_link.cfg` to include the
      resident kernels and HBM mappings.
- [x] Added resident HLS Tcl scripts for Linux-side `csim`/`csynth` runs.
- [x] Local g++ HLS benches pass for resident score/mask/scale, full-row
      softmax, and V weighted-sum, including a two-K-chunk resident V
      accumulation test.

### Verification

- [x] Verify synthetic `S = 8, 64, 128` in `hw_emu`.
      Note: only `S=8 --resident-debug` was run in `hw_emu`; it passed
      intermediate and final output verification, but took about `584179 ms`.
      Larger `hw_emu` resident runs were skipped because real hardware
      validation below covers the full sweep and the cycle simulator would take
      hours.
- [x] Verify synthetic `S = 8, 64, 128, 256, 512` on real U55C.
- [x] Verify `sim/real_tinyllama_s16` and `sim/real_tinyllama_s64`.
- [x] Compare timing against current fused Step 5 results.

### 2026-05-04 Linux/U55C validation status

- [x] Resident HLS `csim/csynth` passed for:
  - `score_mask_scale_resident_u55c_kernel`
  - `softmax_full_row_resident_u55c_kernel`
  - `v_weighted_sum_resident_u55c_kernel`
- [x] `hw_emu` xclbin link passed after shortening resident CU instance names
      in `host/vpp_link.cfg` with `nk=...:sms_res_1/sfr_res_1/vws_res_1`.
- [x] Real hardware xclbin build passed. UUID:
      `687b5e4a-591f-9d82-9263-e27bb7a727be`.
- [x] Real hardware routed timing met constraints: `WNS 0.003 ns`, `TNS 0`,
      `WHS 0.009 ns`, no failing endpoints.
- [x] Real hardware resident synthetic sweep passed:
  - `S=8 0.930 ms`
  - `S=64 4.131 ms`
  - `S=128 12.755 ms`
  - `S=256 45.809 ms`
  - `S=512 173.489 ms`
- [x] Real hardware resident real-vector runs passed:
  - `sim/real_tinyllama_s16 1.042 ms`
  - `sim/real_tinyllama_s64 3.369 ms`
- [x] O2 report artifacts copied under `docs/` with the `o2_` prefix.

## Expected result

- Same numerical output.
- Less host/DMA/sync gap.
- Still many launches, but far fewer host-visible intermediate transfers.

## Actual O2 result

- Correctness goal met on real hardware.
- Host/DMA/sync gap improved sharply at `S=512`: O1 `16.231 ms` -> O2
  `0.757 ms`.
- End-to-end performance regressed at `S=512`: O1 `60.328 ms` -> O2
  `173.489 ms`.
- Root cause is the resident V output accumulation path. The resident V HLS
  report misses II badly on the HBM read-modify-write output loop, and real
  hardware shows `v_weighted_sum_resident` dominating at `148.581 ms`.
- Track O3 should start with multi-K V accumulation that keeps the output tile
  on chip and writes final `attn_out` once.

---

# Track O3 - Reduce Kernel Launch Count

**Goal:** move from per-tile launches to larger work units.

**Effort:** 2-4 days  
**Requires:** HLS and host redesign  
**Risk:** Medium-high

## Current launch count

For `S=512`:

```text
score_mask_scale: 64 q_chunks * 8 k_chunks = 512 launches
softmax_full_row: 64 q_chunks = 64 launches
v_weighted_sum: 64 q_chunks * 8 k_chunks = 512 launches
total = 1088 launches
```

## Target launch count

First target:

```text
one pre-softmax launch per Q chunk     -> 64 launches
one softmax launch per Q chunk         -> 64 launches
one V accumulation launch per Q chunk  -> 64 launches
total                                  -> 192 launches
```

Better target:

```text
one fused attention launch per Q chunk -> 64 launches
```

Best target:

```text
one launch per sequence/head
```

## What to do

### Pre-softmax multi-K kernel

- [ ] Add a kernel variant that processes all K chunks for one Q chunk in one
      launch.
- [ ] Inputs:
  - full Q
  - full K
  - output logits buffer
  - `query_base`
  - `seq_len`
  - scale
- [ ] Internal loop:
  ```text
  for key_base in 0..S step 64:
      load K chunk
      compute score tile
      apply mask + scale
      write logits tile
  ```
- [ ] Verify against current fused per-tile path.

### V multi-K accumulation kernel

- [x] Add a kernel variant that loops across all K/V chunks for one Q chunk.
- [x] Accumulate `8 x 64` output on chip.
- [x] Write final `attn_out` tile once.
- [x] Avoid returning each partial output to host.
- [x] Clear the HLS gate before XRT integration.
- [x] Baseline complete dim=1/dim=2 accumulator partitioning was tried first:
      `v_weighted_sum_multik_u55c_kernel` passed csim, but csynth reported the
      hot accumulation loop at achieved II=3 vs target II=1.
- [x] Full 64-way unroll of the dot-product loop plus complete local
      `weights_local`/`v_local` partitioning was tried and still reports II=3;
      the remaining blocker is the final `acc[row][dim] += partial` update.
- [x] Dim-outer/row-unrolled version from commit `1a3a7af` improves Fmax to
      `283.37 MHz`, but still reports achieved II=3 vs target II=1 on
      `VITIS_LOOP_235_8`; resource estimate rises to `858 DSP`,
      `478810 FF`, `176824 LUT`.
- [x] Simple `#pragma HLS DEPENDENCE variable=acc inter false` on the
      dim-pipelined loop was tried, recognized by analysis, and removed after
      it still reported achieved II=3 on `VITIS_LOOP_235_8`.
- [x] Restructure the final accumulator update so the hot loop reaches II=1:
      ping-pong accumulator uses `acc_next[row][dim] = acc[row][dim] + partial`
      in the hot loop and copies `acc_next` back to `acc` afterward.
- [x] Vitis HLS 2022.2 ping-pong result: csim PASS, csynth PASS, loop
      constraints satisfied, estimated Fmax `342.47 MHz`, hot loop
      `VITIS_LOOP_238_8` achieved II=1 vs target II=1.
- [x] Integrate into the XRT host behind `--multik`; old O2 resident V path
      remains available with `--resident`.
- [x] Add build support for the O3 multi-K kernel in the full profile.
- [x] Add lean `o3_multik` build profile using only resident score/mask/scale,
      resident full-row softmax, and O3 V multi-K.
- [x] Full-profile `hw_emu` xclbin build PASS.
- [x] `hw_emu` `S=8 --multik` PASS:
      `Resident attention output verification PASSED` and
      `XRT chain verification PASSED`.
- [x] Preserve O3 `hw_emu` artifacts:
      `docs/o3_multik_hw_emu_xclbin.info`,
      `docs/o3_multik_hw_emu_link_summary`, and
      `docs/o3_multik_hw_emu_s8_run.txt`.
- [x] Attempt real U55C full-profile hardware link; failed during Vivado
      `place_design`. Full-profile synthed utilization shows
      `3389 / 9024 DSP = 37.56%`, with `vws_mk_1` at `375302 LUT`,
      `596554 REG`, and `2580 DSP`.
- [x] Attempt real U55C lean `o3_multik` hardware link; failed during Vivado
      `place_design` with a CLB packing/pblock capacity error:
      `37424 CLBs` available vs `40471 CLBs` required by unplaced instances,
      plus `7670` control sets.
- [ ] Reduce O3 V multi-K placement footprint before the next real hardware
      build. Recommended first pass: factor-32 or factor-16 column parallelism
      with local/ping-pong accumulation, then re-check HLS II and resources.
- [ ] Re-run lean `o3_multik` real hardware link after narrowing the datapath.
- [ ] If lean hardware places, run real U55C verification and timing at
      `S=8, 64, 128, 256, 512`.

### Host update

- [x] Replace nested per-tile V launches with one per-Q-chunk launch for
      `--multik`.
- [x] Keep old O2 resident V path available as `--resident` until the new path
      is stable.
- [x] Add `host/vpp_link_o3_multik.cfg` for a smaller hardware build target.

## Expected result

- Large launch-count reduction.
- Bigger improvement at larger `S`.
- Still materializes logits/probabilities unless Track O4 is also done.
- Current blocker: the O3 V multi-K ping-pong datapath passes HLS and `hw_emu`
  but is too large or too control-set-heavy to place on the current U55C shell.

---

# Track O4 - Online Softmax Fused With V

**Goal:** avoid materializing the full `S x S` probability matrix and compute
`softmax(QK) @ V` directly.

**Effort:** 4-7 days  
**Requires:** new HLS kernel and careful numerical verification  
**Risk:** High

## Why this matters

The current design computes:

```text
logits tile -> full-row softmax -> probability tile -> V weighted sum
```

This forces full-row probability storage and repeated V-kernel launches.
Online softmax allows a streaming form:

```text
for each K/V chunk:
    compute scaled masked score tile
    update row max
    update row exp sum
    update weighted V accumulator

attn_out = accumulator / exp_sum
```

## Algorithm sketch

For each query row:

```text
m_old = -inf
l_old = 0
acc_old[64] = 0

for each K/V chunk:
    scores = q @ k_chunk.T
    apply causal mask and scale

    m_new = max(m_old, max(scores))
    alpha = exp(m_old - m_new)
    p = exp(scores - m_new)
    l_new = l_old * alpha + sum(p)
    acc_new = acc_old * alpha + p @ v_chunk

    m_old = m_new
    l_old = l_new
    acc_old = acc_new

attn_out = acc_old / l_old
```

## What to do

- [ ] Add Python reference for online softmax attention.
- [ ] Verify online reference matches current brute-force reference at:
  - `S = 8`
  - `S = 64`
  - `S = 128`
  - `S = 256`
  - `S = 512`
- [ ] Add HLS kernel under `hls/online_attention/`.
- [ ] Process one Q chunk (`8 x 64`) per kernel launch first.
- [ ] Stream over all K/V chunks inside the kernel.
- [ ] Keep row-wise running max, exp sum, and `8 x 64` accumulator on chip.
- [ ] Write final `8 x 64` `attn_out` tile.
- [ ] Compare against the existing FPGA reference path within tolerance.
- [ ] Run Vitis HLS `csim` and `csynth`.
- [ ] Integrate XRT host path.
- [ ] Verify `hw_emu` at `S = 8, 64, 128`.
- [ ] Verify real U55C at `S = 8, 64, 128, 256, 512`.

## Expected result

- Eliminates full `S x S` probability materialization.
- Removes separate full-row softmax and V weighted-sum launches.
- Produces final `attn_out` directly.
- This is the first optimization likely to move the FPGA close to or past CPU
  for larger sequence lengths.

---

# Track O5 - Parallelism Across Q Blocks Or Heads

**Goal:** use more of the U55C. The current fused design uses only a small
fraction of available DSP/LUT resources.

**Effort:** 2-5 days depending on approach  
**Requires:** xclbin rebuild and HBM planning  
**Risk:** Medium

## What to do

### Replicate compute units

- [ ] Check current routed utilization from `docs/PostRouteKernelUtilization.rpt`.
- [ ] Decide whether to replicate:
  - online attention compute unit
  - score/mask/scale compute unit
  - V weighted-sum compute unit
- [ ] Start with 2 compute units, then 4 if timing/resources allow.
- [ ] Assign separate HBM banks or non-conflicting buffer regions.
- [ ] Update host to dispatch multiple Q chunks concurrently.

### Multi-head parallelism

- [ ] Extend input export and host layout to carry multiple heads.
- [ ] Run independent heads on separate compute units.
- [ ] Verify grouped-query attention mapping is still correct for TinyLlama.
- [ ] Compare throughput in heads/sec, not just one-head latency.

## Expected result

- Better use of available DSPs and HBM bandwidth.
- More credible FPGA advantage, especially when running multiple heads.

---

# Track O6 - HLS Math, Memory, And Clock Tuning

**Goal:** polish the kernels after launch count and data movement are fixed.

**Effort:** 1-3 days  
**Requires:** HLS/Vitis rebuilds  
**Risk:** Medium

## What to do

### Softmax reduction path

- [ ] Address the `sum_exp` recurrence warnings in:
  - `hls/softmax/`
  - `hls/softmax_full_row/`
  - future `hls/online_attention/`
- [ ] Try a tree reduction or partial sums per row.
- [ ] Compare latency, II, Fmax, and numerical error.

### Numeric format

- [ ] Evaluate whether FP32 is required throughout online softmax/V.
- [ ] Try fixed-point or mixed precision only after FP32 correctness is locked.
- [ ] Record max/mean error versus PyTorch and current CPU reference.

### Memory burst tuning

- [ ] Review Vitis guidance for missed bursts.
- [ ] Ensure full-buffer kernels use aligned 512-bit accesses where possible.
- [ ] Keep Q/K/V layouts contiguous for the access patterns used in HLS.

### Clock tuning

- [ ] Consider `--kernel_frequency` only after timing is clean.
- [ ] Use Vitis guidance as a hint, not as a correctness requirement.
- [ ] Re-run post-route timing and hardware validation after any clock change.

## Expected result

- Smaller incremental speedups after the main architecture changes.
- Cleaner Vitis guidance reports.
- Higher confidence for presentation/report claims.

---

# Track O7 - Larger Real TinyLlama Vector Coverage

**Goal:** strengthen the demo by showing that the optimized FPGA path works on
larger real TinyLlama-derived inputs, not only synthetic `--seq-len` data.

**Effort:** 0.5-1 day after the target optimized path works  
**Requires:** Python TinyLlama environment and U55C run machine  
**Risk:** Low-medium

## What to do

- [ ] Generate real TinyLlama vector directories:
  - `sim/real_tinyllama_s128`
  - `sim/real_tinyllama_s256`
  - `sim/real_tinyllama_s512`
- [ ] Verify exported Q/K/V shapes, scales, and metadata.
- [ ] Run CPU/GPU baselines for those directories.
- [ ] Run FPGA optimized path for those directories.
- [ ] Compare final `attn_out` against quantized-pipeline reference and
      PyTorch full-float reference.
- [ ] Update:
  - `docs/track_d_results.md`
  - project presentation notes
  - `AGENTS.md`

## Expected result

- Stronger real-input story for the final presentation.
- Clear distinction between synthetic sequence sweeps and real TinyLlama vector
  sweeps.

---

# Suggested Milestones

## Milestone 1 - Low-Risk Optimization

- [x] Track O1 complete
- [x] Track O2 complete
- [x] Same outputs as current fused Step 5 path
- [x] Measurable reduction in host/DMA/sync gap

## Milestone 2 - Fewer Launches

- [ ] Track O3 pre-softmax multi-K kernel complete
- [ ] Track O3 V multi-K accumulation kernel complete
      HLS, host integration, and `hw_emu S=8` are complete; real hardware link
      is blocked by placement until the V datapath is narrowed.
- [ ] `S=512` launch count reduced substantially
- [ ] FPGA timing improves over current fused Step 5 timing

## Milestone 3 - FPGA-Native Attention

- [ ] Track O4 online attention kernel complete
- [ ] No full `S x S` probability matrix materialized
- [ ] Final `attn_out` produced directly on FPGA
- [ ] Real U55C `S=512` approaches or beats one-head CPU baseline

## Milestone 4 - Scale Out

- [ ] Track O5 parallel Q blocks or heads complete
- [ ] Utilization is intentionally higher
- [ ] Throughput story is stronger than single-head latency alone

---

# Definition Of Done

An optimization should not be considered complete until:

- [ ] local C++ or Python reference verification passes
- [ ] Vitis HLS `csim` passes
- [ ] Vitis HLS `csynth` passes
- [ ] `hw_emu` passes for at least `S = 8, 64, 128`
- [ ] real U55C passes for `S = 8, 64, 128, 256, 512`
- [ ] at least one real TinyLlama vector directory passes
- [ ] timing is compared against the current fused Step 5 baseline
- [ ] resource and timing reports are copied under `docs/`
- [ ] `docs/track_d_results.md` and `AGENTS.md` are updated
