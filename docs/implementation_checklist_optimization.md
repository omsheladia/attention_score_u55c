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

- [ ] Freeze the current fused xclbin UUID, commit hash, and command lines used
      for timing.
- [ ] Add a short baseline table for:
  - `S = 8, 64, 128, 256, 512`
  - total time
  - kernel launch/wait sum
  - host/DMA/sync gap
  - launch count by kernel
- [ ] Enable XRT profiling for at least one `S=512` run.
- [ ] Capture whether time is dominated by:
  - kernel launch overhead
  - BO sync/write/read traffic
  - score compute
  - softmax
  - V weighted sum
- [ ] Record results in a new section of `docs/track_d_results.md`.
- [ ] Keep the same CPU/GPU baseline scripts so speedup claims stay apples to
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

- [ ] Allocate full-sequence device buffers instead of only tile buffers:
  - `q_full_bo` for `(S, 64)` INT8
  - `k_full_bo` for `(S, 64)` INT8
  - `v_full_bo` for `(S, 64)` FP32
  - `logits_bo` for `(S, S)` FP32, if still materializing logits
  - `probs_bo` for `(S, S)` FP32, if still materializing probabilities
  - `attn_out_bo` for `(S, 64)` FP32
- [ ] Copy Q/K/V to the device once per sequence.
- [ ] Read back only final `attn_out` for verification.
- [ ] Keep optional debug mode that can read back logits/probabilities for
      comparison during bring-up.

### Kernel interface changes

- [ ] Update `score_mask_scale_u55c_kernel` or add a new variant that accepts:
  - full Q buffer
  - full K buffer
  - full logits output buffer
  - `seq_len`
  - `query_base`
  - `key_base`
  - scale
- [ ] Kernel reads the proper tile from full Q/K using offsets.
- [ ] Kernel writes the scaled logit tile directly into the correct offset of
      the full `S x S` logits buffer.
- [ ] Update `softmax_full_row_u55c_kernel` to read/write row offsets in full
      logits/probability buffers.
- [ ] Update `v_weighted_sum_u55c_kernel` to read weights and V from full
      buffers using offsets, and to write partial/final output on device.

### Verification

- [ ] Verify synthetic `S = 8, 64, 128` in `hw_emu`.
- [ ] Verify synthetic `S = 8, 64, 128, 256, 512` on real U55C.
- [ ] Verify `sim/real_tinyllama_s16` and `sim/real_tinyllama_s64`.
- [ ] Compare timing against current fused Step 5 results.

## Expected result

- Same numerical output.
- Less host/DMA/sync gap.
- Still many launches, but far fewer host-visible intermediate transfers.

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

- [ ] Add a kernel variant that loops across all K/V chunks for one Q chunk.
- [ ] Accumulate `8 x 64` output on chip.
- [ ] Write final `attn_out` tile once.
- [ ] Avoid returning each partial output to host.

### Host update

- [ ] Replace nested per-tile launch loops with per-Q-chunk launches.
- [ ] Keep old path behind a debug flag until the new path is stable.

## Expected result

- Large launch-count reduction.
- Bigger improvement at larger `S`.
- Still materializes logits/probabilities unless Track O4 is also done.

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

- [x] Generate real TinyLlama vector directories:
  - `sim/real_tinyllama_s128`
  - `sim/real_tinyllama_s256`
  - `sim/real_tinyllama_s512`
- [x] Verify exported Q/K/V shapes, scales, and metadata.
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

- [ ] Track O1 complete
- [ ] Track O2 complete
- [ ] Same outputs as current fused Step 5 path
- [ ] Measurable reduction in host/DMA/sync gap

## Milestone 2 - Fewer Launches

- [ ] Track O3 pre-softmax multi-K kernel complete
- [ ] Track O3 V multi-K accumulation kernel complete
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

