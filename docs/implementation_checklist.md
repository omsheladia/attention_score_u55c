# Implementation Checklist

This checklist covers four tracks:

- **Track A — Speedup:** make the existing pipeline faster and capable of real
  sequence lengths using synthetic inputs
- **Track B — Complete attention block:** add `softmax @ V` as a 5th stage so
  the FPGA produces a complete attention output, not just weights
- **Track C — Real inputs:** replace synthetic Q/K/V vectors with real TinyLlama
  inference data for a stronger demo
- **Track D — Baseline and benchmarking:** measure FPGA latency against CPU/GPU
  to produce the "how much faster?" numbers the course will require

**Priority order: A → B → C → D**

Track B is more important than Track C. Without Track B the FPGA produces
incomplete attention weights that nothing downstream can use. Track C only
improves the data source — the computation is the same either way. Complete
the full attention block first, then worry about where the inputs come from.

---

# Track A — Speedup with Synthetic Inputs

Steps are ordered easiest to hardest. Steps 1–3 are the priority target.
Step 4 is a stretch goal. Step 5 is a future milestone, not a 2-day task.

---

## Step 1 — Increase GEMM Unroll Factor

**Effort:** 30 minutes  
**Requires Vitis re-synthesis:** Yes (Linux only)  
**Can be code-edited on Windows:** Yes

### What to do

- [ ] Open `hls/attention_score/attention_score_core_hls.cpp`
- [ ] Find line with `#pragma HLS UNROLL factor=8`
- [ ] Change to `factor=16` (safe, ~2× GEMM speedup, 64 DSPs) or `factor=64`
      (full unroll, ~8× GEMM speedup, 256 DSPs — still fits on U55C)
- [ ] Re-run Vitis csynth for the `attention_score` kernel
- [ ] Confirm new DSP count in synthesis report
- [ ] Confirm csim still passes
- [ ] Record new MHz and DSP numbers in `CLAUDE.md` verification table

### Expected result

| Unroll | Cycles/dot product | DSPs |
|--------|--------------------|------|
| 8 (current) | 8 | 32 |
| 16 | 4 | 64 |
| 64 | 1 | 256 |

---

## Step 2 — Merge Causal Mask + Score Scale Into One Kernel

**Effort:** 2–3 hours  
**Requires Vitis re-synthesis:** Yes (Linux only)  
**Can be code-edited and locally bench-tested on Windows:** Yes

### What to do

#### HLS kernel
- [ ] Create `hls/mask_and_scale/mask_scale_core_hls.hpp`
- [ ] Create `hls/mask_and_scale/mask_scale_core_hls.cpp` with a single
      `kScoreRowsPerTile × kScoreColsPerTile` loop that applies the causal mask
      condition and multiplies by `total_scale` in one pass:
      ```
      bool masked = (row >= query_row_count) ||
                    (col >= key_col_count)   ||
                    ((key_pos_base + col) > (query_pos_base + row));
      out[row][col] = masked ? -1e9f : (float)in[row][col] * total_scale;
      ```
- [ ] Add `#pragma HLS PIPELINE II=1` to the inner loop
- [ ] Write AXI interface pragmas (m_axi for in/out, s_axilite for scalars)

#### Testbench
- [ ] Create `hls/mask_and_scale/tb_mask_scale.cpp`
- [ ] Load reference vectors from `sim/attention_score_tile/score_raw.txt`
      and `sim/attention_score_tile/score_scaled.txt`
- [ ] Compile and run locally with g++ — confirm pass

#### Vitis
- [ ] Create `hls/mask_and_scale/run_hls.tcl` (copy pattern from existing kernels)
- [ ] Run csim — confirm pass
- [ ] Run csynth — record MHz and DSP count

#### Host app
- [ ] Remove `mask_kernel` and `scale_kernel` XRT objects from
      `host/attention_score_chain_xrt.cpp`
- [ ] Add `mask_scale_kernel` XRT object
- [ ] Remove `masked_score_bo` intermediate buffer (now on-chip)
- [ ] Update kernel launch sequence: `score → mask_scale → softmax`
- [ ] Update `vpp_link.cfg` to include the new kernel and remove the old two
- [ ] Rebuild xclbin and host app

### Expected result
- Eliminates one HBM read + write per tile (~4 KB saved per tile call)
- Reduces kernel launch overhead from 4 calls to 3 calls per tile

---

## Step 3 — Add Host Tiling Loop for Variable Sequence Length

**Effort:** 4–6 hours  
**Requires Vitis/XRT:** Only for FPGA testing — Python side runs locally now  
**Can be fully tested on Windows (Python side):** Yes

### Part A — Python reference tiling (do this first)

- [ ] Open `model/attention_score_ref.py`
- [ ] Add a new function `compute_full_attention_score(q_full, k_full)` that
      uses a **two-pass** structure. Softmax must normalize across all S keys
      for each query row — running it per tile is wrong for S > 64 because each
      tile only sees 64 of the S keys, so each tile row sums to 1 instead of
      the full row summing to 1:
  ```
  # pass 1 — collect all scaled/masked logits
  logits = zeros(S, S)
  for q_chunk in [0 .. ceil(S/8)):
      for k_chunk in [0 .. ceil(S/64)):
          tile = compute_attention_score_tile(Q[q_chunk], K[k_chunk])
          tile = apply_causal_mask(tile, query_pos_base, key_pos_base)
          tile = scale_scores(tile, q_scale, k_scale)
          logits[q_chunk*8.., k_chunk*64..] = tile

  # pass 2 — softmax across the full row (all S keys)
  # NOTE: the existing softmax_rows() helper in attention_score_ref.py is
  # tile-shaped (padded to 8×64). It cannot be called with key_col_count=S
  # for S > 64. Add a new helper:
  #   def softmax_full_rows(logits):  # logits shape (S, S), no padding
  #       for each row: subtract max, exp, divide by sum
  #       return probs shape (S, S)
  softmax_weights = softmax_full_rows(logits)
  ```
- [ ] Add `softmax_full_rows(logits)` to `model/attention_score_ref.py` that
      operates on an unpadded `(S, S)` logit matrix — no fixed tile size assumed
- [ ] Add a test script or CLI flag that runs the full tiling at:
  - S = 8   (single tile sanity check — one K-chunk, passes either way)
  - S = 64  (8 Q-chunks × 1 K-chunk — still one K-chunk, safe boundary)
  - S = 128 (16 × 2 — first case where per-tile softmax would silently fail)
  - S = 256 (32 × 4)
  - S = 512 (64 × 8)
- [ ] For each S, verify the tiled output matches a brute-force reference
      (compute the full score matrix directly in Python and compare)

### Part B — XRT host tiling loop

- [ ] Open `host/attention_score_chain_xrt.cpp`
- [ ] Add `seq_len` as a command-line argument (`--seq-len`)
- [ ] Replace the single-tile kernel launch with a **two-pass** loop.
      The score/mask/scale kernels run per tile as before, but softmax must
      see the full row of S logits — not just one 64-wide tile — to normalize
      correctly. Running the softmax kernel per tile is only correct when S ≤ 64:
  ```
  // pass 1 — score, mask, scale for all (q, k) tile pairs
  // result: full S×S scaled logit matrix in HBM
  for q_chunk in [0 .. ceil(S/8)):
      for k_chunk in [0 .. ceil(S/64)):
          DMA Q[q_chunk] and K[k_chunk] to device
          run score kernel → mask kernel → scale kernel
          DMA scaled logit tile back
          write into logits[q_chunk*8.., k_chunk*64..]

  // pass 2 — softmax across full rows (all S keys per query row)
  // NOTE: the existing softmax kernel has a fixed 8×64 array and cannot
  // accept key_col_count > 64. A new full-row softmax kernel is required
  // (see Step 3B below) before this pass works for S > 64.
  for q_chunk in [0 .. ceil(S/8)):
      DMA logits[q_chunk*8.., 0..S] to device   // full row, all K chunks
      run full_row_softmax kernel with key_col_count = S
      DMA softmax output back
      write into softmax_weights[q_chunk*8..]
  ```
- [ ] Allocate output buffer large enough for the full `S × S` logit matrix
      and a separate `S × S` softmax weights matrix
- [ ] Update reference loading to generate full-sequence reference vectors
      (use the Python tiling function from Part A to produce expected outputs)
- [ ] Test in hw_emu at S = 8, 64, 128

### Part C — Full-row softmax kernel (blocker for S > 64)

The existing softmax kernel (`hls/softmax/softmax_core_hls.cpp`) has fixed
on-chip arrays sized at `kScoreColsPerTile = 64`. Calling it with
`key_col_count = S` for S > 64 would index out of bounds. A new kernel is
needed that can handle a full row of up to S = 512 keys.

Two implementation options:

**Option A — Wider fixed buffer (simpler)**
- New kernel with on-chip array sized `[8][512]` instead of `[8][64]`
- Accept `key_col_count` up to 512 at runtime
- Same three-pass loop structure (max → exp → divide)
- Cost: 16 KB on-chip SRAM (8 rows × 512 cols × 4 bytes) instead of 2 KB
  (8 × 64 × 4), but straightforward

**Option B — Online/tiled softmax (more complex, no extra SRAM)**
- Process K-chunks one at a time, tracking running max and running sum
- Uses the numerically stable online softmax algorithm:
  ```
  for each K-chunk:
      update running_max = max(running_max, chunk_max)
      rescale previous sum: sum *= exp(old_max - new_max)
      sum += sum of exp(x - running_max) for x in chunk
  final weights = exp(x - running_max) / sum
  ```
- No large on-chip buffer needed, but more complex to implement and verify

**Recommendation:** Use Option A for correctness first, switch to Option B
if on-chip SRAM becomes a constraint at large S.

- [ ] Implement chosen option as `hls/softmax_full_row/softmax_full_row_hls.cpp`
- [ ] Write testbench verifying correctness at S = 64, 128, 256, 512
- [ ] Run csim and csynth — confirm pass and record resource usage
- [ ] Update host app pass 2 to use the new kernel instead of the old one
- [ ] Keep the original `softmax` kernel unchanged — it is still used for the
      single-tile path (S ≤ 64) and existing testbenches depend on it

### Expected result
- Project can now process real sentence-length inputs
- Performance scales predictably: tile count = `ceil(S/8) × ceil(S/64)`
- Softmax correctly normalizes across all S keys for every query row

---

## Step 4 — Double-Buffer DMA Transfers *(stretch goal)*

**Effort:** 1 day (including debugging)  
**Requires:** XRT on Linux  
**Prerequisite:** Step 3 complete

### What to do

- [ ] Allocate two sets of Q/K input buffer objects (`q_bo[2]`, `k_bo[2]`)
- [ ] Allocate two sets of output buffer objects (`softmax_bo[2]`)
- [ ] Pre-load tile 0 into buffer set 0 before the loop starts
- [ ] Inside the tiling loop:
  - Launch kernel on buffer set `i % 2`
  - Simultaneously DMA tile `i+1` into buffer set `(i+1) % 2`
  - Wait for kernel on buffer set `i % 2`
  - DMA result back from buffer set `i % 2`
- [ ] Verify output matches Step 3 output exactly

### Expected result
- DMA latency for tile N+1 is hidden behind compute time for tile N
- Benefit grows with sequence length (more tiles = more overlap opportunity)

---

## Step 5 — Merge Pre-Softmax Stages Into One Dataflow Kernel *(future milestone)*

**Effort:** 2–3 days minimum  
**Risk:** High — softmax multi-pass structure conflicts with HLS DATAFLOW rules  
**Do not attempt within a 2-day sprint**

**Scope note:** This covers merging the score GEMM + mask/scale stages (pass 1
of the two-pass pipeline). Softmax and the V kernel remain separate because
softmax requires a full row of S logits and the V kernel requires full-row
softmax weights — neither fits the single-pass dataflow model naturally.

### What needs to happen

- [ ] Merge score GEMM + mask/scale into one dataflow kernel (3 or 2 stages
      depending on whether Step 2 was done)
- [ ] Create `hls/score_and_mask_scale/score_mask_scale_hls.cpp`
      with sub-functions connected by on-chip FIFOs
- [ ] Add `#pragma HLS DATAFLOW` and verify Vitis does not reject it
- [ ] Update `vpp_link.cfg` to replace the separate kernels
- [ ] Rebuild xclbin, re-run all sequence length tests
- [ ] Confirm timing closure (softmax timing warning may worsen)

### Expected result
- Eliminates HBM traffic between score and mask/scale stages only
- These stages run as a pipeline — mask/scale starts while score finishes
- Softmax and V kernel remain separate — their HBM round-trips are unchanged

---

## Track A Priority Order Summary

| Step | Effort | Testable on Windows now | Do in 2 days? |
|------|--------|------------------------|---------------|
| 1 — UNROLL factor | 30 min | Code yes, verify needs Vitis | Yes |
| 2 — Merge mask+scale | 2–3 hrs | Local bench yes, csynth needs Vitis | Yes |
| 3 — Tiling loop | 4–6 hrs | Python side fully, XRT needs Linux | Yes |
| 4 — Double buffering | ~1 day | No (needs XRT) | Stretch |
| 5 — Dataflow merge | 2–3 days | No | No |

---

# Track B — Complete Attention Block (softmax @ V)

**Goal:** Add a 5th kernel that computes `softmax_weights @ V → attn_out`, making
the FPGA handle the complete attention computation for one head rather than
stopping at softmax weights.

**Why float32:** The FPGA already runs float32 in stages 3 and 4. V contains
actual content values that compound errors across 22 layers if quantized — keeping
it float32 is correct and consistent with the existing pipeline.

**Prerequisite:** Track A Step 3 (tiling loop) must be working first, because
the V multiply requires accumulating partial results across K/V-chunks.

**Effort:** ~1 day total (half a day if Track A Step 3 is already working).

---

## Track B Step 1 — Add softmax @ V to Python Reference

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [ ] Open `model/attention_score_ref.py`
- [ ] Add `compute_v_weighted_sum(softmax_weights, v_full)` function:
  - accepts `softmax_weights` shape `(S, S)` and `v_full` shape `(S, 64)`
  - for each Q-chunk (8 rows at a time):
    - initialize `accumulator[8][64] = 0`
    - for each K/V-chunk (64 rows at a time):
      - multiply `softmax_weights[q_chunk, k_chunk]` (8×64) by `V[k_chunk]` (64×64)
      - accumulate into `accumulator`
    - write `accumulator` into `attn_out[q_chunk*8 ..]`
  - returns `attn_out` shape `(S, 64)`
- [ ] Update `export_attention_score_vectors.py` to:
  - generate a full synthetic V matrix of shape `(S, 64)` using the same
    deterministic pattern as Q/K
  - export `v_full.txt` — the complete `(S, 64)` V matrix for the sequence.
    Note: `v_tile.txt` naming is only meaningful for single-chunk (S ≤ 64);
    for multi-chunk sequences the host reads per-chunk slices from `v_full.txt`
  - export `attn_out.txt` as the new expected output — shape `(S, 64)`
- [ ] Verify: `attn_out` from tiled function matches brute-force `softmax @ V`
      computed directly in Python at S = 8, 64, 128, 256, 512

---

## Track B Step 2 — Implement the V Weighted Sum HLS Kernel

**Effort:** 2–3 hours  
**Requires Vitis re-synthesis:** Yes (Linux only)  
**Can be code-edited and locally bench-tested on Windows:** Yes

### What to do

#### HLS kernel
- [ ] Create `hls/v_weighted_sum/v_weighted_sum_core_hls.hpp`
- [ ] Create `hls/v_weighted_sum/v_weighted_sum_core_hls.cpp`:
  - inputs: `weights_tile[8][64]` (float32), `v_tile[64][64]` (float32)
  - output: `out_tile[8][64]` (float32) — one **partial** contribution to
    `attn_out`; the kernel computes `out[row][d] = sum_col(weights[row][col]
    * v[col][d])` for the 64 columns in this K-chunk only. The host
    accumulates across all K-chunks to produce the final `attn_out` row.
  - add `#pragma HLS PIPELINE II=1` and `#pragma HLS ARRAY_PARTITION` on V tile
- [ ] Add AXI interface pragmas (separate bundles for weights, V, output)

#### Testbench
- [ ] Create `hls/v_weighted_sum/tb_v_weighted_sum.cpp`
- [ ] This testbench verifies a **single K-chunk only** (the kernel computes
      one partial contribution, not the full `attn_out`). Test as follows:
  - Use the existing `score_softmax.txt` (8×64 softmax weights, one K-chunk)
    and a synthetic `v_tile.txt` (64×64 V values)
  - Compute expected partial output in Python:
    `partial = softmax_weights (8×64) @ v_tile (64×64)` → shape `(8, 64)`
  - Write expected partial output to `v_partial_expected.txt`
  - Confirm kernel output matches `v_partial_expected.txt`
- [ ] Note: `attn_out.txt` (the final accumulated output) is only correct after
      the host sums all K-chunks — it cannot be used as the kernel-level
      expected output for multi-chunk sequences

#### Vitis
- [ ] Create `hls/v_weighted_sum/run_hls.tcl`
- [ ] Run csim — confirm pass
- [ ] Run csynth — record MHz and DSP count

---

## Track B Step 3 — Update Host App for V Kernel

**Effort:** 2–3 hours  
**Requires:** XRT on Linux

### What to do

- [ ] Add `v_kernel` XRT object to `host/attention_score_chain_xrt.cpp`
- [ ] Allocate HBM buffer for V input and `attn_out` output
- [ ] Update the tiling loop to three passes. Do NOT run softmax inside the
      inner K-chunk loop — it must see all S keys for a query row first:
  ```
  // pass 1 — score + mask + scale for all (q, k) tile pairs → logits
  for q_chunk in [0 .. ceil(S/8)):
      for k_chunk in [0 .. ceil(S/64)):
          run score kernel → mask_scale kernel
          write scaled logit tile → logits[q_chunk*8.., k_chunk*64..]

  // pass 2 — full-row softmax (uses full-row softmax kernel from Track A Step 3C)
  for q_chunk in [0 .. ceil(S/8)):
      DMA logits[q_chunk*8.., 0..S] to device   // full S-wide row
      run full_row_softmax kernel with key_col_count = S
      DMA result back → softmax_weights[q_chunk*8..]

  // pass 3 — V weighted sum accumulation
  for q_chunk in [0 .. ceil(S/8)):
      accumulator[8][64] = 0
      for k_chunk in [0 .. ceil(S/64)):
          run v_kernel(softmax_weights[q_chunk, k_chunk*64..], V[k_chunk])
          accumulator += kernel output
      write accumulator → attn_out[q_chunk*8 ..]
  ```
- [ ] Update `vpp_link.cfg` to include `v_weighted_sum_u55c_kernel`
- [ ] Load `v_full.txt` and slice into per-K-chunk buffers inside the tiling
      loop; compare final accumulated output against `attn_out.txt`
- [ ] Test in hw_emu at S = 8, 64, 128
- [ ] Update pass signal check to include `attn_out` verification

### Expected result
- FPGA now produces the complete attention output `(S, 64)` for one head
- The output matches the Python tiling reference within float tolerance (~1e-4)

---

## Track B Extension — Real V Vectors (Track B + C combined)

Once Track C Step 2 (PyTorch Q/K hook) is working:

- [ ] Extend the hook to also capture `V` tensors from the same attention layer
- [ ] Add V to `model/export_real_vectors.py`
- [ ] Verify: FPGA `attn_out` matches PyTorch's own attention output
      (`torch.nn.functional.scaled_dot_product_attention`) for the same head
      within quantization tolerance (~1e-3)

---

## Track B Priority Order Summary

| Step | Effort | Testable on Windows | Prerequisite |
|------|--------|---------------------|--------------|
| B1 — Python reference | 1–2 hrs | Yes | Track A Step 3A |
| B2 — V kernel HLS | 2–3 hrs | Local bench yes, csynth needs Vitis | B1 |
| B3 — Host app update | 2–3 hrs | No (needs XRT) | Track A Step 3 + B2 |
| B extension — Real V | 1 hr | Yes (Python), No (FPGA) | Track C Step 2 + B3 |

---

# Track C — Real TinyLlama Inputs *(extension, after Track A + B on hardware)*

**Goal:** Replace the synthetic deterministic Q/K/V vectors with real Q, K,
and V tensors extracted from an actual TinyLlama forward pass on real text
input. The FPGA hardware and host XRT pipeline do not change — only the data
source changes. V must be included once Track B (softmax @ V) is working.

**Prerequisite:** Track A Step 3 (tiling loop) must be working first.
Track B (V kernel) must be working before real V is useful.

---

## Track C Step 1 — Set Up TinyLlama Inference in Python

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python + PyTorch  
**Testable on Windows:** Yes

### What to do

- [x] Install dependencies:
  ```
  pip install torch transformers sentencepiece
  ```
- [x] Download TinyLlama-1.1B weights (HuggingFace):
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  ```
- [x] Confirm a basic forward pass runs without error on a short sentence
- [x] Note: model weights are ~2.2 GB — ensure enough disk space

---

## Track C Step 2 — Hook Into the Attention Layer and Extract Q, K, and V

**Effort:** 2–3 hours  
**Runs on:** Any machine with PyTorch  
**Testable on Windows:** Yes

### What to do

- [x] Register a forward hook on one transformer layer's attention module to
      intercept Q and K after RoPE rotation, and V after projection (V is not
      RoPE-rotated — it is the raw projected value tensor):
  ```python
  def hook_fn(module, input, output):
      # capture q_rot, k_rot, and v here
      pass
  model.model.layers[0].self_attn.register_forward_hook(hook_fn)
  ```
- [x] Run a forward pass on a test sentence, e.g. `"The cat sat on the mat"`
- [x] Confirm captured Q shape is `(1, 32, S, 64)` — batch, heads, seq, head_dim
- [x] Confirm captured K shape is `(1, 4, S, 64)` — TinyLlama uses 4 KV heads
      (grouped query attention), so K is shared across groups of 8 Q heads
- [x] Confirm captured V shape is `(1, 4, S, 64)` — same layout as K
- [x] Extract one head's Q, K, and V:
  ```python
  q_head = q_rot[0, 0, :, :]   # shape (S, 64), float32
  k_head = k_rot[0, 0, :, :]   # shape (S, 64), float32
  v_head = v[0, 0, :, :]       # shape (S, 64), float32
  ```
- [x] Verify against PyTorch's own attention output:
  ```python
  score_ref  = (q_head @ k_head.T) / 8.0
  weights    = torch.softmax(score_ref, dim=-1)
  attn_ref   = weights @ v_head              # shape (S, 64)
  ```

---

## Track C Step 3 — Quantize Q and K to INT8

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [x] Apply per-tensor symmetric INT8 quantization to Q and K:
  ```python
  q_scale = q_head.abs().max() / 127.0
  k_scale = k_head.abs().max() / 127.0
  q_int8  = (q_head / q_scale).round().clamp(-128, 127).to(torch.int8)
  k_int8  = (k_head / k_scale).round().clamp(-128, 127).to(torch.int8)
  ```
- [x] Verify the quantization error is small:
  ```python
  q_reconstructed = q_int8.float() * q_scale
  print((q_reconstructed - q_head).abs().max())   # should be < 0.02
  ```
- [x] Confirm total_scale factor is correct:
  ```python
  total_scale  = float(q_scale) * float(k_scale) * (1.0 / 8.0)
  score_dequant = (q_int8.float() @ k_int8.float().T) * total_scale
  # compare to score_ref — should match within quantization error
  ```

---

## Track C Step 4 — Export Real Vectors to Files

**Effort:** 1 hour  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [ ] Create `model/export_real_vectors.py` that:
  - takes `--text`, `--layer`, and `--head` arguments
  - runs TinyLlama forward pass and hooks Q, K, and V as in Track C Step 2
  - quantizes Q/K as in Track C Step 3 (V stays float32)
  - writes `q_tile.txt`, `k_tile.txt`, `v_tile.txt`, `kernel_meta.txt` and all
    expected intermediate outputs in the same format as the synthetic exporter
  - writes `attn_out.txt` once Track B (V kernel) is complete by running the
    Python tiling reference including the V weighted sum pass
- [ ] Run on a test sentence and confirm all output files are generated
- [ ] Diff file format against the synthetic exporter — host app should read
      them without any changes

---

## Track C Step 5 — Run Real Vectors Through the FPGA Pipeline

**Effort:** 1–2 hours (assuming Track A Step 3 is already working)  
**Requires:** XRT on Linux, hw_emu or real hw

### What to do

- [ ] Copy the real vector files to the Linux machine
- [ ] Pass the real vector directory to the host app:
  ```bash
  ./build/host_attention_score_chain \
    --xclbin build/attention_score_chain.xclbin \
    --vectors /path/to/real_vectors \
    --seq-len <S> \
    --device 0
  ```
- [ ] Confirm `XRT chain verification PASSED`
- [ ] If Track B is complete: compare FPGA `attn_out` against PyTorch's full
      attention output (`torch.nn.functional.scaled_dot_product_attention`)
      for the same head — should match within ~1e-3 (quantization error expected)
- [ ] If Track B is not yet complete: compare FPGA softmax weights output
      against PyTorch softmax only, and note the limitation explicitly
- [ ] Test on multiple sentences at different lengths

---

## Track C Priority Order Summary

| Step | Effort | Testable on Windows | Prerequisite |
|------|--------|---------------------|--------------|
| C1 — TinyLlama setup | 1–2 hrs | Yes | None |
| C2 — Hook Q/K/V extraction | 2–3 hrs | Yes | C1 |
| C3 — INT8 quantization | 1–2 hrs | Yes | C2 |
| C4 — Export real vectors | 1 hr | Yes | C3 |
| C5 — Run on FPGA | 1–2 hrs | No (needs XRT) | Track A Step 3 + C4 |

Track C Steps 1–4 can all be done on Windows as pure Python work while
Track A hardware bring-up is happening in parallel on the Linux machine.

---

# Track D — CPU/GPU Baseline and Performance Comparison

**Goal:** Produce the "how much faster?" numbers the course will require.
Without a baseline there is no acceleration claim — just a working demo.
CPU and GPU baselines (Steps D1 and D2) can be done entirely on Windows.
Step D3 (FPGA timing) requires Linux + XRT + hardware.

**Scope:** Baselines should cover the same computation the FPGA handles.
Once Track B is complete that means score + mask + scale + softmax + V weighted
sum, and the comparison metric should be `attn_out` latency, not just softmax.
Until Track B is done, compare score + softmax only and note the limitation.

**Prerequisite:** Track A Step 3A (Python tiling) must be done so the baseline
runs the same computation as the FPGA.

---

## Track D Step 1 — CPU Baseline in Python

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [ ] Write `model/benchmark_cpu.py` that:
  - generates synthetic Q, K, and V at S = 8, 64, 128, 256, 512
  - runs the full two-pass tiled computation (Track A Step 3A + Track B Step 1):
    pass 1 = score + mask + scale for all tiles, pass 2 = softmax across full
    rows, pass 3 = V weighted sum accumulation
  - times end-to-end (`attn_out` latency) using `time.perf_counter()` —
    10 iterations, report mean and standard deviation
  - also times score+softmax only (for comparison before Track B is done)
  - reports latency in milliseconds and throughput in tiles/second
- [ ] Also time the brute-force (un-tiled) numpy version for comparison
- [ ] Record results in a table:

  | S | Tiled CPU (ms) | Brute-force CPU (ms) |
  |---|---|---|
  | 8 | | |
  | 64 | | |
  | 128 | | |
  | 256 | | |
  | 512 | | |

---

## Track D Step 2 — GPU Baseline in PyTorch *(if GPU available)*

**Effort:** 1–2 hours  
**Runs on:** Any machine with a CUDA GPU + PyTorch  
**Testable on Windows:** Yes (if GPU present)

### What to do

- [ ] Write `model/benchmark_gpu.py` that:
  - generates the same synthetic Q, K, and V as Track D Step 1
  - runs the full attention block on GPU:
    `scores = Q @ K.T`, causal mask, scale, `torch.softmax`, `softmax @ V`
  - uses `torch.cuda.synchronize()` before timing to avoid async errors
  - times end-to-end `attn_out` latency with 10 warm-up + 100 timed runs
  - also times score+softmax only separately so the comparison is fair
    regardless of whether Track B is done
  - reports mean latency and throughput
- [ ] Compare against the CPU baseline from Step 1

---

## Track D Step 3 — FPGA Timing Measurement

**Effort:** 1–2 hours  
**Requires:** XRT on Linux, hw_emu or real hw  
**Prerequisite:** Track A Step 3 (tiling loop on FPGA) working

### What to do

- [ ] Add wall-clock timing to `host/attention_score_chain_xrt.cpp`:
  - total time: first DMA-to-device → last DMA-from-device
  - compute-only time: kernel launches only, excluding DMA
  - use `std::chrono::high_resolution_clock`
- [ ] Run at S = 8, 64, 128, 256, 512 and record both times
- [ ] Record results:

  | S | FPGA total (ms) | FPGA compute only (ms) | DMA (ms) |
  |---|---|---|---|
  | 8 | | | |
  | 64 | | | |
  | 128 | | | |
  | 256 | | | |
  | 512 | | | |

---

## Track D Step 4 — Comparison and Analysis

**Effort:** 1–2 hours  
**Runs on:** Any machine

### What to do

- [ ] Combine results into one comparison table:

  | S | CPU (ms) | GPU (ms) | FPGA total (ms) | FPGA compute (ms) | Speedup vs CPU |
  |---|---|---|---|---|---|
  | 8 | | | | | |
  | 64 | | | | | |
  | 128 | | | | | |
  | 256 | | | | | |
  | 512 | | | | | |

- [ ] Plot latency vs sequence length on a log-log scale — confirm S² growth
- [ ] Identify whether FPGA time is DMA-dominated or compute-dominated:
  - DMA-dominated → double buffering (Track A Step 4) is the right next step
  - compute-dominated → more unrolling or dataflow merge (Track A Steps 1/5)
- [ ] Note honest limitations:
  - FPGA processes one head sequentially; GPU batches all 32 heads in parallel
  - comparison is for the isolated pipeline only, not full attention

---

## Track D Priority Order Summary

| Step | Effort | Testable on Windows | Prerequisite |
|------|--------|---------------------|--------------|
| D1 — CPU baseline | 1–2 hrs | Yes | Track A Step 3A |
| D2 — GPU baseline | 1–2 hrs | Yes (needs CUDA GPU) | None |
| D3 — FPGA timing | 1–2 hrs | No (needs XRT) | Track A Step 3 |
| D4 — Comparison | 1–2 hrs | Yes | D1 + D3 |

Steps D1 and D2 can be done immediately on Windows. D3 is blocked on hardware
but is a straightforward instrumentation change once Track A Step 3 is running.
