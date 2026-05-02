# Implementation Checklist

This checklist covers two tracks running in parallel:

- **Track A — Speedup:** make the existing hardware pipeline faster and capable
  of real sequence lengths using synthetic inputs
- **Track B — Real inputs (Option 2):** replace synthetic Q/K vectors with real
  TinyLlama inference data

**Priority:** Complete Track A Steps 1–3 first. Track B is an extension once
Track A is working on hardware.

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
- [ ] Update kernel launch sequence:
      `score → mask_scale → softmax`
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
- [ ] Add a new function `compute_full_attention_score(q_full, k_full)` that:
  - accepts `q_full` of shape `(S, 64)` and `k_full` of shape `(S, 64)`
  - loops over Q-chunks `(ceil(S/8))` and K-chunks `(ceil(S/64))`
  - calls `compute_attention_score_tile` for each `(q_chunk, k_chunk)` pair
  - sets `query_pos_base = q_chunk * 8`, `key_pos_base = k_chunk * 64`
  - calls `apply_causal_mask`, `scale_scores`, `softmax_rows` per tile
  - assembles and returns the full `(S, S)` score matrix
- [ ] Add a test script or CLI flag that runs the full tiling at:
  - S = 8   (single tile sanity check)
  - S = 64  (8 Q-chunks × 1 K-chunk)
  - S = 128 (16 × 2)
  - S = 256 (32 × 4)
  - S = 512 (64 × 8)
- [ ] For each S, verify the tiled output matches a brute-force reference
      (compute the full score matrix directly in Python and compare)

### Part B — XRT host tiling loop

- [ ] Open `host/attention_score_chain_xrt.cpp`
- [ ] Add `seq_len` as a command-line argument (`--seq-len`)
- [ ] Replace the single-tile kernel launch with a nested loop:
  ```
  for q_chunk in [0 .. ceil(S/8)):
      for k_chunk in [0 .. ceil(S/64)):
          slice q_rows and k_rows from full Q/K buffers
          set query_pos_base = q_chunk * 8
          set key_pos_base   = k_chunk * 64
          set query_row_count = min(8,  S - q_chunk*8)
          set key_col_count   = min(64, S - k_chunk*64)
          DMA q tile and k tile to device
          launch kernel chain
          wait
          DMA softmax tile back
          write into output[q_chunk*8 .. , k_chunk*64 ..]
  ```
- [ ] Allocate output buffer large enough for the full `S × S` score matrix
- [ ] Update reference loading to generate full-sequence reference vectors
      (use the Python tiling function from Part A to produce expected outputs)
- [ ] Test in hw_emu at S = 8, 64, 128

### Expected result
- Project can now process real sentence-length inputs
- Performance scales predictably: tile count = `ceil(S/8) × ceil(S/64)`

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

## Step 5 — Merge All 4 Stages Into One Dataflow Kernel *(future milestone)*

**Effort:** 2–3 days minimum  
**Risk:** High — softmax multi-pass structure conflicts with HLS DATAFLOW rules  
**Do not attempt within a 2-day sprint**

### What needs to happen

- [ ] Restructure softmax to be single-pass compatible, or accept it runs
      sequentially inside the merged kernel without DATAFLOW
- [ ] Create `hls/attention_score_chain/attention_score_chain_hls.cpp`
      with all 4 stages (or 3 after Step 2) as sub-functions
- [ ] Add `#pragma HLS DATAFLOW` and verify Vitis does not reject it
- [ ] Update `vpp_link.cfg` to a single kernel
- [ ] Rebuild xclbin, re-run all sequence length tests
- [ ] Confirm timing closure (softmax timing warning may worsen)

### Expected result
- Eliminates all intermediate HBM traffic between stages
- All stages run as a pipeline — stage N+1 starts while stage N finishes
- Biggest single-kernel latency improvement available

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

# Track B — Real TinyLlama Inputs *(extension, after Track A works on hardware)*

**Goal:** Replace the synthetic deterministic Q/K vectors with real Q and K
extracted from an actual TinyLlama forward pass on real text input. The FPGA
hardware and host XRT pipeline do not change — only the data source changes.

**Prerequisite:** Track A Step 3 (tiling loop) must be working first.

---

## Track B Step 1 — Set Up TinyLlama Inference in Python

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python + PyTorch  
**Testable on Windows:** Yes

### What to do

- [ ] Install dependencies:
  ```
  pip install torch transformers sentencepiece
  ```
- [ ] Download TinyLlama-1.1B weights (HuggingFace):
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  ```
- [ ] Confirm a basic forward pass runs without error on a short sentence
- [ ] Note: model weights are ~2.2 GB — ensure enough disk space

---

## Track B Step 2 — Hook Into the Attention Layer and Extract Q, K

**Effort:** 2–3 hours  
**Runs on:** Any machine with PyTorch  
**Testable on Windows:** Yes

### What to do

- [ ] Register a forward hook on one transformer layer's attention module to
      intercept the Q and K tensors after RoPE is applied but before the score
      computation:
  ```python
  def hook_fn(module, input, output):
      # capture q_rot and k_rot here
      pass
  model.model.layers[0].self_attn.register_forward_hook(hook_fn)
  ```
- [ ] Run a forward pass on a test sentence, e.g. `"The cat sat on the mat"`
- [ ] Confirm captured Q shape is `(1, 32, S, 64)` — batch, heads, seq, head_dim
- [ ] Confirm captured K shape is `(1, 4, S, 64)` — TinyLlama uses 4 KV heads
      (grouped query attention), so K is shared across groups of 8 Q heads
- [ ] Extract one head's Q and K:
  ```python
  q_head = q_rot[0, 0, :, :]   # shape (S, 64), float32
  k_head = k_rot[0, 0, :, :]   # shape (S, 64), float32
  ```
- [ ] Verify against PyTorch's own attention score:
  ```python
  score_ref = (q_head @ k_head.T) / 8.0   # divide by sqrt(64)
  ```

---

## Track B Step 3 — Quantize Q and K to INT8

**Effort:** 1–2 hours  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [ ] Apply per-tensor symmetric INT8 quantization to Q and K:
  ```python
  q_scale  = q_head.abs().max() / 127.0
  k_scale  = k_head.abs().max() / 127.0
  q_int8   = (q_head / q_scale).round().clamp(-128, 127).to(torch.int8)
  k_int8   = (k_head / k_scale).round().clamp(-128, 127).to(torch.int8)
  ```
- [ ] Verify the quantization error is small:
  ```python
  q_reconstructed = q_int8.float() * q_scale
  print((q_reconstructed - q_head).abs().max())   # should be < 0.02
  ```
- [ ] Compute the expected score using dequantized values to confirm the
      total_scale factor is correct:
  ```python
  total_scale = float(q_scale) * float(k_scale) * (1.0 / 8.0)
  score_dequant = (q_int8.float() @ k_int8.float().T) * total_scale
  # compare to score_ref — should match within quantization error
  ```

---

## Track B Step 4 — Export Real Vectors to Files

**Effort:** 1 hour  
**Runs on:** Any machine with Python  
**Testable on Windows:** Yes

### What to do

- [ ] Create `model/export_real_vectors.py` that:
  - takes a `--text` argument for the input sentence
  - takes a `--layer` and `--head` argument to select which layer/head to export
  - runs the TinyLlama forward pass and hooks Q/K as in Track B Step 2
  - quantizes Q/K as in Track B Step 3
  - writes `q_tile.txt`, `k_tile.txt`, `kernel_meta.txt`, and all expected
    output files in the same format as `export_attention_score_vectors.py`
  - writes expected outputs by running the Python tiling reference
    (Track A Step 3A) on the real Q/K
- [ ] Run it on a test sentence and confirm output files are generated
- [ ] Diff the file format against the synthetic exporter output —
      the XRT host app should be able to read them without any changes

---

## Track B Step 5 — Run Real Vectors Through the FPGA Pipeline

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
- [ ] Compare FPGA softmax output against PyTorch's native attention softmax
      output for the same head — they should match within float tolerance
- [ ] Test on multiple sentences of different lengths to confirm tiling works
      with real data

### Pass signal
The FPGA softmax output for a real sentence matches PyTorch's own attention
softmax for the same layer and head within a tolerance of ~1e-3 (quantization
error is expected).

---

## Track B Priority Order Summary

| Step | Effort | Testable on Windows | Prerequisite |
|------|--------|---------------------|--------------|
| B1 — TinyLlama setup | 1–2 hrs | Yes | None |
| B2 — Hook Q/K extraction | 2–3 hrs | Yes | B1 |
| B3 — INT8 quantization | 1–2 hrs | Yes | B2 |
| B4 — Export real vectors | 1 hr | Yes | B3 |
| B5 — Run on FPGA | 1–2 hrs | No (needs XRT) | Track A Step 3 + B4 |

Track B Steps 1–4 can all be done on Windows as pure Python work while
Track A hardware bring-up is happening in parallel on the Linux machine.
