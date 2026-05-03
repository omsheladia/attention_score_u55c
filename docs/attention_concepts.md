# Attention Score — Concepts, Terminology, and Project Scope

This document explains the attention mechanism, all relevant terminology, and how
this project fits into TinyLlama inference. It is written for someone new to
transformer internals.

---

## 1. From Text to Tokens to Vectors

When TinyLlama reads a sentence like `"The cat sat on the mat"`, it first splits
the text into **tokens** — roughly one token per word or word-piece. Each token
is then converted into a vector of numbers called an **embedding**.

In TinyLlama, every token becomes a vector of **2048 numbers**. That vector
captures the "meaning" of that token in context. For a 6-token sentence:

```
input = shape (6 tokens, 2048 numbers per token)
```

The sequence length `S` is simply the number of tokens in the input. All the
matrix sizes below scale with `S`.

---

## 2. What are Q, K, and V?

Inside each of TinyLlama's 22 transformer layers, the model asks:
*"which tokens should pay attention to which other tokens?"*

To answer this, each token's 2048-number vector is projected into three smaller
vectors using learned weight matrices:

- **Q (Query)** — "what am I looking for?"
- **K (Key)**   — "what do I offer?"
- **V (Value)** — "what do I actually contain?"

After projection, for one attention head:

```
Q = shape (S, 64)   — one 64-dim vector per token
K = shape (S, 64)   — one 64-dim vector per token
V = shape (S, 64)   — one 64-dim vector per token
```

The 64 here is the **head dimension** (explained in section 4). It is fixed
regardless of sequence length. Only the number of rows (tokens) changes.

---

## 3. What is the Attention Score?

The attention score for one head is:

```
score = Q @ K^T
      = (S, 64) @ (64, S)
      = (S, S)
```

Entry `score[i, j]` answers: *"how much should token i attend to token j?"*
It is computed as a dot product of two 64-number vectors — one multiplication
and one addition per dimension, 64 times.

After the raw score is computed, three more operations follow:

1. **Causal mask** — set all positions where `j > i` to −∞ so a token cannot
   attend to future tokens it hasn't seen yet
2. **Scale** — divide by `sqrt(head_dim) = sqrt(64) = 8` to prevent very large
   values from making the softmax numerically unstable
3. **Softmax** — convert each row into probabilities that sum to 1

The softmax output is then multiplied by V to produce the final attended output:

```
attn_out = softmax(score) @ V
```

The current staged FPGA design implements this final `softmax @ V` step for one
attention head with a V weighted-sum kernel. The full TinyLlama decoder around
that one-head block remains out of scope.

---

## 4. What are Attention Heads, and Why 32?

A single attention computation can only capture one type of relationship between
tokens at a time. But language has many simultaneous relationship types:

- **"it"** refers back to **"animal"** — coreference
- **"tired"** describes **"animal"** — semantic agreement
- **"cross"** relates to **"street"** — action/object binding

To capture multiple relationship types in parallel, the model runs attention
**32 times simultaneously**, each with its own Q, K, V projection weights.
Each parallel run is one **attention head**. Each head learns independently
what to look for — specialization into syntax, coreference, position, etc.
emerges from training; it is not hand-designed.

The number 32 comes from dividing the hidden dimension by the head dimension:

```
num_heads = hidden_dim / head_dim
          = 2048 / 32
          = 64    ← wait, that's head_dim

actually:
num_heads = 32
head_dim  = hidden_dim / num_heads = 2048 / 32 = 64
```

More heads → each head sees a smaller slice → more relationship diversity.
Fewer heads → each head sees more → less diversity. 32 is the standard choice
at the ~1B parameter scale for this hidden dim.

---

## 5. What is Head Dimension, and Why 64?

The **head dimension** (head_dim = 64) is the length of each Q, K, V vector
after projection. It is fixed for the entire model.

It is chosen as:

```
head_dim = hidden_dim / num_heads = 2048 / 32 = 64
```

64 is also a convenient hardware number:
- `sqrt(64) = 8` exactly — the attention scale is a clean divide-by-8
- 64 fits neatly into SIMD registers, DSP arrays, and SRAM banks
- 64 INT8 multiplications unroll cleanly with a factor-of-8 unroll pragma

The head_dim axis is the one that gets **fully computed per tile** — it never
needs tiling. Only the sequence-length axis needs tiling.

---

## 6. Why INT8?

Q and K after projection are float32. Before sending to the FPGA, they are
**quantized** to INT8 — each 32-bit float is compressed to an 8-bit integer,
with a per-tensor scale factor saved separately to recover the original magnitude.

Benefits:
- 4× memory reduction
- FPGA DSP blocks do INT8 multiplication natively and efficiently
- Integer accumulation is deterministic and easy to verify

The dequantization (recovering float values) happens in the `score_scale`
kernel, which multiplies by `q_scale * k_scale * (1 / sqrt(64))`.

---

## 7. Real Sizes at Different Sequence Lengths

For **one head**:

| Seq len | Q shape   | K shape   | Score matrix | Score matrix size |
|---------|-----------|-----------|--------------|-------------------|
| 8       | (8, 64)   | (8, 64)   | (8, 8)       | 256 B (INT32)     |
| 64      | (64, 64)  | (64, 64)  | (64, 64)     | 16 KB             |
| 128     | (128, 64) | (128, 64) | (128, 128)   | 64 KB             |
| 256     | (256, 64) | (256, 64) | (256, 256)   | 256 KB            |
| 512     | (512, 64) | (512, 64) | (512, 512)   | 1 MB              |
| 2048    | (2048,64) | (2048,64) | (2048, 2048) | 16 MB             |

These are per head. TinyLlama has 32 Q heads and 4 KV heads (grouped query
attention), and 22 layers.

---

## 8. What is a Tile?

The FPGA has limited on-chip SRAM. For seq_len=512, the full score matrix is
`512×512×4 bytes = 1 MB`, which does not fit in on-chip buffers. So the
computation is split into **tiles** — small rectangular sub-blocks that do fit.

The hardware is sized for:

```
Q tile:     8 rows  × 64 cols   (8 query tokens,  each 64-dim)  = 512 INT8
K tile:    64 rows  × 64 cols   (64 key tokens,   each 64-dim)  = 4096 INT8
Score tile: 8 rows  × 64 cols   (output block)                  = 512 INT32
```

One tile call computes the dot products for 8 query tokens against 64 key
tokens, producing an 8×64 block of the full score matrix.

To cover the full S×S score matrix, tiles are iterated in a nested loop:

```
for each Q-chunk  (8 rows at a time):        # ceil(S / 8)  iterations
    for each K-chunk  (64 cols at a time):   # ceil(S / 64) iterations
        run 4-stage kernel chain on this tile
        write 8×64 result block into score_matrix
```

Visual for seq_len = 16 (score matrix = 16×16):

```
         K-chunk 0      K-chunk 0
         cols [0..63]   (only cols [0..15] active)

Q-chunk 0  ┌────────┐
rows [0..7]│ tile 0 │
           └────────┘
Q-chunk 1  ┌────────┐
rows [8..15]│ tile 1 │
            └────────┘
```

---

## 9. Number of Tile Calls in TinyLlama

For one head at sequence length S:

```
Q-chunks = ceil(S / 8)
K-chunks = ceil(S / 64)
tiles per head = Q-chunks × K-chunks
```

| Seq len | Q-chunks | K-chunks | Tiles/head | Tiles/layer (×32) | Full model (×22 layers) |
|---------|----------|----------|------------|-------------------|-------------------------|
| 8       | 1        | 1        | 1          | 32                | 704                     |
| 64      | 8        | 1        | 8          | 256               | 5,632                   |
| 128     | 16       | 2        | 32         | 1,024             | 22,528                  |
| 256     | 32       | 4        | 128        | 4,096             | 90,112                  |
| 512     | 64       | 8        | 512        | 16,384            | 360,448                 |
| 2048    | 256      | 32       | 8,192      | 262,144           | 5,767,168               |

This project now supports the tiled staged flow for **one attention head**. The
host tiling loop covers synthetic sequence lengths `S = 8, 64, 128, 256, 512`,
and the legacy single-tile path remains useful as a sanity check. Real
TinyLlama vector-mode runs are verified for the legacy single-tile directory
and for current full-sequence directories at `S=16` and `S=64`.

### How the tile counts are derived

The formula is:

```
tiles per head = ceil(S / 8) × ceil(S / 64)
```

You multiply because for **every** Q-chunk you must pair it with **every**
K-chunk to fill the full S×S score matrix.

**S = 8**
```
ceil(8/8)  = 1 Q-chunk
ceil(8/64) = 1 K-chunk   (8 < 64, fits in one K-chunk)
1 × 1 = 1 tile
```

**S = 64**
```
ceil(64/8)  = 8 Q-chunks
ceil(64/64) = 1 K-chunk  (exactly fills one K-chunk)
8 × 1 = 8 tiles
```

**S = 128**
```
ceil(128/8)  = 16 Q-chunks
ceil(128/64) =  2 K-chunks
16 × 2 = 32 tiles
```

**S = 256**
```
ceil(256/8)  = 32 Q-chunks
ceil(256/64) =  4 K-chunks
32 × 4 = 128 tiles
```

**S = 512**
```
ceil(512/8)  = 64 Q-chunks
ceil(512/64) =  8 K-chunks
64 × 8 = 512 tiles
```

Notice the tile count grows as **S²** because the score matrix is S×S.
Double the sequence length → quadruple the tiles. This is why long-context
inference is expensive.

---

## 10. Where This Project Fits in the Full Inference Pipeline

```
CPU / host side                         FPGA (this project)
────────────────────────────────────    ────────────────────────────────────
Tokenize input text
Token embeddings  (S, 2048)
Q/K/V projection  (S, 2048) → (S, 64)
RoPE embedding
Quantize to INT8
                            ──────────▶  Q_rot_int8  (S, 64)  ┐
                                         K_rot_int8  (S, 64)  ┘ per head
                                           │
                                           ▼  score GEMM (tiled)
                                         score_raw_int32  (S, S)
                                           │
                                           ▼  causal mask
                                         score_masked_int32  (S, S)
                                           │
                                           ▼  scale (dequantize)
                                         score_scaled_fp32  (S, S)
                                           │
                                           ▼  row-wise softmax
                                         score_softmax_fp32  (S, S)
                                           │
                            ◀──────────────┘
softmax @ V  (S, 64)   staged FPGA V weighted-sum kernel
Output projection
Feed-forward layer
...
```

---

## 11. Recommended Test Sequence Lengths

| Seq len | Tokens | Why |
|---------|--------|-----|
| **8**   | ~1-2 words | Single tile, no tiling loop exercised — pure sanity check |
| **64**  | ~a short phrase | Exactly fills one K-chunk; first real Q-tiling case |
| **128** | ~1-2 sentences | Small real prompt; 16×2 = 32 tiles per head |
| **256** | ~a short paragraph | Medium prompt; 32×4 = 128 tiles per head |
| **512** | ~half a page | Good stress test; 64×8 = 512 tiles per head |

2048 (TinyLlama max context) is skipped for `hw_emu` — it would require
8,192 tile calls per head and make emulation very slow. It can be tested once
real hardware is available.

---

## 12. What Has Been Built, and What Remains

The current repo has the correctness-first tiled path built and verified for one
attention head:

1. **Python reference** (`model/attention_score_ref.py`) supports full-sequence
   tiled score computation, full-row softmax, and `softmax @ V`. The
   `--check-full-tiling` check covers `S = 8, 64, 128, 256, 512`.

2. **XRT host app** (`host/attention_score_chain_xrt.cpp`) supports synthetic
   `--seq-len` mode and saved-vector `--vectors` mode. It launches the staged
   score, mask/scale, full-row softmax, and V weighted-sum kernels.

3. **HLS kernels** include the score GEMM, merged mask/scale, full-row softmax
   for `S <= 512`, and V weighted-sum stages.

4. **Real TinyLlama vectors** can be exported from PyTorch Q/K/V hooks. The
   legacy single-tile case and full-sequence `S=16` and `S=64` directories have
   been run through the real U55C flow.

What remains is performance and integration work rather than basic correctness:

- reduce staged HBM round trips and repeated kernel launch overhead
- run larger real-vector sweeps such as `S=128`, `S=256`, and `S=512`
- connect the one-head block into a larger TinyLlama attention subgraph
- eventually handle full decoder-layer integration and KV-cache-aware decode
