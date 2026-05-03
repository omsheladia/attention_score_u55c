# Model Utilities

These Python files isolate the attention-score block from the larger TinyLlama
reference.

- `attention_score_ref.py`
  - small reference implementation for one score tile
  - includes raw score, causal mask, score scaling, and softmax helpers
  - includes Track A Step 3 full-sequence tiled score/softmax reference helpers
  - includes Track B Step 1 `softmax @ V` reference helpers
- `export_attention_score_vectors.py`
  - emits deterministic vectors under `sim/attention_score_tile/`
  - exports Track B `v_full.txt`, `v_tile.txt`, `v_partial_expected.txt`, and
    `attn_out.txt`
- `check_tinyllama_setup.py`
  - Track C Step 1 setup checker for real-vector work
  - verifies `torch`, `transformers`, and `sentencepiece`
  - loads TinyLlama and runs one short forward pass
- `extract_tinyllama_qkv.py`
  - Track C Steps 2-3 extractor and Q/K quantizer for real Q/K/V tensors
  - captures Q and K after RoPE and V after projection
  - verifies one selected head against PyTorch scaled-dot-product attention
  - quantizes selected Q/K tensors to INT8 and reports scale/error metrics
- `export_real_vectors.py`
  - Track C Step 4/5 exporter for real TinyLlama vector test cases
  - keeps the legacy single-tile export for short prompts
  - writes full-sequence tiled vector directories when `--seq-len <S>` is used
  - also writes `v_full.txt`, quantized `attn_out.txt`, and
    `attn_ref_float.txt` for Track B vector-mode `attn_out` verification
- `benchmark_common.py`
  - shared Track D helpers for synthetic input generation, real-vector loading,
    tiled CPU math, brute-force CPU math, validation, and timing
- `benchmark_cpu.py`
  - Track D Step 1 CPU baseline
  - benchmarks synthetic scaling lengths and optional real exported vectors
  - reports both current score+mask+scale+softmax scope and future +V scope
- `benchmark_gpu.py`
  - Track D Step 2 CUDA baseline
  - uses the same synthetic and real-vector inputs as the CPU baseline
  - exits cleanly when CUDA is unavailable

The current XRT host verifies the score/softmax runtime outputs against:

- `score_raw.txt`
- `score_scaled.txt`
- `score_softmax.txt`

Track B reference/export files are now also available:

- `v_full.txt`
- `v_tile.txt`
- `v_partial_expected.txt`
- `attn_out.txt`

`score_masked.txt` is still exported for the legacy standalone mask kernel and
for debugging, but the current hardware chain merges mask and scale into
`mask_scale_u55c_kernel`.

The exported vectors assume the offload boundary starts after RoPE. In other
words, the FPGA score kernel consumes `Q_rot` and `K_rot`, not the pre-RoPE
projection outputs.

## Full-Sequence Python Tiling Check

Track A Step 3 Part A is implemented in `attention_score_ref.py`. It assembles
the full scaled logit matrix with 8 x 64 score tiles, then applies full-row
softmax across all `S` keys.

Run:

```bash
python3 model/attention_score_ref.py --check-full-tiling
```

The verified local run covered `S = 8, 64, 128, 256, 512` and matched the
brute-force reference with zero max difference for raw scores, scaled logits,
and softmax probabilities; Track B `attn_out` max difference was
`2.77555756e-17`.

## Track B Softmax @ V Reference

Track B Step 1 is implemented in `attention_score_ref.py`:

- `deterministic_v_matrix(row_count)`
- `compute_v_weighted_sum_partial(weights_tile, v_tile)`
- `compute_v_weighted_sum(softmax_weights, v_full)`
- `brute_force_v_weighted_sum(softmax_weights, v_full)`
- `compute_full_attention(q_full, k_full, v_full, ...)`

Generate the default single-tile vectors plus V/attention-output references:

```bash
python3 model/export_attention_score_vectors.py --output-dir sim/attention_score_tile
```

Generate a full-sequence synthetic Track B vector directory:

```bash
python3 model/export_attention_score_vectors.py \
  --seq-len 128 \
  --output-dir /tmp/attention_score_track_b_s128
```

## TinyLlama Setup Check

Track C uses TinyLlama only as a PyTorch data source for later Q/K/V extraction.
It does not run the FPGA flow and does not offload the full transformer model.

Run:

```bash
python model/check_tinyllama_setup.py
```

The verified local run loaded `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on CPU,
processed an 8-token prompt, produced logits with shape `(1, 8, 32000)`, and
printed:

```text
TinyLlama forward pass OK
```

Useful options:

```bash
python model/check_tinyllama_setup.py --device cuda --dtype float16
python model/check_tinyllama_setup.py --model-id /path/to/local/tinyllama --local-files-only
```

## TinyLlama Q/K/V Extraction And Q/K Quantization

Run:

```bash
python model/extract_tinyllama_qkv.py --local-files-only
```

The verified local run loaded the cached TinyLlama model on CPU and captured:

```text
Q_rot: (1, 32, 8, 64)
K_rot: (1, 4, 8, 64)
V:     (1, 4, 8, 64)
```

For layer 0 / Q head 0, the script selected `q_head`, `k_head`, and `v_head`
with shape `(8, 64)`, computed a causal attention reference with output shape
`(8, 64)`, matched PyTorch scaled-dot-product attention with max difference
`3.72529030e-09`, quantized Q/K to INT8, and reported:

```text
q_scale: 4.898416623473e-02
k_scale: 1.743172481656e-02
Q reconstruction max error: 2.44865417e-02
K reconstruction max error: 8.71065259e-03
Score dequant max error: 2.65718549e-02
Score dequant mean error: 5.51600056e-03
```

Pass signals:

```text
TinyLlama Q/K/V extraction OK
TinyLlama Q/K quantization OK
```

## Real TinyLlama Vector Export

Run:

```bash
python model/export_real_vectors.py --local-files-only
```

This writes a current-design vector directory:

```text
sim/real_tinyllama_tile/
```

With no `--seq-len`, short prompts still use the legacy one-Q-tile vector path.
The verified local run used the default 8-token prompt and wrote:

```text
q_tile.txt
k_tile.txt
kernel_meta.txt
score_raw.txt
score_masked.txt
score_scaled.txt
score_softmax.txt
score_packed.txt
q_float.txt
k_float.txt
v_full.txt
attn_ref_float.txt
metadata.json
```

The current local C++ benches passed against that directory:

```text
attention_score test PASSED
mask_scale test PASSED
softmax test PASSED
```

The current XRT host has consumed this directory successfully on the real U55C:

```bash
./build/host_attention_score_chain \
  --xclbin build/attention_score_chain.xclbin \
  --vectors sim/real_tinyllama_tile \
  --device 0
```

Verified real-card result:

```text
attention_score_u55c_kernel 0.059 ms
mask_scale_u55c_kernel      0.029 ms
softmax_u55c_kernel         0.028 ms
total_chain                 0.121 ms
XRT chain verification PASSED
```

For full-sequence tiled vector mode, pass `--seq-len <S>` and a prompt that
tokenizes to at least `S` tokens. The exporter writes `q_full.txt`,
`k_full.txt`, full `S x S` references, quantized `attn_out.txt`, and
`attn_ref_float.txt` in addition to first-tile compatibility files.

Verified exports:

```bash
python model/export_real_vectors.py \
  --seq-len 16 \
  --output-dir sim/real_tinyllama_s16 \
  --text "In a small laboratory, engineers compare attention kernels across hardware targets. The experiment records tokens, latency, and numerical accuracy for each sequence length before the final report is written."

python model/export_real_vectors.py \
  --seq-len 64 \
  --output-dir sim/real_tinyllama_s64 \
  --text "In a small laboratory, engineers compare attention kernels across hardware targets. The experiment records tokens, latency, and numerical accuracy for each sequence length before the final report is written. A second paragraph adds enough context for a longer TinyLlama prompt, describing how query, key, and value tensors move through the FPGA pipeline while software baselines measure the same attention head for validation."
```

Verified real U55C vector-mode results:

| directory | S | total_chain ms | pass signal |
|---|---:|---:|---|
| `sim/real_tinyllama_s16` | 16 | 0.985 | `Attention output verification PASSED` |
| `sim/real_tinyllama_s64` | 64 | 2.510 | `Attention output verification PASSED` |

## CPU/GPU Baseline Benchmarks

Synthetic CPU baseline:

```bash
python model/benchmark_cpu.py
```

Real TinyLlama single-tile CPU baseline:

```bash
python model/benchmark_cpu.py --vectors sim/real_tinyllama_tile
```

CUDA GPU baseline, if a CUDA GPU is available:

```bash
python model/benchmark_gpu.py
python model/benchmark_gpu.py --vectors sim/real_tinyllama_tile
```

The CPU script validates the tiled path against the brute-force NumPy path
before timing. The verified local synthetic run covered `S = 8, 64, 128, 256,
512` with zero tiled-vs-brute differences for both softmax probabilities and
`softmax @ V`. The verified real-vector run matched
`sim/real_tinyllama_tile/score_softmax.txt` with max difference
`2.98023224e-08`.

After installing a CUDA-enabled PyTorch build, the verified GPU run used:

```text
torch 2.11.0+cu128
torch.version.cuda 12.8
NVIDIA GeForce RTX 3050 Laptop GPU
```

The synthetic GPU run validated `S = 8, 64, 128, 256, 512` against the CPU
reference. The real-vector GPU run matched
`sim/real_tinyllama_tile/score_softmax.txt` with max difference
`2.98023224e-08`.

These are software baselines only. Final FPGA speedup tables still require
comparison against the current Track B five-kernel FPGA runs.
