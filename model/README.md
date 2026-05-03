# Model Utilities

These Python files isolate the attention-score block from the larger TinyLlama
reference.

- `attention_score_ref.py`
  - small reference implementation for one score tile
  - includes raw score, causal mask, score scaling, and softmax helpers
- `export_attention_score_vectors.py`
  - emits deterministic vectors under `sim/attention_score_tile/`
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
  - Track C Step 4 exporter for a real TinyLlama single-tile test case
  - writes current-host-compatible `q_tile.txt`, `k_tile.txt`, `kernel_meta.txt`,
    and expected score/mask-scale/softmax outputs
  - also writes `v_full.txt` for later Track B `softmax @ V` work

The current XRT host verifies the three-kernel runtime outputs against:

- `score_raw.txt`
- `score_scaled.txt`
- `score_softmax.txt`

`score_masked.txt` is still exported for the legacy standalone mask kernel and
for debugging, but the current hardware chain merges mask and scale into
`mask_scale_u55c_kernel`.

The exported vectors assume the offload boundary starts after RoPE. In other
words, the FPGA score kernel consumes `Q_rot` and `K_rot`, not the pre-RoPE
projection outputs.

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

## Real TinyLlama Single-Tile Vector Export

Run:

```bash
python model/export_real_vectors.py --local-files-only
```

This writes a current-design vector directory:

```text
sim/real_tinyllama_tile/
```

The exporter requires the tokenized prompt to fit the current one-Q-tile host
path (`S <= 8`). The verified local run used the default 8-token prompt and
wrote:

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
