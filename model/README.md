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
