# Model Utilities

These Python files isolate the attention-score block from the larger TinyLlama
reference.

- `attention_score_ref.py`
  - small reference implementation for one score tile
  - includes raw score, causal mask, score scaling, and softmax helpers
- `export_attention_score_vectors.py`
  - emits deterministic vectors under `attention_score_u55c/sim/attention_score_tile/`

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
