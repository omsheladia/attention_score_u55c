# Model Utilities

These Python files isolate the attention-score block from the larger TinyLlama
reference.

- `attention_score_ref.py`
  - small reference implementation for one score tile
  - includes raw score, causal mask, and optional score scaling helpers
- `export_attention_score_vectors.py`
  - emits deterministic vectors under `attention_score_u55c/sim/attention_score_tile/`

The exported vectors assume the offload boundary starts after RoPE. In other
words, the FPGA score kernel consumes `Q_rot` and `K_rot`, not the pre-RoPE
projection outputs.
