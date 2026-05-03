# Simulation Artifacts

Run:

```bash
python3 attention_score_u55c/model/export_attention_score_vectors.py
```

That generates deterministic vectors under:

- `attention_score_u55c/sim/attention_score_tile/`

The HLS C-sim testbench consumes:

- `q_tile.txt`
- `k_tile.txt`
- `score_raw.txt`
- `score_masked.txt`
- `score_scaled.txt`
- `score_softmax.txt`
- `kernel_meta.txt`
- `metadata.json`

The current XRT host verifies `score_raw`, `score_scaled`, and
`score_softmax`. `score_masked` remains useful for the legacy standalone
causal-mask testbench.
