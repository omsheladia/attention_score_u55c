# Simulation Artifacts

Run:

```bash
python3 model/export_attention_score_vectors.py --output-dir sim/attention_score_tile
```

That generates deterministic vectors under:

- `sim/attention_score_tile/`

For a real TinyLlama-derived single-tile case, run:

```bash
python3 model/export_real_vectors.py --local-files-only
```

That generates vectors under:

- `sim/real_tinyllama_tile/`

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

The real TinyLlama directory keeps the same current-host input/output file
names, plus `q_float.txt`, `k_float.txt`, `v_full.txt`, and
`attn_ref_float.txt` for inspection and later Track B work. The current
single-tile real exporter requires `S <= 8`.
