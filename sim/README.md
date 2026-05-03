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

For full-sequence real TinyLlama tiled-vector cases, run:

```bash
python3 model/export_real_vectors.py --seq-len 16 --output-dir sim/real_tinyllama_s16 --text "<prompt with at least 16 tokens>"
python3 model/export_real_vectors.py --seq-len 64 --output-dir sim/real_tinyllama_s64 --text "<prompt with at least 64 tokens>"
```

Verified generated directories:

- `sim/real_tinyllama_s16/`
- `sim/real_tinyllama_s64/`

The HLS C-sim testbench consumes:

- `q_tile.txt`
- `k_tile.txt`
- `score_raw.txt`
- `score_masked.txt`
- `score_scaled.txt`
- `score_softmax.txt`
- `kernel_meta.txt`
- `metadata.json`

The current XRT host verifies `score_raw`, `score_scaled`, `score_softmax`, and
`attn_out` when `v_full.txt` plus `attn_out.txt` or `attn_ref_float.txt` are
present. `score_masked` remains useful for the legacy standalone causal-mask
testbench.

The real TinyLlama directories keep the same current-host input/output file
names, plus `q_float.txt`, `k_float.txt`, `v_full.txt`, and
`attn_ref_float.txt` for inspection. Full-sequence directories also include
`q_full.txt`, `k_full.txt`, and quantized-pipeline `attn_out.txt`; the XRT host
uses those files to enter tiled `--vectors` mode.
