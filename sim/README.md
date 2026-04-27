# Simulation Artifacts

Run:

```powershell
python attention_score_u55c/model/export_attention_score_vectors.py
```

That generates deterministic vectors under:

- `attention_score_u55c/sim/attention_score_tile/`

The HLS C-sim testbench consumes:

- `q_tile.txt`
- `k_tile.txt`
- `score_raw.txt`
