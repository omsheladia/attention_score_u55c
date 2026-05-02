# RTL Notes

This folder is intentionally lightweight for now.

The first implementation target for the isolated score engine is HLS because the
offload boundary is a dense matrix-dot kernel with a clean memory interface.
Once the kernel contract is stable, the same tile interface can be lowered into
handwritten RTL if needed:

- `q_tile[8][64]` as `int8`
- `k_tile[64][64]` as `int8`
- `score_raw[8][64]` as `int32`

If you later want the same workspace extended with SystemVerilog, this is the
natural landing zone.
