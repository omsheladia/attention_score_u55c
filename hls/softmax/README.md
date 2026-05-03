# Softmax HLS Flow

This kernel is the stage after scaling:

`score_scaled_fp32 -> probability_fp32`

It performs row-wise softmax over the active key columns for one fixed `8 x 64`
tile.

This is correct for the current single-tile verification and for sequence
lengths with only one K chunk (`S <= 64`). For `S > 64`, softmax must normalize
across every key in the full row, so Track A calls for a new full-row softmax
kernel/design before multi-K-chunk sequence tests can be considered correct.
