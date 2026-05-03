# Softmax HLS Flow

This kernel is the stage after scaling:

`score_scaled_fp32 -> probability_fp32`

It performs row-wise softmax over the active key columns for one fixed `8 x 64`
tile.

This is correct for the legacy single-tile verification and for tile-local
softmax checks. For `S > 64`, softmax must normalize across every key in the
full row, so the current tiled XRT path uses `hls/softmax_full_row/` instead.
