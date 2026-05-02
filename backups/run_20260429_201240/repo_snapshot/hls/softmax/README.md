# Softmax HLS Flow

This kernel is the stage after scaling:

`score_scaled_fp32 -> probability_fp32`

It performs row-wise softmax over the active key columns for one fixed `8 x 64`
tile.
