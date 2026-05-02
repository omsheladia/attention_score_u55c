# Score Scale HLS Flow

This kernel is the stage after masking:

`score_masked_int32 -> score_scaled_fp32`

It applies one host-supplied `total_scale` term:

`total_scale = q_scale * k_scale * (1 / sqrt(64))`
