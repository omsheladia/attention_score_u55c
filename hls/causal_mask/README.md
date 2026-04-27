# Causal Mask HLS Flow

This kernel is the next stage after raw score GEMM:

`score_raw_int32 -> score_masked_int32`

It consumes one fixed `8 x 64` score tile plus tile-position metadata and
applies the standard decoder causal rule `key_pos <= query_pos`.
