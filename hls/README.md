# HLS Workspace

This HLS slice implements only the raw attention-score accumulation:

`score_raw = Q_rot_int8 @ K_rot_int8^T`

The top-level kernel uses flattened arrays so it resembles a U55C host-kernel
launch boundary rather than an internal helper function.
