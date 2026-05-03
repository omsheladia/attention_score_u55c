# Full-Row Softmax HLS Kernel

This kernel is the Track A Step 4 correctness bridge for sequence lengths with
more than one 64-column K chunk.

The existing `hls/softmax/` kernel normalizes one fixed `8 x 64` tile. That is
correct for `S <= 64`, but for `S = 128, 256, 512` each query row must be
normalized across all keys in the row. This kernel accepts an `8 x 512` row
block and a runtime `key_col_count`.

Local C++ bench:

```bash
g++ -O2 -std=c++17 \
  hls/softmax_full_row/softmax_full_row_hls.cpp \
  hls/softmax_full_row/tb_softmax_full_row.cpp \
  -Ihls/common \
  -o sim/tb_softmax_full_row

sim/tb_softmax_full_row
```

Vitis HLS:

```bash
vitis_hls -f attention_score_u55c/hls/softmax_full_row/run_hls.tcl
```

This kernel is not yet wired into the XRT host chain or `build_xclbin.sh`.
