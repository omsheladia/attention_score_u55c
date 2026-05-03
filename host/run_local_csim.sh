#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/.."

VECTOR_DIR="attention_score_u55c/sim/attention_score_tile"
SIM_DIR="attention_score_u55c/sim"
COMMON_INC="attention_score_u55c/hls/common"

python3 attention_score_u55c/model/export_attention_score_vectors.py --output-dir "$VECTOR_DIR"

mkdir -p "$SIM_DIR"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp \
  attention_score_u55c/hls/attention_score/tb_attention_score.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_attention_score"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/causal_mask/causal_mask_core_hls.cpp \
  attention_score_u55c/hls/causal_mask/tb_causal_mask.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_causal_mask"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/score_scale/score_scale_core_hls.cpp \
  attention_score_u55c/hls/score_scale/tb_score_scale.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_score_scale"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/softmax/softmax_core_hls.cpp \
  attention_score_u55c/hls/softmax/tb_softmax.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_softmax"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/softmax_full_row/softmax_full_row_hls.cpp \
  attention_score_u55c/hls/softmax_full_row/tb_softmax_full_row.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_softmax_full_row"

g++ -O2 -std=c++17 \
  attention_score_u55c/hls/v_weighted_sum/v_weighted_sum_core_hls.cpp \
  attention_score_u55c/hls/v_weighted_sum/tb_v_weighted_sum.cpp \
  -I"$COMMON_INC" \
  -o "$SIM_DIR/tb_v_weighted_sum"

"$SIM_DIR/tb_attention_score" "$VECTOR_DIR"
"$SIM_DIR/tb_causal_mask" "$VECTOR_DIR"
"$SIM_DIR/tb_score_scale" "$VECTOR_DIR"
"$SIM_DIR/tb_softmax" "$VECTOR_DIR"
"$SIM_DIR/tb_softmax_full_row"
"$SIM_DIR/tb_v_weighted_sum"

echo "Local attention-score C++ simulation chain PASSED"
