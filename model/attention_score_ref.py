"""
Isolated attention-score reference for U55C offload planning.

This module is intentionally dependency-light so it can run in a plain Python
environment. The core offload boundary stays the same:

    Q_rot_int8 @ K_rot_int8^T -> score_raw_int32
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


HEAD_DIM = 64
SCORE_ROWS_PER_CHUNK = 8
SCORE_K_TILE = 64
MASK_NEG_INF = -1_000_000_000


@dataclass(frozen=True)
class ScoreTileMetadata:
    query_pos_base: int
    key_pos_base: int
    query_row_count: int
    key_col_count: int
    head_dim: int = HEAD_DIM


def compute_attention_score_tile(
    q_tile: list[list[int]],
    k_tile: list[list[int]],
) -> list[list[int]]:
    if not q_tile or not k_tile:
        raise ValueError("q_tile and k_tile must both be non-empty")
    if any(len(row) != HEAD_DIM for row in q_tile):
        raise ValueError(f"Expected all q_tile rows to have length {HEAD_DIM}")
    if any(len(row) != HEAD_DIM for row in k_tile):
        raise ValueError(f"Expected all k_tile rows to have length {HEAD_DIM}")
    if len(q_tile) > SCORE_ROWS_PER_CHUNK:
        raise ValueError(f"q_tile row count must be <= {SCORE_ROWS_PER_CHUNK}")
    if len(k_tile) > SCORE_K_TILE:
        raise ValueError(f"k_tile row count must be <= {SCORE_K_TILE}")

    score_active: list[list[int]] = []
    for q_row in q_tile:
        out_row: list[int] = []
        for k_row in k_tile:
            accum = 0
            for dim in range(HEAD_DIM):
                accum += int(q_row[dim]) * int(k_row[dim])
            out_row.append(accum)
        score_active.append(out_row)
    return score_active


def zeros_2d(rows: int, cols: int, fill: int = 0) -> list[list[int]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def build_padded_score_tile(score_active: list[list[int]]) -> list[list[int]]:
    score_tile = zeros_2d(SCORE_ROWS_PER_CHUNK, SCORE_K_TILE, fill=0)
    for row_idx, row in enumerate(score_active):
        for col_idx, value in enumerate(row):
            score_tile[row_idx][col_idx] = int(value)
    return score_tile


def build_padded_q_tile(q_tile: list[list[int]]) -> list[list[int]]:
    q_padded = zeros_2d(SCORE_ROWS_PER_CHUNK, HEAD_DIM, fill=0)
    for row_idx, row in enumerate(q_tile):
        for dim_idx, value in enumerate(row):
            q_padded[row_idx][dim_idx] = int(value)
    return q_padded


def build_padded_k_tile(k_tile: list[list[int]]) -> list[list[int]]:
    k_padded = zeros_2d(SCORE_K_TILE, HEAD_DIM, fill=0)
    for row_idx, row in enumerate(k_tile):
        for dim_idx, value in enumerate(row):
            k_padded[row_idx][dim_idx] = int(value)
    return k_padded


def apply_causal_mask(
    score_tile: list[list[int]],
    meta: ScoreTileMetadata,
) -> list[list[int]]:
    masked = zeros_2d(SCORE_ROWS_PER_CHUNK, SCORE_K_TILE, fill=MASK_NEG_INF)

    for row_local in range(meta.query_row_count):
        query_pos = meta.query_pos_base + row_local
        for col_local in range(meta.key_col_count):
            key_pos = meta.key_pos_base + col_local
            if key_pos <= query_pos:
                masked[row_local][col_local] = int(score_tile[row_local][col_local])

    return masked


def scale_scores(
    score_tile: list[list[int]],
    q_scale: float,
    k_scale: float,
    attn_scale: float | None = None,
) -> list[list[float]]:
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(float(HEAD_DIM))
    total_scale = float(q_scale) * float(k_scale) * float(attn_scale)

    scaled: list[list[float]] = []
    for row in score_tile:
        scaled.append([float(value) * total_scale for value in row])
    return scaled


def softmax_rows(
    score_tile: list[list[float]],
    key_col_count: int,
    query_row_count: int,
) -> list[list[float]]:
    probs = [[0.0 for _ in range(SCORE_K_TILE)] for _ in range(SCORE_ROWS_PER_CHUNK)]

    for row_idx in range(query_row_count):
        row = score_tile[row_idx]
        row_max = max(row[:key_col_count]) if key_col_count > 0 else 0.0
        exp_vals = [math.exp(value - row_max) for value in row[:key_col_count]]
        exp_sum = sum(exp_vals) if exp_vals else 1.0
        for col_idx in range(key_col_count):
            probs[row_idx][col_idx] = exp_vals[col_idx] / exp_sum

    return probs


def pack_score_chunk(score_tile: list[list[int]]) -> list[int]:
    packed = [0 for _ in range(SCORE_ROWS_PER_CHUNK * SCORE_K_TILE)]
    for row_local in range(SCORE_ROWS_PER_CHUNK):
        lane_base = row_local * SCORE_K_TILE
        for col_local in range(SCORE_K_TILE):
            packed[lane_base + col_local] = int(score_tile[row_local][col_local])
    return packed


def deterministic_tiles(
    query_row_count: int,
    key_col_count: int,
) -> tuple[list[list[int]], list[list[int]]]:
    q_tile = zeros_2d(query_row_count, HEAD_DIM, fill=0)
    k_tile = zeros_2d(key_col_count, HEAD_DIM, fill=0)

    for row in range(query_row_count):
        for dim in range(HEAD_DIM):
            q_tile[row][dim] = ((row * 5) + (dim * 3)) % 15 - 7

    for col in range(key_col_count):
        for dim in range(HEAD_DIM):
            k_tile[col][dim] = ((col * 7) - (dim * 2)) % 15 - 7

    return q_tile, k_tile


def shape_2d(tile: list[list[int]] | list[list[float]]) -> tuple[int, int]:
    return len(tile), len(tile[0]) if tile else 0


def demo_inputs() -> tuple[list[list[int]], list[list[int]], ScoreTileMetadata]:
    query_row_count = 4
    key_col_count = 10
    q_tile, k_tile = deterministic_tiles(query_row_count, key_col_count)
    meta = ScoreTileMetadata(
        query_pos_base=6,
        key_pos_base=0,
        query_row_count=query_row_count,
        key_col_count=key_col_count,
    )
    return q_tile, k_tile, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference attention-score tile math.")
    parser.add_argument("--show-demo", action="store_true", help="Print one deterministic demo case.")
    args = parser.parse_args()

    if not args.show_demo:
        parser.print_help()
        return

    q_tile, k_tile, meta = demo_inputs()
    score_active = compute_attention_score_tile(q_tile, k_tile)
    score_tile = build_padded_score_tile(score_active)
    masked = apply_causal_mask(score_tile, meta)
    scaled = scale_scores(masked, q_scale=0.03125, k_scale=0.02734375)

    print("Q tile shape:", shape_2d(q_tile))
    print("K tile shape:", shape_2d(k_tile))
    print("Raw score tile shape:", shape_2d(score_tile))
    print("Masked score tile shape:", shape_2d(masked))
    print("Scaled score tile shape:", shape_2d(scaled))
    print("Packed lane length:", len(pack_score_chunk(score_tile)))


if __name__ == "__main__":
    main()
