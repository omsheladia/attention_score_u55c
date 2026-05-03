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


@dataclass(frozen=True)
class FullAttentionScoreResult:
    raw_scores: list[list[int]]
    logits: list[list[float]]
    softmax: list[list[float]]


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


def zeros_2d_float(rows: int, cols: int, fill: float = 0.0) -> list[list[float]]:
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


def softmax_full_rows(logits: list[list[float]]) -> list[list[float]]:
    """Softmax over each unpadded full row.

    This is the Track A Step 3 reference helper for S-wide normalization. It is
    intentionally separate from softmax_rows(), which is fixed to an 8 x 64 tile.
    """

    if not logits:
        raise ValueError("logits must be non-empty")
    key_col_count = len(logits[0])
    if key_col_count == 0:
        raise ValueError("logits must have at least one column")
    if any(len(row) != key_col_count for row in logits):
        raise ValueError("All logits rows must have the same length")

    probs: list[list[float]] = []
    for row in logits:
        row_max = max(row)
        exp_vals = [math.exp(value - row_max) for value in row]
        exp_sum = sum(exp_vals)
        probs.append([value / exp_sum for value in exp_vals])
    return probs


def deterministic_full_sequence(seq_len: int) -> tuple[list[list[int]], list[list[int]]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return deterministic_tiles(seq_len, seq_len)


def _validate_full_inputs(q_full: list[list[int]], k_full: list[list[int]]) -> None:
    if not q_full or not k_full:
        raise ValueError("q_full and k_full must both be non-empty")
    if any(len(row) != HEAD_DIM for row in q_full):
        raise ValueError(f"Expected all q_full rows to have length {HEAD_DIM}")
    if any(len(row) != HEAD_DIM for row in k_full):
        raise ValueError(f"Expected all k_full rows to have length {HEAD_DIM}")


def _total_scale(
    q_scale: float,
    k_scale: float,
    attn_scale: float | None,
) -> float:
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(float(HEAD_DIM))
    return float(q_scale) * float(k_scale) * float(attn_scale)


def brute_force_full_attention_score(
    q_full: list[list[int]],
    k_full: list[list[int]],
    q_scale: float = 0.03125,
    k_scale: float = 0.02734375,
    *,
    query_pos_base: int = 0,
    key_pos_base: int = 0,
    attn_scale: float | None = None,
) -> FullAttentionScoreResult:
    """Direct full-matrix score/mask/scale/softmax reference."""

    _validate_full_inputs(q_full, k_full)
    query_count = len(q_full)
    key_count = len(k_full)
    total_scale = _total_scale(q_scale, k_scale, attn_scale)

    raw_scores = zeros_2d(query_count, key_count, fill=0)
    logits = zeros_2d_float(query_count, key_count, fill=0.0)
    for row_idx, q_row in enumerate(q_full):
        query_pos = query_pos_base + row_idx
        for col_idx, k_row in enumerate(k_full):
            accum = 0
            for dim in range(HEAD_DIM):
                accum += int(q_row[dim]) * int(k_row[dim])
            raw_scores[row_idx][col_idx] = accum

            key_pos = key_pos_base + col_idx
            masked_or_raw = MASK_NEG_INF if key_pos > query_pos else accum
            logits[row_idx][col_idx] = float(masked_or_raw) * total_scale

    return FullAttentionScoreResult(
        raw_scores=raw_scores,
        logits=logits,
        softmax=softmax_full_rows(logits),
    )


def compute_full_attention_score(
    q_full: list[list[int]],
    k_full: list[list[int]],
    q_scale: float = 0.03125,
    k_scale: float = 0.02734375,
    *,
    query_pos_base: int = 0,
    key_pos_base: int = 0,
    attn_scale: float | None = None,
) -> FullAttentionScoreResult:
    """Tile-shaped full-sequence attention-score reference.

    Pass 1 covers the full score matrix with 8 x 64 hardware-shaped tiles and
    assembles an unpadded full logit matrix. Pass 2 normalizes each full row
    across all keys using softmax_full_rows().
    """

    _validate_full_inputs(q_full, k_full)
    query_count = len(q_full)
    key_count = len(k_full)
    raw_scores = zeros_2d(query_count, key_count, fill=0)
    logits = zeros_2d_float(query_count, key_count, fill=0.0)

    for query_base in range(0, query_count, SCORE_ROWS_PER_CHUNK):
        q_chunk = q_full[query_base : query_base + SCORE_ROWS_PER_CHUNK]
        query_row_count = len(q_chunk)
        for key_base in range(0, key_count, SCORE_K_TILE):
            k_chunk = k_full[key_base : key_base + SCORE_K_TILE]
            key_col_count = len(k_chunk)
            meta = ScoreTileMetadata(
                query_pos_base=query_pos_base + query_base,
                key_pos_base=key_pos_base + key_base,
                query_row_count=query_row_count,
                key_col_count=key_col_count,
            )

            score_active = compute_attention_score_tile(q_chunk, k_chunk)
            score_tile = build_padded_score_tile(score_active)
            masked = apply_causal_mask(score_tile, meta)
            scaled = scale_scores(
                masked,
                q_scale=q_scale,
                k_scale=k_scale,
                attn_scale=attn_scale,
            )

            for row_local in range(query_row_count):
                row_global = query_base + row_local
                for col_local in range(key_col_count):
                    col_global = key_base + col_local
                    raw_scores[row_global][col_global] = score_tile[row_local][col_local]
                    logits[row_global][col_global] = scaled[row_local][col_local]

    return FullAttentionScoreResult(
        raw_scores=raw_scores,
        logits=logits,
        softmax=softmax_full_rows(logits),
    )


def max_abs_diff(
    lhs: list[list[int]] | list[list[float]],
    rhs: list[list[int]] | list[list[float]],
) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"Row count mismatch: {len(lhs)} vs {len(rhs)}")
    max_diff = 0.0
    for row_idx, (lhs_row, rhs_row) in enumerate(zip(lhs, rhs)):
        if len(lhs_row) != len(rhs_row):
            raise ValueError(f"Column count mismatch at row {row_idx}")
        for lhs_value, rhs_value in zip(lhs_row, rhs_row):
            diff = abs(float(lhs_value) - float(rhs_value))
            if diff > max_diff:
                max_diff = diff
    return max_diff


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


def deterministic_v(seq_len: int) -> list[list[float]]:
    """Deterministic float32 V matrix of shape (seq_len, HEAD_DIM).

    Uses a different pattern from Q/K so V values are independent.
    """
    v = [[0.0] * HEAD_DIM for _ in range(seq_len)]
    for row in range(seq_len):
        for dim in range(HEAD_DIM):
            v[row][dim] = float(((row * 3) + (dim * 7)) % 15 - 7)
    return v


def compute_v_weighted_sum(
    softmax_weights: list[list[float]],
    v_full: list[list[float]],
) -> list[list[float]]:
    """Tiled softmax_weights @ v_full -> attn_out.

    Mirrors the FPGA V-kernel: one (8 x SCORE_K_TILE) weight tile and one
    (SCORE_K_TILE x HEAD_DIM) V tile per call, accumulating partial results
    across all K/V chunks before writing the final attn_out row.

    softmax_weights : (S, S)        full-row softmax probabilities
    v_full          : (S, HEAD_DIM) value matrix (float32)
    Returns attn_out: (S, HEAD_DIM)
    """
    s = len(softmax_weights)
    head_dim = len(v_full[0]) if v_full else HEAD_DIM
    attn_out = zeros_2d_float(s, head_dim)

    for q_start in range(0, s, SCORE_ROWS_PER_CHUNK):
        q_end = min(q_start + SCORE_ROWS_PER_CHUNK, s)
        q_count = q_end - q_start
        accum = zeros_2d_float(q_count, head_dim)

        for k_start in range(0, s, SCORE_K_TILE):
            k_end = min(k_start + SCORE_K_TILE, s)
            k_count = k_end - k_start
            for r in range(q_count):
                for d in range(head_dim):
                    for c in range(k_count):
                        accum[r][d] += (
                            softmax_weights[q_start + r][k_start + c]
                            * v_full[k_start + c][d]
                        )

        for r in range(q_count):
            attn_out[q_start + r] = accum[r][:]

    return attn_out


def shape_2d(tile: list[list[int]] | list[list[float]]) -> tuple[int, int]:
    return len(tile), len(tile[0]) if tile else 0


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not lengths:
        raise ValueError("At least one sequence length is required")
    for seq_len in lengths:
        if seq_len <= 0:
            raise ValueError("Sequence lengths must be positive")
    return lengths


def run_full_tiling_checks(
    lengths: list[int],
    *,
    q_scale: float,
    k_scale: float,
    tolerance: float,
) -> None:
    print("Full-sequence tiled attention-score checks")
    print("seq_len,q_chunks,k_chunks,raw_diff,logit_diff,softmax_diff")
    for seq_len in lengths:
        q_full, k_full = deterministic_full_sequence(seq_len)
        tiled = compute_full_attention_score(q_full, k_full, q_scale=q_scale, k_scale=k_scale)
        brute = brute_force_full_attention_score(q_full, k_full, q_scale=q_scale, k_scale=k_scale)

        raw_diff = max_abs_diff(tiled.raw_scores, brute.raw_scores)
        logit_diff = max_abs_diff(tiled.logits, brute.logits)
        softmax_diff = max_abs_diff(tiled.softmax, brute.softmax)
        print(
            f"{seq_len},"
            f"{math.ceil(seq_len / SCORE_ROWS_PER_CHUNK)},"
            f"{math.ceil(seq_len / SCORE_K_TILE)},"
            f"{raw_diff:.8e},"
            f"{logit_diff:.8e},"
            f"{softmax_diff:.8e}"
        )
        if raw_diff > tolerance or logit_diff > tolerance or softmax_diff > tolerance:
            raise SystemExit(f"Full tiling check failed for S={seq_len}")
    print("Full-sequence tiled attention-score checks PASSED")


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
    parser.add_argument(
        "--check-full-tiling",
        action="store_true",
        help="Validate full-sequence tiled score/softmax against brute force.",
    )
    parser.add_argument(
        "--lengths",
        default="8,64,128,256,512",
        help="Comma-separated lengths for --check-full-tiling.",
    )
    parser.add_argument("--q-scale", type=float, default=0.03125, help="Quantized Q dequant scale.")
    parser.add_argument("--k-scale", type=float, default=0.02734375, help="Quantized K dequant scale.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Validation tolerance.")
    args = parser.parse_args()

    if args.check_full_tiling:
        run_full_tiling_checks(
            parse_lengths(args.lengths),
            q_scale=args.q_scale,
            k_scale=args.k_scale,
            tolerance=args.tolerance,
        )
        return

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
