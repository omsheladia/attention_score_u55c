"""
Deterministic vector export for the isolated attention-score kernel.

This exporter is plain-Python so it can run even when NumPy is unavailable.
It writes the padded kernel inputs, expected raw scores, and simple metadata
files consumed by the HLS testbench.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_score_ref import (
    HEAD_DIM,
    SCORE_K_TILE,
    SCORE_ROWS_PER_CHUNK,
    ScoreTileMetadata,
    apply_causal_mask,
    build_padded_k_tile,
    build_padded_q_tile,
    build_padded_score_tile,
    compute_full_attention,
    compute_attention_score_tile,
    compute_v_weighted_sum,
    compute_v_weighted_sum_partial,
    deterministic_full_attention_inputs,
    deterministic_v_matrix,
    deterministic_tiles,
    pack_score_chunk,
    scale_scores,
    softmax_rows,
)


def write_text_matrix(path: Path, arr: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in arr:
            for value in row:
                handle.write(f"{int(value)}\n")


def write_text_float_matrix(path: Path, arr: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in arr:
            for value in row:
                handle.write(f"{float(value):.8f}\n")


def write_text_vector(path: Path, arr: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in arr:
            handle.write(f"{int(value)}\n")


def write_json(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def pad_float_matrix(
    rows: list[list[float]],
    row_count: int,
    col_count: int,
) -> list[list[float]]:
    padded = [[0.0 for _ in range(col_count)] for _ in range(row_count)]
    for row_idx, row in enumerate(rows[:row_count]):
        for col_idx, value in enumerate(row[:col_count]):
            padded[row_idx][col_idx] = float(value)
    return padded


def write_kernel_meta(
    path: Path,
    *,
    query_row_count: int,
    key_col_count: int,
    query_pos_base: int,
    key_pos_base: int,
    q_scale: float,
    k_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_scale = float(q_scale) * float(k_scale) * (1.0 / 8.0)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"query_row_count {query_row_count}\n")
        handle.write(f"key_col_count {key_col_count}\n")
        handle.write(f"query_pos_base {query_pos_base}\n")
        handle.write(f"key_pos_base {key_pos_base}\n")
        handle.write(f"q_scale {q_scale:.16f}\n")
        handle.write(f"k_scale {k_scale:.16f}\n")
        handle.write(f"total_scale {total_scale:.16f}\n")


def export_single_tile(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    q_tile, k_tile = deterministic_tiles(args.query_rows, args.key_cols)
    meta = ScoreTileMetadata(
        query_pos_base=args.query_pos_base,
        key_pos_base=args.key_pos_base,
        query_row_count=args.query_rows,
        key_col_count=args.key_cols,
    )

    score_active = compute_attention_score_tile(q_tile, k_tile)
    q_tile_padded = build_padded_q_tile(q_tile)
    k_tile_padded = build_padded_k_tile(k_tile)
    score_raw = build_padded_score_tile(score_active)
    score_masked = apply_causal_mask(score_raw, meta)
    score_scaled = scale_scores(score_masked, q_scale=args.q_scale, k_scale=args.k_scale)
    score_softmax = softmax_rows(score_scaled, args.key_cols, args.query_rows)
    score_packed = pack_score_chunk(score_raw)

    v_active = deterministic_v_matrix(args.key_cols)
    v_tile = pad_float_matrix(v_active, SCORE_K_TILE, HEAD_DIM)
    weights_active = [row[: args.key_cols] for row in score_softmax[: args.query_rows]]
    attn_out_active = compute_v_weighted_sum(weights_active, v_active)
    attn_out_padded = pad_float_matrix(attn_out_active, SCORE_ROWS_PER_CHUNK, HEAD_DIM)
    v_partial_expected = pad_float_matrix(
        compute_v_weighted_sum_partial(weights_active, v_active),
        SCORE_ROWS_PER_CHUNK,
        HEAD_DIM,
    )

    write_text_matrix(output_dir / "q_tile.txt", q_tile_padded)
    write_text_matrix(output_dir / "k_tile.txt", k_tile_padded)
    write_text_matrix(output_dir / "score_raw.txt", score_raw)
    write_text_matrix(output_dir / "score_masked.txt", score_masked)
    write_text_vector(output_dir / "score_packed.txt", score_packed)
    write_text_float_matrix(output_dir / "score_scaled.txt", score_scaled)
    write_text_float_matrix(output_dir / "score_softmax.txt", score_softmax)
    write_text_float_matrix(output_dir / "v_full.txt", v_active)
    write_text_float_matrix(output_dir / "v_tile.txt", v_tile)
    write_text_float_matrix(output_dir / "attn_out.txt", attn_out_padded)
    write_text_float_matrix(output_dir / "v_partial_expected.txt", v_partial_expected)
    write_kernel_meta(
        output_dir / "kernel_meta.txt",
        query_row_count=args.query_rows,
        key_col_count=args.key_cols,
        query_pos_base=args.query_pos_base,
        key_pos_base=args.key_pos_base,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
    )

    return {
        "mode": "single_tile",
        "pattern": (
            "q[row,dim]=((row*5)+(dim*3))%15-7 ; "
            "k[col,dim]=((col*7)-(dim*2))%15-7 ; "
            "v[row,dim]=(((row*11)+(dim*5))%23-11)/8"
        ),
        "head_dim": HEAD_DIM,
        "query_row_count": args.query_rows,
        "key_col_count": args.key_cols,
        "query_pos_base": args.query_pos_base,
        "key_pos_base": args.key_pos_base,
        "q_scale": args.q_scale,
        "k_scale": args.k_scale,
        "score_formula": "score_raw[row, col] = sum_d q_tile[row, d] * k_tile[col, d]",
        "offload_boundary": "Q_rot_int8, K_rot_int8 -> score_raw_int32",
        "softmax_output": "row-wise probabilities over active key columns",
        "v_output": "v_full.txt stores active V rows; v_tile.txt is padded to 64x64",
        "attn_out_output": "attn_out.txt stores padded softmax@V output for active query rows",
        "v_partial_expected": "single K-chunk partial output for hls/v_weighted_sum",
    }


def export_full_sequence(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    q_full, k_full, v_full = deterministic_full_attention_inputs(args.seq_len)
    full = compute_full_attention(
        q_full,
        k_full,
        v_full,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
    )

    write_text_matrix(output_dir / "q_full.txt", q_full)
    write_text_matrix(output_dir / "k_full.txt", k_full)
    write_text_float_matrix(output_dir / "v_full.txt", v_full)
    write_text_matrix(output_dir / "score_raw.txt", full.raw_scores)
    write_text_float_matrix(output_dir / "score_scaled.txt", full.logits)
    write_text_float_matrix(output_dir / "score_softmax.txt", full.softmax)
    write_text_float_matrix(output_dir / "attn_out.txt", full.attn_out)

    q_first = build_padded_q_tile(q_full[:SCORE_ROWS_PER_CHUNK])
    k_first = build_padded_k_tile(k_full[:SCORE_K_TILE])
    first_weights = [
        row[: min(SCORE_K_TILE, args.seq_len)]
        for row in full.softmax[: min(SCORE_ROWS_PER_CHUNK, args.seq_len)]
    ]
    first_v = v_full[: min(SCORE_K_TILE, args.seq_len)]
    write_text_matrix(output_dir / "q_tile.txt", q_first)
    write_text_matrix(output_dir / "k_tile.txt", k_first)
    write_text_float_matrix(output_dir / "v_tile.txt", pad_float_matrix(first_v, SCORE_K_TILE, HEAD_DIM))
    write_text_float_matrix(
        output_dir / "v_partial_expected.txt",
        pad_float_matrix(
            compute_v_weighted_sum_partial(first_weights, first_v),
            SCORE_ROWS_PER_CHUNK,
            HEAD_DIM,
        ),
    )
    write_kernel_meta(
        output_dir / "kernel_meta.txt",
        query_row_count=min(SCORE_ROWS_PER_CHUNK, args.seq_len),
        key_col_count=min(SCORE_K_TILE, args.seq_len),
        query_pos_base=0,
        key_pos_base=0,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
    )

    return {
        "mode": "full_sequence",
        "pattern": (
            "q[row,dim]=((row*5)+(dim*3))%15-7 ; "
            "k[col,dim]=((col*7)-(dim*2))%15-7 ; "
            "v[row,dim]=(((row*11)+(dim*5))%23-11)/8"
        ),
        "head_dim": HEAD_DIM,
        "seq_len": args.seq_len,
        "q_scale": args.q_scale,
        "k_scale": args.k_scale,
        "score_formula": "score_raw = q_full @ k_full.T",
        "softmax_output": "score_softmax.txt stores SxS full-row probabilities",
        "attn_out_output": "attn_out.txt stores Sx64 softmax@V output",
        "v_partial_expected": "first Q chunk and first K/V chunk partial output for hls/v_weighted_sum",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export isolated attention-score vectors.")
    parser.add_argument(
        "--output-dir",
        default="attention_score_u55c/sim/attention_score_tile",
        help="Where to write the generated case files.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Optional full-sequence export length. When set, writes SxS/Sx64 Track B vectors.",
    )
    parser.add_argument("--query-rows", type=int, default=4, help="Active query rows, <= 8.")
    parser.add_argument("--key-cols", type=int, default=10, help="Active key columns, <= 64.")
    parser.add_argument("--query-pos-base", type=int, default=6, help="Base query position.")
    parser.add_argument("--key-pos-base", type=int, default=0, help="Base key position.")
    parser.add_argument("--q-scale", type=float, default=0.03125, help="Quantized Q dequant scale.")
    parser.add_argument("--k-scale", type=float, default=0.02734375, help="Quantized K dequant scale.")
    args = parser.parse_args()

    if args.seq_len is not None and args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if not (1 <= args.query_rows <= SCORE_ROWS_PER_CHUNK):
        raise ValueError(f"--query-rows must be in [1, {SCORE_ROWS_PER_CHUNK}]")
    if not (1 <= args.key_cols <= SCORE_K_TILE):
        raise ValueError(f"--key-cols must be in [1, {SCORE_K_TILE}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seq_len is None:
        metadata = export_single_tile(args, output_dir)
    else:
        metadata = export_full_sequence(args, output_dir)
    write_json(output_dir / "metadata.json", metadata)

    print(f"Wrote deterministic attention-score vectors to {output_dir}")


if __name__ == "__main__":
    main()
