"""
Track D Step 1: CPU baseline for the isolated attention block.

Default mode uses deterministic synthetic Q/K/V at S = 8, 64, 128, 256, 512.
Use --vectors sim/real_tinyllama_tile to benchmark the current real TinyLlama
single-tile export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_common import (
    DEFAULT_LENGTHS,
    brute_force_attention,
    brute_force_score_softmax,
    load_expected_softmax,
    load_vector_case,
    max_abs_diff,
    parse_lengths,
    synthetic_cases,
    tiled_attention,
    tiled_score_softmax,
    tiles_per_head,
    tiles_per_second,
    time_function,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU baseline for Track D Step 1.")
    parser.add_argument(
        "--lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
        help="Comma-separated synthetic sequence lengths. Ignored when --vectors is set.",
    )
    parser.add_argument(
        "--vectors",
        default=None,
        help="Optional exported vector directory, for example sim/real_tinyllama_tile.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Synthetic RNG seed.")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up iterations.")
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Validation tolerance for tiled vs brute-force outputs.",
    )
    return parser.parse_args()


def print_header(args: argparse.Namespace) -> None:
    print("Track D Step 1 CPU baseline")
    print("Scope: isolated one-head attention block")
    print("Reports both current score+mask+scale+softmax scope and future +V scope.")
    if args.vectors:
        print(f"Input mode: real/exported vectors from {args.vectors}")
        print("Note: current FPGA comparison includes final one-head softmax @ V.")
    else:
        print(f"Input mode: synthetic lengths {args.lengths}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Timed iterations: {args.iterations}")
    print()


def main() -> None:
    args = parse_args()
    print_header(args)

    if args.vectors:
        cases = [load_vector_case(args.vectors)]
    else:
        cases = synthetic_cases(parse_lengths(args.lengths), args.seed)

    print("Validation")
    print(
        "case,tiles,softmax_tiled_vs_brute,attn_out_tiled_vs_brute,"
        "softmax_vs_expected_file"
    )
    for case in cases:
        brute_probs, brute_out = brute_force_attention(case)
        tiled_probs, tiled_out = tiled_attention(case)
        softmax_diff = max_abs_diff(tiled_probs, brute_probs)
        attn_diff = max_abs_diff(tiled_out, brute_out)
        expected_diff = ""
        if args.vectors:
            expected = load_expected_softmax(Path(args.vectors), case.seq_len)
            if expected is not None:
                expected_diff = f"{max_abs_diff(tiled_probs, expected):.8e}"
        print(
            f"{case.name},{tiles_per_head(case.seq_len)},"
            f"{softmax_diff:.8e},{attn_diff:.8e},{expected_diff}"
        )
        if softmax_diff > args.tolerance or attn_diff > args.tolerance:
            raise SystemExit(f"Validation failed for {case.name}")
    print()

    print("Timing")
    print(
        "case,tiles,tiled_score_softmax_ms,brute_score_softmax_ms,"
        "tiled_full_attn_ms,brute_full_attn_ms,tiled_tiles_per_second"
    )
    for case in cases:
        tiled_score = time_function(
            lambda case=case: tiled_score_softmax(case),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        brute_score = time_function(
            lambda case=case: brute_force_score_softmax(case),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        tiled_full = time_function(
            lambda case=case: tiled_attention(case),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        brute_full = time_function(
            lambda case=case: brute_force_attention(case),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"{case.name},{tiles_per_head(case.seq_len)},"
            f"{tiled_score.mean_ms:.4f}+/-{tiled_score.std_ms:.4f},"
            f"{brute_score.mean_ms:.4f}+/-{brute_score.std_ms:.4f},"
            f"{tiled_full.mean_ms:.4f}+/-{tiled_full.std_ms:.4f},"
            f"{brute_full.mean_ms:.4f}+/-{brute_full.std_ms:.4f},"
            f"{tiles_per_second(case.seq_len, tiled_score.mean_ms):.2f}"
        )


if __name__ == "__main__":
    main()
