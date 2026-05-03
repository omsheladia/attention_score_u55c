"""
Track D Step 2: CUDA GPU baseline for the isolated attention block.

Default mode uses deterministic synthetic Q/K/V at S = 8, 64, 128, 256, 512.
Use --vectors sim/real_tinyllama_tile to benchmark the current real TinyLlama
single-tile export. If CUDA is unavailable, the script exits cleanly.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from benchmark_common import (
    DEFAULT_LENGTHS,
    MASK_NEG_INF,
    brute_force_attention,
    load_expected_softmax,
    load_vector_case,
    max_abs_diff,
    parse_lengths,
    synthetic_cases,
    tiles_per_head,
    tiles_per_second,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA GPU baseline for Track D Step 2.")
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
    parser.add_argument("--device", default="cuda", help="CUDA device, for example cuda or cuda:0.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Validation tolerance for GPU vs CPU outputs.",
    )
    return parser.parse_args()


def to_gpu_case(case, torch, device):
    q = torch.as_tensor(case.q_int8.astype(np.float32), device=device)
    k = torch.as_tensor(case.k_int8.astype(np.float32), device=device)
    v = torch.as_tensor(case.v_float.astype(np.float32), device=device)
    return q, k, v, float(case.total_scale)


def gpu_score_softmax(q, k, total_scale, torch):
    scores = q @ k.transpose(0, 1)
    seq_len = scores.shape[0]
    future_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    masked = scores.masked_fill(future_mask, MASK_NEG_INF)
    logits = masked * total_scale
    return torch.softmax(logits, dim=-1)


def gpu_full_attention(q, k, v, total_scale, torch):
    probs = gpu_score_softmax(q, k, total_scale, torch)
    return probs @ v


def time_cuda(func, torch, *, warmup: int, iterations: int) -> tuple[float, float, object]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    for _ in range(max(0, warmup)):
        func()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    last_result = None
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        last_result = func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed_ms.append((end - start) * 1000.0)

    mean_ms = statistics.fmean(elapsed_ms)
    std_ms = statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
    return mean_ms, std_ms, last_result


def print_header(args: argparse.Namespace) -> None:
    print("Track D Step 2 GPU baseline")
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

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the GPU baseline") from exc

    if not torch.cuda.is_available():
        print("CUDA is not available; GPU baseline skipped.")
        return

    device = torch.device(args.device)
    print_header(args)
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print()

    if args.vectors:
        cases = [load_vector_case(args.vectors)]
    else:
        cases = synthetic_cases(parse_lengths(args.lengths), args.seed)

    print("Validation")
    print("case,tiles,softmax_gpu_vs_cpu,attn_out_gpu_vs_cpu,softmax_vs_expected_file")
    for case in cases:
        cpu_probs, cpu_out = brute_force_attention(case)
        q, k, v, total_scale = to_gpu_case(case, torch, device)
        gpu_probs = gpu_score_softmax(q, k, total_scale, torch).detach().cpu().numpy()
        gpu_out = gpu_full_attention(q, k, v, total_scale, torch).detach().cpu().numpy()
        softmax_diff = max_abs_diff(gpu_probs, cpu_probs)
        attn_diff = max_abs_diff(gpu_out, cpu_out)
        expected_diff = ""
        if args.vectors:
            expected = load_expected_softmax(Path(args.vectors), case.seq_len)
            if expected is not None:
                expected_diff = f"{max_abs_diff(gpu_probs, expected):.8e}"
        print(
            f"{case.name},{tiles_per_head(case.seq_len)},"
            f"{softmax_diff:.8e},{attn_diff:.8e},{expected_diff}"
        )
        if softmax_diff > args.tolerance or attn_diff > args.tolerance:
            raise SystemExit(f"Validation failed for {case.name}")
    print()

    print("Timing")
    print("case,tiles,gpu_score_softmax_ms,gpu_full_attn_ms,gpu_tiles_per_second")
    for case in cases:
        q, k, v, total_scale = to_gpu_case(case, torch, device)
        score_mean, score_std, _ = time_cuda(
            lambda q=q, k=k, total_scale=total_scale: gpu_score_softmax(q, k, total_scale, torch),
            torch,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        full_mean, full_std, _ = time_cuda(
            lambda q=q, k=k, v=v, total_scale=total_scale: gpu_full_attention(
                q,
                k,
                v,
                total_scale,
                torch,
            ),
            torch,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"{case.name},{tiles_per_head(case.seq_len)},"
            f"{score_mean:.4f}+/-{score_std:.4f},"
            f"{full_mean:.4f}+/-{full_std:.4f},"
            f"{tiles_per_second(case.seq_len, score_mean):.2f}"
        )


if __name__ == "__main__":
    main()
