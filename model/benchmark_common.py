"""
Shared helpers for Track D CPU/GPU attention baselines.

The benchmark scope is the isolated one-head attention block used by this repo:

    Q_int8 @ K_int8.T -> causal mask -> scale -> softmax -> optional softmax @ V

Synthetic mode covers the scaling lengths used for baseline curves. Vector mode
loads a current exported directory such as sim/real_tinyllama_tile.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


HEAD_DIM = 64
Q_TILE_ROWS = 8
K_TILE_COLS = 64
MASK_NEG_INF = -1_000_000_000.0
DEFAULT_LENGTHS = (8, 64, 128, 256, 512)
DEFAULT_Q_SCALE = 0.03125
DEFAULT_K_SCALE = 0.02734375


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    q_int8: np.ndarray
    k_int8: np.ndarray
    v_float: np.ndarray
    q_scale: float
    k_scale: float
    source: str

    @property
    def seq_len(self) -> int:
        return int(self.q_int8.shape[0])

    @property
    def total_scale(self) -> float:
        return float(self.q_scale) * float(self.k_scale) * (1.0 / math.sqrt(HEAD_DIM))


@dataclass(frozen=True)
class TimingResult:
    mean_ms: float
    std_ms: float
    last_result: object


def parse_lengths(text: str) -> list[int]:
    lengths = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not lengths:
        raise ValueError("At least one sequence length is required")
    for seq_len in lengths:
        if seq_len <= 0:
            raise ValueError("Sequence lengths must be positive")
    return lengths


def tiles_per_head(seq_len: int) -> int:
    return math.ceil(seq_len / Q_TILE_ROWS) * math.ceil(seq_len / K_TILE_COLS)


def synthetic_case(seq_len: int, seed: int) -> BenchmarkCase:
    rng = np.random.default_rng(seed + seq_len)
    q_int8 = rng.integers(-127, 128, size=(seq_len, HEAD_DIM), dtype=np.int16).astype(np.int8)
    k_int8 = rng.integers(-127, 128, size=(seq_len, HEAD_DIM), dtype=np.int16).astype(np.int8)
    v_float = rng.standard_normal(size=(seq_len, HEAD_DIM)).astype(np.float32)
    return BenchmarkCase(
        name=f"S={seq_len}",
        q_int8=q_int8,
        k_int8=k_int8,
        v_float=v_float,
        q_scale=DEFAULT_Q_SCALE,
        k_scale=DEFAULT_K_SCALE,
        source="synthetic",
    )


def synthetic_cases(lengths: Iterable[int], seed: int) -> list[BenchmarkCase]:
    return [synthetic_case(seq_len, seed) for seq_len in lengths]


def read_kernel_meta(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 2:
                continue
            key, value = parts
            if key.endswith("_count") or key.endswith("_base"):
                values[key] = int(value)
            else:
                values[key] = float(value)
    return values


def load_flat_values(path: Path, dtype: type[float] | type[int]) -> list[float] | list[int]:
    with path.open("r", encoding="utf-8") as handle:
        if dtype is int:
            return [int(float(value)) for value in handle.read().split()]
        return [float(value) for value in handle.read().split()]


def load_matrix(path: Path, rows: int, cols: int, dtype: type[float] | type[int]) -> np.ndarray:
    values = load_flat_values(path, dtype)
    expected = rows * cols
    if len(values) != expected:
        raise ValueError(f"{path} has {len(values)} values, expected {expected}")
    np_dtype = np.float32 if dtype is float else np.int32
    return np.asarray(values, dtype=np_dtype).reshape(rows, cols)


def load_v_matrix(path: Path, min_rows: int) -> np.ndarray:
    values = load_flat_values(path, float)
    if len(values) % HEAD_DIM != 0:
        raise ValueError(f"{path} value count must be divisible by {HEAD_DIM}")
    rows = len(values) // HEAD_DIM
    if rows < min_rows:
        raise ValueError(f"{path} has {rows} rows, expected at least {min_rows}")
    return np.asarray(values, dtype=np.float32).reshape(rows, HEAD_DIM)[:min_rows, :]


def load_vector_case(vector_dir: str | Path) -> BenchmarkCase:
    vector_path = Path(vector_dir)
    meta = read_kernel_meta(vector_path / "kernel_meta.txt")
    query_rows = int(meta["query_row_count"])
    key_cols = int(meta["key_col_count"])
    if query_rows != key_cols:
        raise ValueError(
            "Benchmark vector mode expects one square self-attention case; "
            f"got query_row_count={query_rows}, key_col_count={key_cols}"
        )

    q_padded = load_matrix(vector_path / "q_tile.txt", Q_TILE_ROWS, HEAD_DIM, int)
    k_padded = load_matrix(vector_path / "k_tile.txt", K_TILE_COLS, HEAD_DIM, int)
    v_path = vector_path / "v_full.txt"
    if v_path.exists():
        v_float = load_v_matrix(v_path, query_rows)
    else:
        v_float = np.zeros((query_rows, HEAD_DIM), dtype=np.float32)

    return BenchmarkCase(
        name=f"{vector_path.name}:S={query_rows}",
        q_int8=q_padded[:query_rows, :].astype(np.int8),
        k_int8=k_padded[:key_cols, :].astype(np.int8),
        v_float=v_float.astype(np.float32),
        q_scale=float(meta["q_scale"]),
        k_scale=float(meta["k_scale"]),
        source=str(vector_path),
    )


def load_expected_softmax(vector_dir: str | Path, seq_len: int) -> np.ndarray | None:
    path = Path(vector_dir) / "score_softmax.txt"
    if not path.exists():
        return None
    padded = load_matrix(path, Q_TILE_ROWS, K_TILE_COLS, float)
    return padded[:seq_len, :seq_len].astype(np.float32)


def apply_causal_mask_and_scale(raw_scores: np.ndarray, total_scale: float) -> np.ndarray:
    seq_len = raw_scores.shape[0]
    masked = raw_scores.astype(np.float32, copy=True)
    future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    masked[future_mask] = MASK_NEG_INF
    return masked * np.float32(total_scale)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    row_max = np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(logits - row_max, dtype=np.float32)
    row_sum = np.sum(exp_vals, axis=1, keepdims=True)
    return (exp_vals / row_sum).astype(np.float32)


def brute_force_score_softmax(case: BenchmarkCase) -> np.ndarray:
    raw_scores = case.q_int8.astype(np.int32) @ case.k_int8.astype(np.int32).T
    logits = apply_causal_mask_and_scale(raw_scores, case.total_scale)
    return softmax_rows(logits)


def brute_force_attention(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    probs = brute_force_score_softmax(case)
    attn_out = probs @ case.v_float.astype(np.float32)
    return probs, attn_out.astype(np.float32)


def tiled_score_softmax(case: BenchmarkCase) -> np.ndarray:
    seq_len = case.seq_len
    raw_scores = np.zeros((seq_len, seq_len), dtype=np.int32)
    q_all = case.q_int8.astype(np.int32)
    k_all = case.k_int8.astype(np.int32)

    for q_base in range(0, seq_len, Q_TILE_ROWS):
        q_chunk = q_all[q_base : q_base + Q_TILE_ROWS, :]
        for k_base in range(0, seq_len, K_TILE_COLS):
            k_chunk = k_all[k_base : k_base + K_TILE_COLS, :]
            raw_scores[
                q_base : q_base + q_chunk.shape[0],
                k_base : k_base + k_chunk.shape[0],
            ] = q_chunk @ k_chunk.T

    logits = apply_causal_mask_and_scale(raw_scores, case.total_scale)
    return softmax_rows(logits)


def tiled_attention(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    probs = tiled_score_softmax(case)
    attn_out = probs @ case.v_float.astype(np.float32)
    return probs, attn_out.astype(np.float32)


def time_function(
    func: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> TimingResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    for _ in range(max(0, warmup)):
        func()

    elapsed_ms: list[float] = []
    last_result: object = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()
        elapsed_ms.append((end - start) * 1000.0)

    mean_ms = statistics.fmean(elapsed_ms)
    std_ms = statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
    return TimingResult(mean_ms=mean_ms, std_ms=std_ms, last_result=last_result)


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))))


def tiles_per_second(seq_len: int, mean_ms: float) -> float:
    if mean_ms <= 0.0:
        return float("inf")
    return tiles_per_head(seq_len) / (mean_ms / 1000.0)
