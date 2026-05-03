"""
Export real TinyLlama vectors for the current XRT chain.

This is Track C Step 4/5 for the current design. It uses TinyLlama as a PyTorch
data source, extracts one layer/head, quantizes Q/K to INT8, keeps V as float32
for the Track B weighted-sum stage, and writes the file names consumed by the
host app.

For the legacy single-tile path it writes:
    q_tile.txt, k_tile.txt, kernel_meta.txt,
    score_raw.txt, score_masked.txt, score_scaled.txt, score_softmax.txt,
    v_full.txt, attn_ref_float.txt

For the full-sequence tiled path it also writes:
    q_full.txt, k_full.txt, attn_out.txt

The full-sequence path currently supports S <= 512, matching the full-row
softmax kernel and XRT host path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attention_score_ref import (
    HEAD_DIM,
    SCORE_K_TILE,
    SCORE_ROWS_PER_CHUNK,
    ScoreTileMetadata,
    apply_causal_mask,
    build_padded_k_tile,
    build_padded_q_tile,
    build_padded_score_tile,
    compute_attention_score_tile,
    compute_full_attention,
    compute_v_weighted_sum_partial,
    max_abs_diff,
    pack_score_chunk,
    scale_scores,
    softmax_rows,
)
from check_tinyllama_setup import DEFAULT_MODEL_ID, DEFAULT_TEXT
from export_attention_score_vectors import (
    pad_float_matrix,
    write_kernel_meta,
    write_text_float_matrix,
    write_text_matrix,
    write_text_vector,
)
from extract_tinyllama_qkv import (
    add_qkv_capture_hook,
    causal_attention_reference,
    choose_device,
    choose_dtype,
    map_q_head_to_kv_head,
    quantization_metrics,
    require_packages,
)


DEFAULT_OUTPUT_DIR = "sim/real_tinyllama_tile"
MAX_FULL_SEQUENCE_LEN = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real TinyLlama Q/K vectors for the current single-tile XRT flow.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id or local model path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Prompt text to export. Default: {DEFAULT_TEXT!r}",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help=(
            "Optional token count to export from the start of the prompt. "
            f"When set, writes full-sequence tiled vectors. Max: {MAX_FULL_SEQUENCE_LEN}."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Transformer layer index to inspect. Default: 0.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Q head index to export. Default: 0.",
    )
    parser.add_argument(
        "--kv-head",
        type=int,
        default=None,
        help="Optional KV head index. Default maps from Q head for grouped-query attention.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output vector directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use. Default: auto.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype. Default: float16 on CUDA, float32 on CPU.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HuggingFace cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from local HuggingFace cache or local model path.",
    )
    return parser.parse_args()


def tensor_to_int_matrix(tensor: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in tensor.tolist()]


def tensor_to_float_matrix(tensor: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in tensor.tolist()]


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def max_abs_diff_float(lhs: list[list[float]], rhs: list[list[float]]) -> float:
    return max_abs_diff(lhs, rhs)


def main() -> None:
    args = parse_args()
    require_packages()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device, torch)
    dtype = choose_dtype(args.dtype, device, torch)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    print(f"Loading model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    num_layers = len(model.model.layers)
    if args.layer < 0 or args.layer >= num_layers:
        raise SystemExit(f"--layer must be in [0, {num_layers - 1}]")

    attn_module = model.model.layers[args.layer].self_attn
    captured: dict[str, Any] = {}
    hook_handle = add_qkv_capture_hook(attn_module, captured)

    inputs = tokenizer(args.text, return_tensors="pt")
    tokenized_seq_len = int(inputs["input_ids"].shape[1])
    if args.seq_len is not None:
        if args.seq_len <= 0:
            raise SystemExit("--seq-len must be positive")
        if args.seq_len > tokenized_seq_len:
            raise SystemExit(
                f"--seq-len {args.seq_len} exceeds tokenized prompt length {tokenized_seq_len}"
            )
        if args.seq_len > MAX_FULL_SEQUENCE_LEN:
            raise SystemExit(
                f"--seq-len must be <= {MAX_FULL_SEQUENCE_LEN} for the current full-row softmax path"
            )
        inputs = {name: value[:, : args.seq_len] for name, value in inputs.items()}
    elif tokenized_seq_len > MAX_FULL_SEQUENCE_LEN:
        raise SystemExit(
            f"Prompt tokenized to {tokenized_seq_len}; current full-row softmax path supports "
            f"at most {MAX_FULL_SEQUENCE_LEN} tokens. Use --seq-len to truncate."
        )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_ids = inputs["input_ids"].detach().cpu()
    seq_len = int(input_ids.shape[1])

    print(f"Input ids shape: {tuple(input_ids.shape)}")
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        hook_handle.remove()

    if not {"q_rot", "k_rot", "v"}.issubset(captured):
        raise SystemExit("Q/K/V capture failed")

    q_rot = captured["q_rot"]
    k_rot = captured["k_rot"]
    v = captured["v"]
    _, num_q_heads, _, head_dim = q_rot.shape
    _, num_kv_heads, _, k_head_dim = k_rot.shape
    _, v_num_kv_heads, _, v_head_dim = v.shape

    if head_dim != HEAD_DIM or k_head_dim != HEAD_DIM or v_head_dim != HEAD_DIM:
        raise SystemExit(
            f"Expected head_dim {HEAD_DIM}, got Q={head_dim}, K={k_head_dim}, V={v_head_dim}"
        )
    if v_num_kv_heads != num_kv_heads:
        raise SystemExit("K and V KV-head counts do not match")
    if args.head < 0 or args.head >= num_q_heads:
        raise SystemExit(f"--head must be in [0, {num_q_heads - 1}]")

    kv_head = args.kv_head
    if kv_head is None:
        kv_head = map_q_head_to_kv_head(args.head, num_q_heads, num_kv_heads)
    if kv_head < 0 or kv_head >= num_kv_heads:
        raise SystemExit(f"--kv-head must be in [0, {num_kv_heads - 1}]")

    q_head = q_rot[0, args.head, :, :]
    k_head = k_rot[0, kv_head, :, :]
    v_head = v[0, kv_head, :, :]
    _, _, attn_ref = causal_attention_reference(q_head, k_head, v_head)
    quant = quantization_metrics(q_head, k_head)

    q_int_matrix = tensor_to_int_matrix(quant["q_int8"])
    k_int_matrix = tensor_to_int_matrix(quant["k_int8"])
    v_float_matrix = tensor_to_float_matrix(v_head.float())
    attn_ref_float = tensor_to_float_matrix(attn_ref)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_sequence_export = args.seq_len is not None or seq_len > SCORE_ROWS_PER_CHUNK
    if full_sequence_export:
        full = compute_full_attention(
            q_int_matrix,
            k_int_matrix,
            v_float_matrix,
            q_scale=quant["q_scale"],
            k_scale=quant["k_scale"],
        )

        write_text_matrix(output_dir / "q_full.txt", q_int_matrix)
        write_text_matrix(output_dir / "k_full.txt", k_int_matrix)
        write_text_float_matrix(output_dir / "v_full.txt", v_float_matrix)
        write_text_matrix(output_dir / "score_raw.txt", full.raw_scores)
        write_text_float_matrix(output_dir / "score_scaled.txt", full.logits)
        write_text_float_matrix(output_dir / "score_softmax.txt", full.softmax)
        write_text_float_matrix(output_dir / "attn_out.txt", full.attn_out)
        write_text_float_matrix(output_dir / "attn_ref_float.txt", attn_ref_float)
        write_text_float_matrix(output_dir / "q_float.txt", tensor_to_float_matrix(q_head.float()))
        write_text_float_matrix(output_dir / "k_float.txt", tensor_to_float_matrix(k_head.float()))

        q_first = build_padded_q_tile(q_int_matrix[:SCORE_ROWS_PER_CHUNK])
        k_first = build_padded_k_tile(k_int_matrix[:SCORE_K_TILE])
        first_key_cols = min(SCORE_K_TILE, seq_len)
        first_query_rows = min(SCORE_ROWS_PER_CHUNK, seq_len)
        first_weights = [
            row[:first_key_cols]
            for row in full.softmax[:first_query_rows]
        ]
        first_v = v_float_matrix[:first_key_cols]
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
            query_row_count=first_query_rows,
            key_col_count=first_key_cols,
            query_pos_base=0,
            key_pos_base=0,
            q_scale=quant["q_scale"],
            k_scale=quant["k_scale"],
        )

        attn_ref_max_error = max_abs_diff_float(full.attn_out, attn_ref_float)
        metadata = {
            "source": "TinyLlama real Q/K/V extraction",
            "mode": "full_sequence",
            "model_id": args.model_id,
            "text": args.text,
            "input_ids": input_ids[0].tolist(),
            "seq_len": seq_len,
            "layer": args.layer,
            "q_head": args.head,
            "kv_head": kv_head,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": HEAD_DIM,
            "q_scale": quant["q_scale"],
            "k_scale": quant["k_scale"],
            "total_scale": quant["total_scale"],
            "q_recon_max_error": quant["q_recon_max_error"],
            "k_recon_max_error": quant["k_recon_max_error"],
            "score_dequant_max_error": quant["score_max_error"],
            "score_dequant_mean_error": quant["score_mean_error"],
            "quantized_vs_float_attn_out_max_error": attn_ref_max_error,
            "host_mode": "--vectors with q_full.txt/k_full.txt uses tiled full-sequence path",
            "attn_out_output": "attn_out.txt is the quantized Q/K FPGA reference; attn_ref_float.txt is the full-float TinyLlama reference.",
        }
        write_metadata(output_dir / "metadata.json", metadata)

        print(f"Logits shape: {tuple(outputs.logits.shape)}")
        print(f"Captured Q_rot shape: {tuple(q_rot.shape)}")
        print(f"Captured K_rot shape: {tuple(k_rot.shape)}")
        print(f"Captured V shape: {tuple(v.shape)}")
        print(f"Selected q_head shape: {tuple(q_head.shape)}")
        print(f"Selected k_head shape: {tuple(k_head.shape)}")
        print(f"Selected v_head shape: {tuple(v_head.shape)}")
        print(f"q_scale: {quant['q_scale']:.12e}")
        print(f"k_scale: {quant['k_scale']:.12e}")
        print(f"total_scale: {quant['total_scale']:.12e}")
        print(f"Score dequant max error: {quant['score_max_error']:.8e}")
        print(f"Quantized-vs-float attn_out max error: {attn_ref_max_error:.8e}")
        print(f"Wrote real TinyLlama full-sequence vectors to {output_dir}")
        print("Real TinyLlama vector export OK")
        return

    q_tile_padded = build_padded_q_tile(q_int_matrix)
    k_tile_padded = build_padded_k_tile(k_int_matrix)

    meta = ScoreTileMetadata(
        query_pos_base=0,
        key_pos_base=0,
        query_row_count=seq_len,
        key_col_count=seq_len,
    )
    score_active = compute_attention_score_tile(q_int_matrix, k_int_matrix)
    score_raw = build_padded_score_tile(score_active)
    score_masked = apply_causal_mask(score_raw, meta)
    score_scaled = scale_scores(score_masked, quant["q_scale"], quant["k_scale"])
    score_softmax = softmax_rows(score_scaled, meta.key_col_count, meta.query_row_count)
    score_packed = pack_score_chunk(score_raw)

    write_text_matrix(output_dir / "q_tile.txt", q_tile_padded)
    write_text_matrix(output_dir / "k_tile.txt", k_tile_padded)
    write_text_matrix(output_dir / "score_raw.txt", score_raw)
    write_text_matrix(output_dir / "score_masked.txt", score_masked)
    write_text_vector(output_dir / "score_packed.txt", score_packed)
    write_text_float_matrix(output_dir / "score_scaled.txt", score_scaled)
    write_text_float_matrix(output_dir / "score_softmax.txt", score_softmax)
    write_text_float_matrix(output_dir / "q_float.txt", tensor_to_float_matrix(q_head.float()))
    write_text_float_matrix(output_dir / "k_float.txt", tensor_to_float_matrix(k_head.float()))
    write_text_float_matrix(output_dir / "v_full.txt", v_float_matrix)
    write_text_float_matrix(output_dir / "attn_ref_float.txt", attn_ref_float)
    write_kernel_meta(
        output_dir / "kernel_meta.txt",
        query_row_count=meta.query_row_count,
        key_col_count=meta.key_col_count,
        query_pos_base=meta.query_pos_base,
        key_pos_base=meta.key_pos_base,
        q_scale=quant["q_scale"],
        k_scale=quant["k_scale"],
    )

    metadata = {
        "source": "TinyLlama real Q/K/V extraction",
        "mode": "single_tile",
        "model_id": args.model_id,
        "text": args.text,
        "input_ids": input_ids[0].tolist(),
        "seq_len": seq_len,
        "layer": args.layer,
        "q_head": args.head,
        "kv_head": kv_head,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": HEAD_DIM,
        "query_row_count": meta.query_row_count,
        "key_col_count": meta.key_col_count,
        "query_pos_base": meta.query_pos_base,
        "key_pos_base": meta.key_pos_base,
        "q_scale": quant["q_scale"],
        "k_scale": quant["k_scale"],
        "total_scale": quant["total_scale"],
        "q_recon_max_error": quant["q_recon_max_error"],
        "k_recon_max_error": quant["k_recon_max_error"],
        "score_dequant_max_error": quant["score_max_error"],
        "score_dequant_mean_error": quant["score_mean_error"],
        "current_design_scope": "single Q tile, score/mask_scale/softmax plus optional Track B V weighted-sum verification",
        "v_note": "v_full.txt feeds the Track B V weighted-sum stage in --vectors mode; attn_ref_float.txt is the full-float PyTorch reference.",
    }
    write_metadata(output_dir / "metadata.json", metadata)

    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Captured Q_rot shape: {tuple(q_rot.shape)}")
    print(f"Captured K_rot shape: {tuple(k_rot.shape)}")
    print(f"Captured V shape: {tuple(v.shape)}")
    print(f"Selected q_head shape: {tuple(q_head.shape)}")
    print(f"Selected k_head shape: {tuple(k_head.shape)}")
    print(f"Selected v_head shape: {tuple(v_head.shape)}")
    print(f"q_scale: {quant['q_scale']:.12e}")
    print(f"k_scale: {quant['k_scale']:.12e}")
    print(f"total_scale: {quant['total_scale']:.12e}")
    print(f"Score dequant max error: {quant['score_max_error']:.8e}")
    print(f"Wrote real TinyLlama single-tile vectors to {output_dir}")
    print("Real TinyLlama vector export OK")


if __name__ == "__main__":
    main()
