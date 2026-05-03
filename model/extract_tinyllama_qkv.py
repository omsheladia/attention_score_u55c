"""
Track C Steps 2-3: extract real TinyLlama Q/K/V tensors and quantize Q/K.

This script runs TinyLlama in PyTorch only. It does not interact with HLS, XRT,
or the FPGA. The extracted tensors are the real-data counterpart to the
synthetic Q/K vectors used by the isolated attention-score pipeline. Q and K
are quantized to INT8 with per-tensor symmetric scales; V remains float32.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from check_tinyllama_setup import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT,
    choose_device,
    choose_dtype,
    require_packages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Q_rot, K_rot, and V tensors from TinyLlama.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id or local model path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Prompt text for extraction. Default: {DEFAULT_TEXT!r}",
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
        help="Q head index to inspect. Default: 0.",
    )
    parser.add_argument(
        "--kv-head",
        type=int,
        default=None,
        help="Optional KV head index. Default maps from Q head for grouped-query attention.",
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


def get_arg_or_kwarg(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def map_q_head_to_kv_head(q_head: int, num_q_heads: int, num_kv_heads: int) -> int:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot map Q head to KV head: num_q_heads={num_q_heads}, "
            f"num_kv_heads={num_kv_heads}"
        )
    group_size = num_q_heads // num_kv_heads
    return q_head // group_size


def add_qkv_capture_hook(attn_module: Any, captured: dict[str, Any]) -> Any:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    def capture_pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_states = get_arg_or_kwarg(args, kwargs, "hidden_states", 0)
        position_embeddings = get_arg_or_kwarg(args, kwargs, "position_embeddings", 1)

        if hidden_states is None:
            raise RuntimeError("Could not capture hidden_states from attention pre-hook")
        if position_embeddings is None:
            raise RuntimeError("Could not capture position_embeddings for RoPE")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)

        query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_rot, key_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        captured["q_rot"] = query_rot.detach().cpu()
        captured["k_rot"] = key_rot.detach().cpu()
        captured["v"] = value_states.detach().cpu()

    return attn_module.register_forward_pre_hook(capture_pre_hook, with_kwargs=True)


def causal_attention_reference(q_head: Any, k_head: Any, v_head: Any) -> tuple[Any, Any, Any]:
    import torch

    q_float = q_head.float()
    k_float = k_head.float()
    v_float = v_head.float()
    head_dim = q_float.shape[-1]
    seq_len = q_float.shape[0]

    scores = (q_float @ k_float.transpose(0, 1)) * (1.0 / math.sqrt(float(head_dim)))
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    attn_out = weights @ v_float
    return scores, weights, attn_out


def symmetric_int8_quantize(tensor: Any) -> tuple[Any, float]:
    import torch

    tensor_float = tensor.float()
    max_abs = tensor_float.abs().max()
    if max_abs.item() == 0.0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = max_abs / 127.0
    quantized = (tensor_float / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, float(scale.item())


def quantization_metrics(q_head: Any, k_head: Any) -> dict[str, Any]:
    import torch

    q_int8, q_scale = symmetric_int8_quantize(q_head)
    k_int8, k_scale = symmetric_int8_quantize(k_head)

    q_reconstructed = q_int8.float() * q_scale
    k_reconstructed = k_int8.float() * k_scale

    attn_scale = 1.0 / math.sqrt(float(q_head.shape[-1]))
    total_scale = q_scale * k_scale * attn_scale
    score_float = (q_head.float() @ k_head.float().transpose(0, 1)) * attn_scale
    score_dequant = (q_int8.float() @ k_int8.float().transpose(0, 1)) * total_scale
    score_abs_error = (score_dequant - score_float).abs()

    if not torch.isfinite(score_dequant).all():
        raise SystemExit("Quantized/dequantized score contains non-finite values")
    if not torch.isfinite(score_abs_error).all():
        raise SystemExit("Quantized score error contains non-finite values")

    return {
        "q_int8": q_int8,
        "k_int8": k_int8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "total_scale": total_scale,
        "q_recon_max_error": (q_reconstructed - q_head.float()).abs().max().item(),
        "k_recon_max_error": (k_reconstructed - k_head.float()).abs().max().item(),
        "score_max_error": score_abs_error.max().item(),
        "score_mean_error": score_abs_error.mean().item(),
    }


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
    inputs = {name: value.to(device) for name, value in inputs.items()}

    print(f"Input ids shape: {tuple(inputs['input_ids'].shape)}")
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

    batch_size, num_q_heads, seq_len, head_dim = q_rot.shape
    _, num_kv_heads, k_seq_len, k_head_dim = k_rot.shape
    _, v_num_kv_heads, v_seq_len, v_head_dim = v.shape

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
    scores, weights, attn_ref = causal_attention_reference(q_head, k_head, v_head)
    quant = quantization_metrics(q_head, k_head)

    sdpa_ref = torch.nn.functional.scaled_dot_product_attention(
        q_head.float().unsqueeze(0).unsqueeze(0),
        k_head.float().unsqueeze(0).unsqueeze(0),
        v_head.float().unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=True,
    ).squeeze(0).squeeze(0)
    sdpa_max_diff = (sdpa_ref - attn_ref).abs().max().item()

    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(f"Captured Q_rot shape: {tuple(q_rot.shape)}")
    print(f"Captured K_rot shape: {tuple(k_rot.shape)}")
    print(f"Captured V shape: {tuple(v.shape)}")
    print(f"Selected Q head: {args.head}")
    print(f"Mapped KV head: {kv_head}")
    print(f"Selected q_head shape: {tuple(q_head.shape)}")
    print(f"Selected k_head shape: {tuple(k_head.shape)}")
    print(f"Selected v_head shape: {tuple(v_head.shape)}")
    print(f"Reference score shape: {tuple(scores.shape)}")
    print(f"Reference weights shape: {tuple(weights.shape)}")
    print(f"Reference attn_out shape: {tuple(attn_ref.shape)}")
    print(f"SDPA max diff: {sdpa_max_diff:.8e}")
    print(f"Q int8 shape: {tuple(quant['q_int8'].shape)}")
    print(f"K int8 shape: {tuple(quant['k_int8'].shape)}")
    print(f"Q int8 range: [{int(quant['q_int8'].min())}, {int(quant['q_int8'].max())}]")
    print(f"K int8 range: [{int(quant['k_int8'].min())}, {int(quant['k_int8'].max())}]")
    print(f"q_scale: {quant['q_scale']:.12e}")
    print(f"k_scale: {quant['k_scale']:.12e}")
    print(f"total_scale: {quant['total_scale']:.12e}")
    print(f"Q reconstruction max error: {quant['q_recon_max_error']:.8e}")
    print(f"K reconstruction max error: {quant['k_recon_max_error']:.8e}")
    print(f"Score dequant max error: {quant['score_max_error']:.8e}")
    print(f"Score dequant mean error: {quant['score_mean_error']:.8e}")

    if batch_size != 1:
        raise SystemExit(f"Expected batch size 1, got {batch_size}")
    if seq_len != k_seq_len or seq_len != v_seq_len:
        raise SystemExit("Q/K/V sequence lengths do not match")
    if head_dim != 64 or k_head_dim != 64 or v_head_dim != 64:
        raise SystemExit(
            f"Expected head_dim 64, got Q={head_dim}, K={k_head_dim}, V={v_head_dim}"
        )
    if v_num_kv_heads != num_kv_heads:
        raise SystemExit("K and V KV-head counts do not match")
    if sdpa_max_diff > 1.0e-5:
        raise SystemExit(f"Manual attention reference disagrees with SDPA: {sdpa_max_diff}")

    print("TinyLlama Q/K/V extraction OK")
    print("TinyLlama Q/K quantization OK")


if __name__ == "__main__":
    main()
