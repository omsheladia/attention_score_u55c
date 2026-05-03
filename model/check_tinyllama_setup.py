"""
Track C Step 1: verify TinyLlama can be loaded and run locally.

This script does not interact with HLS, XRT, or the FPGA. It only confirms the
Python/PyTorch side can load TinyLlama and run a short forward pass, which is
the prerequisite for later Q/K/V extraction.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Any


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TEXT = "The cat sat on the mat."


def require_packages() -> None:
    missing = [
        package
        for package in ("torch", "transformers", "sentencepiece")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        print("Missing required Python packages:", ", ".join(missing), file=sys.stderr)
        print("Install them with:", file=sys.stderr)
        print("  pip install torch transformers sentencepiece", file=sys.stderr)
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load TinyLlama and run one short forward pass.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id or local model path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Prompt text for the setup forward pass. Default: {DEFAULT_TEXT!r}",
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


def choose_device(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise SystemExit("Requested --device cuda, but torch.cuda.is_available() is false")
    return requested


def choose_dtype(requested: str, device: str, torch_module: Any) -> Any:
    if requested == "auto":
        return torch_module.float16 if device == "cuda" else torch_module.float32
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[requested]


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
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    print(f"Input ids shape: {tuple(inputs['input_ids'].shape)}")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print("TinyLlama forward pass OK")


if __name__ == "__main__":
    main()
