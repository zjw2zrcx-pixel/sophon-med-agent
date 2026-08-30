#!/usr/bin/env python3
"""Compare normalized Qwen3 embedding outputs from BF16 and W4BF16 bmodels."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run_isolated(model: Path, tokenizer: Path, text: str, device_id: int, dimensions: int) -> np.ndarray:
    """Run one model in a child process so two large engines never coexist."""
    command = [sys.executable, str(ROOT / "embed_sail.py"), text,
               "--bmodel", str(model), "--tokenizer", str(tokenizer),
               "--device-id", str(device_id), "--dimensions", str(dimensions)]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = completed.stdout.lstrip()
    decoded, _ = json.JSONDecoder().raw_decode(payload)
    return np.asarray(decoded["embedding"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="one or more texts to compare")
    parser.add_argument("--bf16", default="model/qwen3_embedding_bf16_seq512_bm1684x.bmodel")
    parser.add_argument("--w4", default="model/qwen3_embedding_w4bf16_seq512_bm1684x.bmodel")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dimensions", type=int, default=256)
    args = parser.parse_args()

    results = []
    for text in args.text:
        reference = run_isolated(Path(args.bf16), Path(args.tokenizer), text, args.device_id, args.dimensions)
        quantized = run_isolated(Path(args.w4), Path(args.tokenizer), text, args.device_id, args.dimensions)
        results.append({
            "text": text,
            "dimensions": args.dimensions,
            "cosine_similarity": float(np.dot(reference, quantized)),
            "l2_distance": float(np.linalg.norm(reference - quantized)),
            "max_abs_difference": float(np.max(np.abs(reference - quantized))),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
