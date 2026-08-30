#!/usr/bin/env python3
"""Compare Torch vectors against the TPU BF16 golden vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--golden", type=Path, default=root / "golden" / "tpu_bf16_1024.npy")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=root / "golden" / "cases.jsonl")
    parser.add_argument("--minimum-cosine", type=float, default=0.995)
    args = parser.parse_args()

    golden = np.load(args.golden, allow_pickle=False).astype(np.float32, copy=False)
    candidate = np.load(args.candidate, allow_pickle=False).astype(np.float32, copy=False)
    if golden.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: golden={golden.shape}, candidate={candidate.shape}")
    with args.cases.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if len(cases) != golden.shape[0]:
        raise SystemExit("case/vector row count mismatch")

    golden_norm = np.linalg.norm(golden, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    cosine = np.sum(golden * candidate, axis=1) / (golden_norm * candidate_norm)
    delta = golden - candidate
    rows = []
    for index, case in enumerate(cases):
        rows.append({
            "case_index": index,
            "case_id": case["case_id"],
            "cosine_similarity": float(cosine[index]),
            "l2_distance": float(np.linalg.norm(delta[index])),
            "max_abs_difference": float(np.max(np.abs(delta[index]))),
            "mean_abs_difference": float(np.mean(np.abs(delta[index]))),
            "golden_norm": float(golden_norm[index]),
            "candidate_norm": float(candidate_norm[index]),
            "pass": bool(cosine[index] >= args.minimum_cosine),
        })
    report = {
        "shape": list(golden.shape),
        "minimum_cosine_required": args.minimum_cosine,
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "cosine_max": float(cosine.max()),
        "all_pass": bool(np.all(cosine >= args.minimum_cosine)),
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
