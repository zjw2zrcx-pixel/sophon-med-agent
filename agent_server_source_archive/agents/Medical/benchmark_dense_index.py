#!/usr/bin/env python3
"""Benchmark exact cosine top-k over the normalized medical dense index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "med_database"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_topk(vectors: np.ndarray, query: np.ndarray, k: int):
    score_started = time.perf_counter()
    scores = np.asarray(vectors @ query, dtype=np.float32)
    score_seconds = time.perf_counter() - score_started

    topk_started = time.perf_counter()
    candidate = np.argpartition(scores, -k)[-k:]
    candidate = candidate[np.argsort(scores[candidate])[::-1]]
    topk_seconds = time.perf_counter() - topk_started
    return candidate, scores[candidate], score_seconds, topk_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=DATA / "medical_document_vectors_256.npy")
    parser.add_argument("--offsets", type=Path, default=DATA / "medical_document_vector_offsets.npy")
    parser.add_argument("--document-ids", type=Path, default=DATA / "medical_document_vector_doc_ids.npy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    vectors = np.load(args.vectors, mmap_mode="r", allow_pickle=False)
    offsets = np.load(args.offsets, mmap_mode="r", allow_pickle=False)
    document_ids = np.load(args.document_ids, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or offsets.shape != (vectors.shape[0] + 1,):
        raise ValueError("dense index shape mismatch")
    if not 1 <= args.top_k <= vectors.shape[0]:
        parser.error("--top-k is outside the vector row range")

    rng = random.Random(args.seed)
    query_rows = rng.sample(range(vectors.shape[0]), args.warmups + args.queries)
    score_times: list[float] = []
    topk_times: list[float] = []
    map_times: list[float] = []
    total_times: list[float] = []
    self_recall = 0
    expanded_counts: list[int] = []

    for serial, row in enumerate(query_rows):
        query = np.asarray(vectors[row], dtype=np.float32)
        total_started = time.perf_counter()
        candidate, scores, score_seconds, topk_seconds = exact_topk(
            vectors, query, args.top_k
        )
        map_started = time.perf_counter()
        expanded = []
        for vector_row in candidate:
            start = int(offsets[vector_row])
            end = int(offsets[vector_row + 1])
            expanded.extend(int(value) for value in document_ids[start:end])
        map_seconds = time.perf_counter() - map_started
        total_seconds = time.perf_counter() - total_started
        if serial < args.warmups:
            continue
        self_recall += int(row in candidate)
        expanded_counts.append(len(expanded))
        score_times.append(score_seconds)
        topk_times.append(topk_seconds)
        map_times.append(map_seconds)
        total_times.append(total_seconds)

    def stats(values: list[float]) -> dict[str, float]:
        milliseconds = [value * 1000.0 for value in values]
        return {
            "min_ms": min(milliseconds),
            "p50_ms": statistics.median(milliseconds),
            "p95_ms": percentile(milliseconds, 0.95),
            "max_ms": max(milliseconds),
            "mean_ms": statistics.mean(milliseconds),
        }

    report = {
        "strategy": "exact normalized inner product + argpartition + local sort",
        "vectors": str(args.vectors.resolve()),
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "top_k": args.top_k,
        "measured_queries": args.queries,
        "warmup_queries": args.warmups,
        "seed": args.seed,
        "score": stats(score_times),
        "topk_selection": stats(topk_times),
        "document_id_expansion": stats(map_times),
        "search_total": stats(total_times),
        "self_recall_at_k": self_recall / args.queries,
        "expanded_document_ids_mean": statistics.mean(expanded_counts),
        "score_buffer_bytes": vectors.shape[0] * np.dtype(np.float32).itemsize,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
