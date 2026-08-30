"""Benchmark the production medical hybrid retrieval path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

from .dense import DenseDocumentIndex, DenseRetriever, EmbeddingClient
from .retriever import MedicalRetriever


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = (
    "高血压患者平时如何控制血压？",
    "糖尿病有哪些常见并发症？",
    "梅毒一般需要治疗多长时间？",
    "胃食管反流平时饮食要注意什么？",
    "儿童哮喘有哪些常见表现？",
    "甲状腺功能减退需要做哪些检查？",
    "偏头痛通常是什么原因引起的？",
    "脂肪肝可以通过生活方式改善吗？",
    "慢性肾病患者日常应注意什么？",
    "肺结核治疗后需要怎样复查？",
)


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[position]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "med_database" / "med_search.sqlite")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "med_database" / "medical_document_vectors_256.manifest.json",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8006")
    parser.add_argument("--output", type=Path, default=ROOT / "med_database" / "hybrid_retrieval_benchmark.json")
    args = parser.parse_args()

    init_started = time.perf_counter()
    index = DenseDocumentIndex(args.manifest)
    dense_warm_ms = index.warm()
    dense = DenseRetriever(EmbeddingClient(args.endpoint), index)
    retriever = MedicalRetriever(args.database, dense_retriever=dense, dense_top_k=30)
    init_ms = (time.perf_counter() - init_started) * 1000

    rows = []
    total_values: list[float] = []
    embedding_values: list[float] = []
    search_values: list[float] = []
    dense_wait_values: list[float] = []
    sparse_values: list[float] = []
    edge_values: list[float] = []
    fact_values: list[float] = []
    parallel_values: list[float] = []
    for query in DEFAULT_QUERIES:
        started = time.perf_counter()
        result = retriever.consult(query)
        total_ms = (time.perf_counter() - started) * 1000
        retrieval = result["retrieval"]
        total_values.append(total_ms)
        embedding_values.append(float(retrieval["embedding_ms"]))
        search_values.append(float(retrieval["dense_search_ms"]))
        dense_wait_values.append(float(retrieval["dense_wait_ms"]))
        sparse_values.append(float(retrieval["sparse_ms"]))
        edge_values.append(float(retrieval["edge_ms"]))
        fact_values.append(float(retrieval["fact_ms"]))
        parallel_values.append(float(retrieval["parallel_total_ms"]))
        rows.append(
            {
                "query": query,
                "total_ms": round(total_ms, 3),
                "embedding_ms": retrieval["embedding_ms"],
                "dense_search_ms": retrieval["dense_search_ms"],
                "dense_wait_ms": retrieval["dense_wait_ms"],
                "sparse_ms": retrieval["sparse_ms"],
                "edge_ms": retrieval["edge_ms"],
                "fact_ms": retrieval["fact_ms"],
                "parallel_total_ms": retrieval["parallel_total_ms"],
                "parallel": bool(retrieval.get("parallel")),
                "dense_used": bool(retrieval["dense_used"]),
                "status": result["status"],
                "intent": result["intent"],
                "documents": [
                    evidence["question"]
                    for evidence in result["evidence"]
                    if evidence["type"] == "document"
                ],
            }
        )

    report = {
        "format": "medical_hybrid_benchmark_v1",
        "query_format": "original text, no instruction prefix",
        "model": "qwen3-embedding-0.6b",
        "model_variant": "bf16",
        "dimensions": 256,
        "vector_rows": int(index.vectors.shape[0]),
        "mapped_document_ids": int(index.document_ids.shape[0]),
        "retriever_init_ms": round(init_ms, 3),
        "dense_warm_ms": round(dense_warm_ms, 3),
        "queries": len(rows),
        "dense_used_count": sum(bool(row["dense_used"]) for row in rows),
        "latency_ms": {
            "embedding": _summary(embedding_values),
            "dense_exact_search": _summary(search_values),
            "dense_wait_after_sparse": _summary(dense_wait_values),
            "sparse_fts": _summary(sparse_values),
            "graph_edges": _summary(edge_values),
            "structured_facts": _summary(fact_values),
            "parallel_retrieval_region": _summary(parallel_values),
            "end_to_end_consult": _summary(total_values),
        },
        "results": rows,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
