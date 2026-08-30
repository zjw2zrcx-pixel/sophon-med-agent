#!/usr/bin/env python3
"""Generate the 1024-dimensional BF16 TPU golden vectors and metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import urllib.request

import numpy as np
import transformers
from transformers import AutoTokenizer


EXPECTED_VARIANT = "bf16"
EXPECTED_ARTIFACT = "qwen3_embedding_bf16_seq512_bm1684x.bmodel"


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def embed(url: str, texts: list[str]) -> tuple[dict, np.ndarray]:
    payload = json.dumps({
        "model": "qwen3-embedding-0.6b",
        "input": texts,
        "dimensions": 1024,
        "encoding_format": "float",
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    vectors = np.asarray(
        [item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])],
        dtype=np.float32,
    )
    return result, vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8006")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/data/structure/Qwen3_embedding_0.6B/"
                     "qwen3_embedding_w4bf16_bm1684x_seq512/tokenizer"),
    )
    parser.add_argument(
        "--bmodel",
        type=Path,
        default=Path("/data/structure/Qwen3_embedding_0.6B/"
                     "qwen3_embedding_w4bf16_bm1684x_seq512/model/"
                     "qwen3_embedding_bf16_seq512_bm1684x.bmodel"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    cases_path = root / "golden" / "cases.jsonl"
    vectors_path = root / "golden" / "tpu_bf16_1024.npy"
    metadata_path = root / "golden" / "tpu_bf16_1024.metadata.json"
    cases = load_jsonl(cases_path)

    status = get_json(args.endpoint.rstrip("/") + "/status")
    if status.get("status") != "ready":
        raise RuntimeError(f"embedding endpoint is not ready: {status}")
    if status.get("variant") != EXPECTED_VARIANT:
        raise RuntimeError(f"expected BF16 endpoint, got: {status}")
    if status.get("artifact") != EXPECTED_ARTIFACT:
        raise RuntimeError(f"unexpected embedding artifact: {status}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), padding_side="left")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    case_metadata = []
    for case in cases:
        full_ids = tokenizer(case["text"], truncation=False)["input_ids"]
        truncated = tokenizer(
            case["text"], truncation=True, max_length=512, return_attention_mask=True
        )
        input_ids = truncated["input_ids"]
        case_metadata.append({
            **{key: value for key, value in case.items() if key != "text"},
            "utf8_bytes": len(case["text"].encode("utf-8")),
            "characters": len(case["text"]),
            "tokens_before_truncation": len(full_ids),
            "tokens_embedded": len(input_ids),
            "was_truncated": len(full_ids) > 512,
            "embedded_input_ids_sha256": hashlib.sha256(
                np.asarray(input_ids, dtype="<i4").tobytes()
            ).hexdigest(),
        })

    result, vectors = embed(args.endpoint, [case["text"] for case in cases])
    if result.get("model_variant") != EXPECTED_VARIANT:
        raise RuntimeError(f"response did not identify BF16: {result}")
    if vectors.shape != (len(cases), 1024):
        raise RuntimeError(f"unexpected vector shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("golden vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-6, rtol=0):
        raise RuntimeError(f"golden vectors are not normalized: {norms}")

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors, allow_pickle=False)
    for index, item in enumerate(case_metadata):
        item["vector_l2_norm"] = float(norms[index])
        item["vector_float32_le_sha256"] = hashlib.sha256(
            np.asarray(vectors[index], dtype="<f4").tobytes()
        ).hexdigest()

    similarities = {}
    by_id = {case["case_id"]: index for index, case in enumerate(cases)}
    for name, left, right in (
        ("syphilis_document_vs_paraphrase", "med_short_syphilis", "query_syphilis_paraphrase"),
        ("syphilis_document_vs_instructed_query", "med_short_syphilis", "query_instructed_medical"),
        ("syphilis_document_vs_unrelated_diet", "med_short_syphilis", "med_postoperative_diet"),
    ):
        similarities[name] = float(vectors[by_id[left]] @ vectors[by_id[right]])

    tokenizer_files = {}
    for path in sorted(args.tokenizer.iterdir()):
        if path.is_file():
            tokenizer_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    metadata = {
        "format": "qwen3_embedding_tpu_bf16_golden_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vectors": {
            "file": vectors_path.name,
            "shape": list(vectors.shape),
            "dtype": "float32",
            "byte_order": "little-endian",
            "sha256": sha256_file(vectors_path),
            "row_alignment": "golden/cases.jsonl case_index",
        },
        "model": {
            "api_model": result.get("model"),
            "variant": result.get("model_variant"),
            "artifact": result.get("model_artifact"),
            "artifact_bytes": args.bmodel.stat().st_size,
            "artifact_sha256": sha256_file(args.bmodel),
            "hidden_size": 1024,
            "sequence_bucket": 512,
        },
        "preprocessing": {
            "tokenizer": "the bundled tokenizer directory",
            "padding_side": "left",
            "truncation_side": tokenizer.truncation_side,
            "max_length": 512,
            "pooling": "last hidden-state token at index -1",
            "mrl_dimensions": 1024,
            "normalization": "L2 after selecting the first 1024 dimensions",
            "document_instruction": None,
        },
        "tokenizer_files": tokenizer_files,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "transformers": transformers.__version__,
            "sophon_sail": "3.11.0",
            "device": "BM1684X TPU 0",
        },
        "usage": result.get("usage"),
        "similarities": similarities,
        "cases": case_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "vectors": str(vectors_path),
        "metadata": str(metadata_path),
        "shape": list(vectors.shape),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "similarities": similarities,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
