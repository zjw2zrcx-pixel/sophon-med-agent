#!/usr/bin/env python3
"""Derive a compact normalized MRL index and document-ID maps from 1024 vectors."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "med_database"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA / "medical_document_vectors_1024.npy")
    parser.add_argument(
        "--corpus", type=Path,
        default=DEFAULT_DATA / "embedding_handoff_bf16" / "corpus" / "questions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if not 32 <= args.dimensions <= 1024:
        parser.error("--dimensions must be in [32, 1024]")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")

    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.shape[1] < args.dimensions:
        raise ValueError(f"source shape cannot provide requested dimensions: {source.shape}")
    rows = source.shape[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / f"medical_document_vectors_{args.dimensions}.npy"
    offsets_path = args.output_dir / "medical_document_vector_offsets.npy"
    document_ids_path = args.output_dir / "medical_document_vector_doc_ids.npy"
    manifest_path = args.output_dir / f"medical_document_vectors_{args.dimensions}.manifest.json"
    temporary_vector = vector_path.with_name("." + vector_path.name + ".tmp")
    temporary_offsets = offsets_path.with_name("." + offsets_path.name + ".tmp")
    temporary_document_ids = document_ids_path.with_name("." + document_ids_path.name + ".tmp")

    target = np.lib.format.open_memmap(
        temporary_vector, mode="w+", dtype=np.float32, shape=(rows, args.dimensions)
    )
    for start in range(0, rows, args.chunk_rows):
        values = np.asarray(source[start : start + args.chunk_rows, : args.dimensions], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(norms <= 0):
            raise ValueError(f"invalid source vectors in rows {start}:{start + len(values)}")
        target[start : start + len(values)] = values / norms
    target.flush()
    del target

    offsets = [0]
    document_ids: list[int] = []
    with args.corpus.open("r", encoding="utf-8") as handle:
        for expected_row, line in enumerate(handle):
            record = json.loads(line)
            if record["row_index"] != expected_row:
                raise ValueError(f"non-contiguous corpus row at {expected_row}")
            ids = [int(value) for value in record["document_ids"]]
            if not ids:
                raise ValueError(f"corpus row {expected_row} has no document IDs")
            document_ids.extend(ids)
            offsets.append(len(document_ids))
    if len(offsets) != rows + 1:
        raise ValueError(f"corpus/vector row mismatch: {len(offsets) - 1} != {rows}")
    offsets_array = np.asarray(offsets, dtype=np.int64)
    ids_array = np.asarray(document_ids, dtype=np.int64)
    with temporary_offsets.open("wb") as handle:
        np.save(handle, offsets_array, allow_pickle=False)
    with temporary_document_ids.open("wb") as handle:
        np.save(handle, ids_array, allow_pickle=False)

    os.replace(temporary_vector, vector_path)
    os.replace(temporary_offsets, offsets_path)
    os.replace(temporary_document_ids, document_ids_path)
    manifest = {
        "format": "medical_dense_index_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "qwen3-embedding-0.6b",
        "model_variant": "bf16",
        "dimensions": args.dimensions,
        "normalization": "L2 after selecting the MRL prefix",
        "document_instruction": None,
        "rows": rows,
        "mapped_document_ids": len(document_ids),
        "source": {
            "file": str(args.source.resolve()),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "sha256": sha256_file(args.source),
        },
        "corpus": {"file": str(args.corpus.resolve()), "sha256": sha256_file(args.corpus)},
        "vectors": {"file": vector_path.name, "sha256": sha256_file(vector_path)},
        "offsets": {"file": offsets_path.name, "sha256": sha256_file(offsets_path)},
        "document_ids": {"file": document_ids_path.name, "sha256": sha256_file(document_ids_path)},
    }
    temporary_manifest = manifest_path.with_name("." + manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
