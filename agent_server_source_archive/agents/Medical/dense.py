"""Read-only dense retrieval for the medical encyclopedia corpus."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseHit:
    document_id: int
    score: float
    rank: int


@dataclass(frozen=True)
class DenseSearch:
    hits: Sequence[DenseHit]
    embedding_ms: float
    search_ms: float


class EmbeddingClient:
    """Minimal OpenAI-compatible embedding client with strict BF16 checks."""

    def __init__(
        self,
        endpoint: str,
        model: str = "qwen3-embedding-0.6b",
        dimensions: int = 256,
        timeout: float = 5.0,
        query_prefix: str = "",
        require_bf16: bool = True,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1/embeddings"):
            self.endpoint += "/v1/embeddings"
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.require_bf16 = require_bf16

    def embed(self, text: str) -> np.ndarray:
        payload = json.dumps(
            {
                "model": self.model,
                "input": self.query_prefix + text,
                "dimensions": self.dimensions,
                "encoding_format": "float",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"embedding endpoint HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"embedding endpoint unavailable: {exc}") from exc

        if body.get("model") != self.model:
            raise ValueError(f"embedding model mismatch: {body.get('model')!r}")
        if self.require_bf16 and body.get("model_variant") != "bf16":
            raise ValueError(f"embedding variant is not bf16: {body.get('model_variant')!r}")
        try:
            vector = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("embedding response has no valid vector") from exc
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"embedding dimensions mismatch: {vector.shape}, expected {(self.dimensions,)}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if not 0.5 < norm < 1.5:
            raise ValueError(f"embedding has invalid L2 norm: {norm}")
        return vector / norm


class DenseDocumentIndex:
    """Exact cosine/IP index backed by NumPy read-only memory maps."""

    def __init__(
        self,
        manifest_path: str | Path,
        expected_model: str = "qwen3-embedding-0.6b",
        expected_dimensions: int = 256,
        max_concurrent_scans: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: Mapping[str, Any] = json.load(handle)
        if self.manifest.get("format") != "medical_dense_index_v1":
            raise ValueError("unsupported medical dense index format")
        if self.manifest.get("model") != expected_model:
            raise ValueError("dense index model does not match runtime model")
        if self.manifest.get("model_variant") != "bf16":
            raise ValueError("dense index was not produced from the BF16 model")
        if self.manifest.get("dimensions") != expected_dimensions:
            raise ValueError("dense index dimensions do not match runtime dimensions")

        base = self.manifest_path.parent
        self.vectors = np.load(base / self.manifest["vectors"]["file"], mmap_mode="r")
        self.offsets = np.load(base / self.manifest["offsets"]["file"], mmap_mode="r")
        self.document_ids = np.load(
            base / self.manifest["document_ids"]["file"], mmap_mode="r"
        )
        rows = int(self.manifest["rows"])
        mapped_ids = int(self.manifest["mapped_document_ids"])
        if self.vectors.shape != (rows, expected_dimensions):
            raise ValueError(f"invalid dense vector shape: {self.vectors.shape}")
        if self.vectors.dtype != np.float32:
            raise ValueError(f"invalid dense vector dtype: {self.vectors.dtype}")
        if self.offsets.shape != (rows + 1,) or self.offsets.dtype != np.int64:
            raise ValueError("invalid dense offset mapping")
        if self.document_ids.shape != (mapped_ids,) or self.document_ids.dtype != np.int64:
            raise ValueError("invalid dense document-id mapping")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != mapped_ids:
            raise ValueError("dense offset mapping is incomplete")
        self.dimensions = expected_dimensions
        self._scan_slots = threading.BoundedSemaphore(max(1, max_concurrent_scans))

    def warm(self) -> float:
        """Fault vector pages in and initialise the matrix multiply backend."""
        started = time.perf_counter()
        probe = np.zeros(self.dimensions, dtype=np.float32)
        probe[0] = 1.0
        self.search(probe, top_k=1)
        return (time.perf_counter() - started) * 1000

    def search(self, query_vector: np.ndarray, top_k: int = 30) -> List[DenseHit]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
            raise ValueError("invalid dense query vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("dense query vector has zero norm")
        vector = vector / norm
        k = min(max(1, int(top_k)), self.vectors.shape[0])
        with self._scan_slots:
            scores = self.vectors @ vector
            rows = np.argpartition(scores, -k)[-k:]
            rows = rows[np.argsort(scores[rows])[::-1]]

        hits: List[DenseHit] = []
        seen: set[int] = set()
        rank = 0
        for row_index in rows:
            rank += 1
            start = int(self.offsets[row_index])
            stop = int(self.offsets[row_index + 1])
            for document_id in self.document_ids[start:stop]:
                value = int(document_id)
                if value in seen:
                    continue
                seen.add(value)
                hits.append(DenseHit(value, float(scores[row_index]), rank))
        return hits


class DenseRetriever:
    """Compose query encoding and exact index search with latency metadata."""

    def __init__(self, client: EmbeddingClient, index: DenseDocumentIndex) -> None:
        self.client = client
        self.index = index

    def search(self, query: str, top_k: int = 30) -> DenseSearch:
        started = time.perf_counter()
        vector = self.client.embed(query)
        encoded = time.perf_counter()
        hits = self.index.search(vector, top_k=top_k)
        finished = time.perf_counter()
        return DenseSearch(
            hits=hits,
            embedding_ms=(encoded - started) * 1000,
            search_ms=(finished - encoded) * 1000,
        )


def reciprocal_rank_fusion(
    sparse_ids: Sequence[int],
    dense_hits: Sequence[DenseHit],
    constant: int = 60,
    dense_weight: float = 1.25,
) -> List[int]:
    """Fuse ranks with a modest dense preference over generic bigram matches."""
    scores: Dict[int, float] = {}
    for rank, document_id in enumerate(sparse_ids, start=1):
        scores[int(document_id)] = scores.get(int(document_id), 0.0) + 1.0 / (
            constant + rank
        )
    seen_dense: set[int] = set()
    for hit in dense_hits:
        if hit.document_id in seen_dense:
            continue
        seen_dense.add(hit.document_id)
        scores[hit.document_id] = scores.get(hit.document_id, 0.0) + dense_weight / (
            constant + hit.rank
        )
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))


__all__ = [
    "DenseDocumentIndex",
    "DenseHit",
    "DenseRetriever",
    "DenseSearch",
    "EmbeddingClient",
    "reciprocal_rank_fusion",
]
