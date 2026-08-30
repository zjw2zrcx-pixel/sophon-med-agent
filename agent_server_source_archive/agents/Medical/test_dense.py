from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from agents.Medical.dense import (
    DenseDocumentIndex,
    DenseHit,
    DenseRetriever,
    reciprocal_rank_fusion,
)
from agents.Medical.retriever import MedicalRetriever


class _FakeClient:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def embed(self, text: str) -> np.ndarray:
        del text
        return self.vector


class _FailingDense:
    def search(self, query: str, top_k: int = 30):
        del query, top_k
        raise RuntimeError("endpoint down")


class DenseIndexTest(unittest.TestCase):
    def test_exact_search_expands_document_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "vectors.npy", np.asarray([[1, 0], [0, 1], [0.7, 0.7]], np.float32))
            np.save(root / "offsets.npy", np.asarray([0, 2, 3, 4], np.int64))
            np.save(root / "ids.npy", np.asarray([10, 11, 20, 30], np.int64))
            manifest = {
                "format": "medical_dense_index_v1",
                "model": "test-model",
                "model_variant": "bf16",
                "dimensions": 2,
                "rows": 3,
                "mapped_document_ids": 4,
                "vectors": {"file": "vectors.npy"},
                "offsets": {"file": "offsets.npy"},
                "document_ids": {"file": "ids.npy"},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            index = DenseDocumentIndex(
                root / "manifest.json", expected_model="test-model", expected_dimensions=2
            )
            result = DenseRetriever(_FakeClient(np.asarray([1, 0], np.float32)), index).search(
                "query", top_k=2
            )
            self.assertEqual([hit.document_id for hit in result.hits], [10, 11, 30])
            self.assertAlmostEqual(result.hits[0].score, 1.0, places=6)

    def test_rrf_rewards_overlap_and_deduplicates(self) -> None:
        dense = [DenseHit(3, 0.9, 1), DenseHit(2, 0.8, 2), DenseHit(3, 0.7, 3)]
        fused = reciprocal_rank_fusion([1, 2], dense)
        self.assertEqual(fused[0], 2)
        self.assertEqual(len(fused), len(set(fused)))


class HybridFallbackTest(unittest.TestCase):
    def test_dense_failure_falls_back_to_fts(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE documents(id INTEGER PRIMARY KEY, question TEXT, answer TEXT, source TEXT);
            CREATE VIRTUAL TABLE document_fts USING fts5(search_tokens, content='');
            INSERT INTO documents VALUES(1, '高血压如何控制血压', '控制盐摄入并规律监测。', 'fixture');
            INSERT INTO document_fts(rowid, search_tokens) VALUES(1, '高血 血压 压如 如何 何控 控制 制血');
            """
        )
        retriever = MedicalRetriever.__new__(MedicalRetriever)
        retriever._dense_retriever = _FailingDense()
        retriever._dense_top_k = 10
        evidence, metadata = retriever._document_evidence(
            connection, "高血压如何控制", (), "overview", limit=2
        )
        self.assertEqual(evidence[0]["source"], "fixture")
        self.assertEqual(metadata["fallback"], "sparse")
        self.assertFalse(metadata["dense_used"])

    def test_self_medication_document_is_blocked(self) -> None:
        self.assertTrue(MedicalRetriever._unsafe_document_question("高血压吃什么降压药最好"))
        self.assertFalse(MedicalRetriever._unsafe_document_question("高血压患者如何控制血压"))

    def test_dose_wording_is_medication_intent(self) -> None:
        retriever = MedicalRetriever.__new__(MedicalRetriever)
        self.assertEqual(retriever._detect_intent("头痛该吃几片什么药", ()), "medication")

    def test_ambiguous_symptom_does_not_unlock_medication(self) -> None:
        matches = (
            {"surface": "头痛", "label": "疾病", "match": "exact"},
            {"surface": "头痛", "label": "症状", "match": "exact"},
        )
        self.assertFalse(MedicalRetriever._medication_context_allowed(matches))
        self.assertTrue(
            MedicalRetriever._medication_context_allowed(
                (*matches, {"surface": "高血压", "label": "疾病", "match": "exact"})
            )
        )


if __name__ == "__main__":
    unittest.main()
