#!/usr/bin/env python3
"""Export the medical document-question corpus for remote embedding generation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / "med_database" / "med_search.sqlite"
DEFAULT_OUTPUT = ROOT / "med_database" / "embedding_handoff_bf16"


def _sha256(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def export_questions(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    """Write one row per distinct question, ordered by its first document ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = duplicate_documents = total_characters = 0
    first_document_id = last_document_id = None
    digest = hashlib.sha256()
    query = """
        SELECT question, MIN(id) AS first_id, GROUP_CONCAT(id) AS document_ids,
               COUNT(*) AS occurrence_count
        FROM documents
        GROUP BY question
        ORDER BY first_id
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row_index, row in enumerate(connection.execute(query)):
            question = str(row["question"])
            document_ids = [int(value) for value in str(row["document_ids"]).split(",")]
            document_ids.sort()
            record = {
                "row_index": row_index,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "question": question,
                "document_ids": document_ids,
            }
            encoded = _json_line(record).encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1
            duplicate_documents += len(document_ids) - 1
            total_characters += len(question)
            first_document_id = document_ids[0] if first_document_id is None else first_document_id
            last_document_id = max(document_ids) if last_document_id is None else max(last_document_id, *document_ids)
    return {
        "rows": count,
        "duplicate_document_rows_mapped": duplicate_documents,
        "question_characters": total_characters,
        "first_document_id": first_document_id,
        "last_document_id": last_document_id,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _document_questions(connection: sqlite3.Connection, ids: Iterable[int]) -> dict[int, str]:
    values = list(ids)
    placeholders = ",".join("?" for _ in values)
    rows = connection.execute(
        f"SELECT id, question FROM documents WHERE id IN ({placeholders})", values
    )
    return {int(row["id"]): str(row["question"]) for row in rows}


def export_golden_cases(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    source_cases = [
        ("med_short_syphilis", "short medical question", "document", 2),
        ("med_drug_dosage", "drug and dosage intent", "document", 1),
        ("med_postoperative_diet", "postoperative diet", "document", 3),
        ("med_colloquial_symptom", "colloquial symptom question", "document", 8),
        ("med_mixed_ascii", "Chinese and ASCII medical acronym", "document", 82),
        ("med_multi_question", "multiple source paraphrases joined by semicolons", "document", 93),
        ("med_long_source", "long source question near tokenizer boundary", "document", 220055),
        ("med_multi_temporal", "long multi-question COVID title", "document", 273924),
    ]
    questions = _document_questions(connection, (item[3] for item in source_cases))
    cases: list[dict[str, Any]] = []
    for case_id, description, role, document_id in source_cases:
        cases.append({
            "case_id": case_id,
            "description": description,
            "role": role,
            "source_document_id": document_id,
            "text": questions[document_id],
        })

    cases.extend([
        {
            "case_id": "query_syphilis_paraphrase",
            "description": "semantic paraphrase of med_short_syphilis",
            "role": "query",
            "source_document_id": None,
            "text": "晚期梅毒一般需要治疗多长时间，是否能治愈？",
        },
        {
            "case_id": "query_colloquial_abdominal",
            "description": "colloquial symptom query",
            "role": "query",
            "source_document_id": None,
            "text": "肚子疼还拉肚子两天了，应该怎么办？",
        },
        {
            "case_id": "query_instructed_medical",
            "description": "provisional instructed query format",
            "role": "query",
            "source_document_id": None,
            "text": (
                "Instruct: Given a medical question, retrieve relevant passages that answer "
                "the question\nQuery: 三期梅毒一般需要治疗多久？"
            ),
        },
        {
            "case_id": "mixed_numeric_units",
            "description": "Chinese, Latin letters, decimal numbers, percent and units",
            "role": "synthetic",
            "source_document_id": None,
            "text": "患者HbA1c为8.2%，空腹血糖9.6 mmol/L，需要做哪些复查？",
        },
        {
            "case_id": "over_512_truncation",
            "description": "synthetic input deliberately longer than the 512-token bucket",
            "role": "synthetic",
            "source_document_id": None,
            "text": "患者反复出现发热、咳嗽、胸闷和乏力，需要结合病史、体格检查与检验结果综合判断。" * 80,
        },
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, case in enumerate(cases):
            record = {
                "case_index": index,
                **case,
                "text_sha256": hashlib.sha256(case["text"].encode("utf-8")).hexdigest(),
            }
            encoded = _json_line(record).encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return {"cases": len(cases), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    database = args.database.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not database.is_file():
        parser.error(f"database not found: {database}")
    output.mkdir(parents=True, exist_ok=True)

    with _connect_read_only(database) as connection:
        database_meta = dict(connection.execute("SELECT key, value FROM meta"))
        corpus = export_questions(connection, output / "corpus" / "questions.jsonl")
        golden_cases = export_golden_cases(connection, output / "golden" / "cases.jsonl")

    manifest = {
        "format": "medical_embedding_handoff_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_database": {
            "filename": database.name,
            "bytes": database.stat().st_size,
            "sha256": _sha256(database),
            "schema_version": database_meta.get("schema_version"),
            "built_at_utc": database_meta.get("built_at_utc"),
            "documents_count": int(database_meta.get("documents_count", 0)),
            "documents_policy": database_meta.get("documents_policy"),
        },
        "corpus": {
            "file": "corpus/questions.jsonl",
            "record_contract": {
                "row_index": "zero-based vector row; contiguous and stable for this export",
                "question_sha256": "SHA-256 of UTF-8 question text",
                "question": "exact parsed encyclopedia question text to embed",
                "document_ids": "one or more med_search.sqlite documents represented by this vector",
            },
            "ordering": "ascending MIN(documents.id), one row per distinct question",
            **corpus,
        },
        "golden_cases": {"file": "golden/cases.jsonl", **golden_cases},
    }
    manifest_path = output / "corpus" / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
