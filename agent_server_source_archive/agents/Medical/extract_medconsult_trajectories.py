"""Extract compact, deduplicated medical_consult records from benchmark traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _short(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _iter_json_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield line_number, value
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    records = value if isinstance(value, list) else [value]
    for index, record in enumerate(records, 1):
        if isinstance(record, dict):
            yield index, record


def _tool_calls(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if record.get("name") in {"medical_consult", "med_consult"}:
        yield record
    for key in ("tool_call_records", "tool_calls"):
        calls = record.get(key)
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict) and call.get("name") in {
                    "medical_consult",
                    "med_consult",
                }:
                    yield call


def _parse_result(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"unparsed": _short(value, 500)}
    return parsed if isinstance(parsed, dict) else {"unparsed": _short(value, 500)}


def _compact_result(result: dict[str, Any], text_limit: int) -> dict[str, Any]:
    compact: dict[str, Any] = {
        key: result[key]
        for key in (
            "status",
            "intent",
            "positive_symptoms",
            "negative_symptoms",
            "normalized_terms",
            "red_flags",
            "departments",
            "questions",
            "message",
        )
        if key in result
    }
    compact["associations"] = [
        {
            key: item.get(key)
            for key in ("matched", "relation", "related", "related_type", "direction")
            if item.get(key) not in (None, "")
        }
        for item in result.get("associations", [])
        if isinstance(item, dict)
    ]
    evidence = []
    for item in result.get("evidence", []):
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                key: _short(item[key], text_limit) if key == "text" else item[key]
                for key in ("type", "subject", "aspect", "question", "text", "source")
                if key in item
            }
        )
    compact["evidence"] = evidence
    retrieval = result.get("retrieval")
    if isinstance(retrieval, dict):
        compact["retrieval"] = {
            key: retrieval[key]
            for key in (
                "mode",
                "reason",
                "dense_enabled",
                "dense_used",
                "sparse_candidates",
                "dense_candidates",
            )
            if key in retrieval
        }
    if "unparsed" in result:
        compact["unparsed"] = result["unparsed"]
    return compact


def extract(paths: Iterable[Path], text_limit: int = 320) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for root in paths:
        candidates = (
            [root]
            if root.is_file()
            else sorted(
                path
                for path in root.rglob("*")
                if path.name in {"tool_calls.jsonl", "runs.jsonl", "runs.json"}
            )
        )
        for path in candidates:
            for line_number, record in _iter_json_records(path):
                for call in _tool_calls(record):
                    params = call.get("params") or call.get("arguments") or {}
                    query = str(params.get("query", "") if isinstance(params, dict) else "").strip()
                    result = _parse_result(call.get("result"))
                    compact = _compact_result(result, text_limit)
                    signature_payload = json.dumps(
                        {"query": query, "result": compact},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:16]
                    occurrence = {
                        "file": str(path),
                        "line": line_number,
                        "scenario_id": call.get("scenario_id") or record.get("scenario_id"),
                        "model": call.get("model") or record.get("model"),
                        "category": call.get("category") or record.get("category"),
                    }
                    item = grouped.setdefault(
                        signature,
                        {
                            "id": signature,
                            "query": query,
                            "result": compact,
                            "occurrences": [],
                        },
                    )
                    if occurrence not in item["occurrences"]:
                        item["occurrences"].append(occurrence)
    output = sorted(grouped.values(), key=lambda item: (item["query"], item["id"]))
    for item in output:
        item["occurrence_count"] = len(item["occurrences"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Trace files or directories")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--text-limit", type=int, default=320)
    args = parser.parse_args()
    records = extract(args.paths, max(80, args.text_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
