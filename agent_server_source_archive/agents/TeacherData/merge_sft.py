"""Deterministically merge reviewed SFT decision projections."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path


def _rehash(row: dict) -> str:
    value = dict(row)
    value.pop("sample_sha256", None)
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _repair_single_missing_json_brace(row: dict) -> bool:
    """Repair only the proven historical `<tool>{...}</tool>` truncation."""
    output = str(row.get("output", "")).strip()
    if not (output.startswith("<tool>") and output.endswith("</tool>")):
        return False
    body = output[6:-7]
    try:
        json.loads(body)
        return False
    except json.JSONDecodeError:
        pass
    try:
        json.loads(body + "}")
    except json.JSONDecodeError:
        return False
    row["output"] = "<tool>" + body + "}</tool>"
    provenance = copy.deepcopy(
        row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    )
    provenance["format_repair"] = {
        "type": "append_single_missing_json_brace",
        "semantic_content_changed": False,
    }
    row["provenance"] = provenance
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="合并、去重并重排多个 SFT 决策集")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = [Path(value).resolve() for value in args.inputs]
    by_hash = {}
    manifests = []
    format_repairs = 0
    for source in sources:
        manifests.append(json.loads((source / "manifest.json").read_text("utf-8")))
        for line in (source / "sft_decisions.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_sample_sha256 = str(row.get("sample_sha256", ""))
            if not source_sample_sha256:
                source_sample_sha256 = _rehash(row)
            format_repairs += int(_repair_single_missing_json_brace(row))
            # Different generation batches historically reused ids such as
            # ``case-000001``.  Namespace them before merging, otherwise
            # samples from unrelated intents would be grouped into one case
            # and corrupt category/multi-turn statistics.
            original_case_id = str(row.get("case_id", ""))
            row["original_case_id"] = original_case_id
            row["case_id"] = f"{source.name}::{original_case_id}"
            tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
            if not isinstance(tags.get("intent_metadata"), dict):
                # Historical accepted SFT projections predate the explicit
                # provenance field.  They are treated as original independent
                # intents, while future paraphrase/ASR data must carry an
                # explicit non-independent flag from project_sft.
                tags = copy.deepcopy(tags)
                tags["intent_metadata"] = {
                    "schema_version": "teacher-intent-origin.v1",
                    "origin": "original_independent_legacy",
                    "independent_intent": True,
                    "semantic_family_id": (
                        f"legacy:{tags.get('source_run', 'unknown')}:{row.get('case_id', '')}"
                    ),
                    "dialogue_mode": (
                        "multi_turn" if str(tags.get("conversation_type")) == "multi_turn"
                        else "single_turn"
                    ),
                    "is_paraphrase": False,
                    "is_pronunciation_variant": False,
                }
                row["tags"] = tags
            # Preserve the first (highest-priority) source for an exact
            # decision collision.  The old implementation silently replaced
            # canonical rows with whichever duplicate source was listed last.
            by_hash.setdefault(source_sample_sha256, row)
    rows = sorted(by_hash.values(), key=lambda row: (
        str(row.get("case_id", "")), int(row.get("external_turn") or 0),
        int(row.get("agent_iteration") or 0), str(row.get("sample_sha256", "")),
    ))
    # Namespacing changes the serialized sample.  Keep the original digest as
    # provenance, then hash the actual emitted row so integrity checks remain
    # meaningful after merging.
    for row in rows:
        row["source_sample_sha256"] = str(row.get("sample_sha256", ""))
        row["sample_sha256"] = _rehash(row)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows]
    for filename in ("train.jsonl", "sft_decisions.jsonl"):
        with (output / filename).open("w", encoding="utf-8") as handle:
            handle.writelines(serialized)
    case_rows = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        entry = case_rows.setdefault(case_id, {
            "case_id": case_id,
            "original_case_id": row.get("original_case_id", case_id),
            "category": row.get("category", "unknown"),
            "tags": row.get("tags") or {},
            "sample_count": 0,
            "actions": Counter(),
            "external_turns": set(),
        })
        entry["sample_count"] += 1
        entry["actions"][str(row.get("action"))] += 1
        entry["external_turns"].add(int(row.get("external_turn") or 0))
    with (output / "case_index.jsonl").open("w", encoding="utf-8") as handle:
        for case_id in sorted(case_rows):
            entry = case_rows[case_id]
            entry["actions"] = dict(entry["actions"])
            entry["external_turns"] = sorted(entry["external_turns"])
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "agent-sft-merged.v2",
        "sources": [str(value) for value in sources],
        "source_manifests": manifests,
        "sample_count": len(rows),
        "case_count": len({row.get("case_id") for row in rows}),
        "actions": dict(Counter(str(row.get("action")) for row in rows)),
        "ordering": "case_id, external_turn, agent_iteration",
        "dedupe_key": "sample_sha256",
        "reasoning_removed": True,
        "format_repairs": format_repairs,
        "canonical_training_file": "train.jsonl",
        "case_index_file": "case_index.jsonl",
        "tag_contract": {
            "category": ["medical", "navigation", "general", "mixed"],
            "conversation_type": ["single_turn", "multi_turn"],
            "additional": ["difficulty", "conversation_turns", "scripted_turns",
                           "required_tools", "risk_tags", "teacher_model",
                           "prompt_model", "source_run", "medical_grounded",
                           "intent_metadata"],
        },
        "category_cases": dict(Counter(
            row.get("category", "unknown") for row in {
                value.get("case_id"): value for value in rows
            }.values()
        )),
        "category_samples": dict(Counter(
            row.get("category", "unknown") for row in rows
        )),
        "conversation_type_cases": dict(Counter(
            (row.get("tags") or {}).get("conversation_type", "unknown")
            for row in {value.get("case_id"): value for value in rows}.values()
        )),
        "source_run_cases": dict(Counter(
            (row.get("tags") or {}).get("source_run", "unknown")
            for row in {value.get("case_id"): value for value in rows}.values()
        )),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
