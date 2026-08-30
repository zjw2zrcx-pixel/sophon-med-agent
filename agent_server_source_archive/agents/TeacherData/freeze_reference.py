"""Freeze preflight-approved prompt sets into a content-addressed reference."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def freeze(sources: list[Path], output_dir: Path, reference_id: str) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"冻结目录已存在且非空，拒绝覆盖: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    source_rows = []
    for source_index, source_path in enumerate(sources, 1):
        source_path = Path(source_path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_tag = f"s{source_index}"
        source_rows.append({
            "tag": source_tag,
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "case_count": len(payload.get("cases", [])),
            "generation": payload.get("generation", {}),
        })
        for source_case in payload.get("cases", []):
            case = copy.deepcopy(source_case)
            original_id = str(case.get("id", ""))
            case["id"] = f"{reference_id}-{source_tag}-{original_id}"
            case["reference_provenance"] = {
                "source_tag": source_tag,
                "source_case_id": original_id,
                "source_path": str(source_path),
            }
            case["case_sha256"] = _sha({
                key: value for key, value in case.items() if key != "case_sha256"
            })
            cases.append(case)
    signatures = [tuple(case.get("turns") or [case.get("prompt", "")]) for case in cases]
    if len(signatures) != len(set(signatures)):
        raise ValueError("冻结集合包含重复对话文本")
    prompts_payload = {
        "schema_version": "teacher-prompts.v1",
        "generation": {
            "frozen": True,
            "reference_id": reference_id,
            "source_count": len(sources),
            "case_count": len(cases),
            "mutation_policy": "immutable; create a new reference version to change cases",
        },
        "cases": cases,
    }
    prompts_sha = _sha(prompts_payload)
    (output_dir / "prompts.json").write_text(
        json.dumps(prompts_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    categories = Counter(str(case.get("category", "")) for case in cases)
    manifest = {
        "schema_version": "teacher-reference.v1",
        "reference_id": reference_id,
        "immutable": True,
        "prompts_sha256": prompts_sha,
        "case_count": len(cases),
        "category_counts": dict(categories),
        "sources": source_rows,
        "cases": [
            {
                "id": case["id"],
                "category": case.get("category"),
                "case_sha256": case["case_sha256"],
                "required_tools": (case.get("expected") or {}).get("required_tools", []),
                "navigation_target": (case.get("expected") or {}).get("navigation_target"),
                "medical_source": case.get("medical_source"),
            }
            for case in cases
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结确定的提示词回归参照集")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-id", required=True)
    args = parser.parse_args()
    freeze([Path(path) for path in args.source], Path(args.output_dir), args.reference_id)


if __name__ == "__main__":
    main()
