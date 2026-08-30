"""Merge already-grounded prompt sets and restore unique sequential IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from agents.TeacherData.generate import _validate_cases


def _key(case: dict) -> tuple[str, tuple[str, ...]]:
    turns = tuple(
        re.sub(r"\s+", "", str(value)).lower()
        for value in (case.get("turns") or [case.get("prompt", "")])
    )
    return str(case.get("category", "")), turns


def merge(base: dict, supplement: dict, target_count: int | None = None) -> dict:
    cases: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    duplicates = []
    for source_name, payload in (("base", base), ("supplement", supplement)):
        for case in payload.get("cases", []):
            key = _key(case)
            if key in seen:
                duplicates.append({"source": source_name, "id": case.get("id")})
                continue
            seen.add(key)
            cases.append(copy.deepcopy(case))
    if target_count is not None:
        cases = cases[:target_count]
    for index, case in enumerate(cases, 1):
        case["id"] = f"case-{index:06d}"
    cases = _validate_cases({"cases": cases}, len(cases))
    generation = {
        **dict(base.get("generation") or {}),
        "merged_supplement": str(supplement.get("generation") or {}),
        "raw_case_count": len(base.get("cases", [])) + len(supplement.get("cases", [])),
        "unique_case_count": len(cases),
        "duplicates_removed": len(duplicates),
        "merge_sources": ["base", "supplement"],
        "category_counts": dict(Counter(case["category"] for case in cases)),
    }
    return {"schema_version": "teacher-prompts.v1", "generation": generation, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="合并去重已 grounding 的提示词集合")
    parser.add_argument("--base", required=True)
    parser.add_argument("--supplement", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", type=int, default=0)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    supplement = json.loads(Path(args.supplement).read_text(encoding="utf-8"))
    payload = merge(base, supplement, args.target_count or None)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "count": len(payload["cases"]),
        "category_counts": payload["generation"]["category_counts"],
        "duplicates_removed": payload["generation"]["duplicates_removed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
