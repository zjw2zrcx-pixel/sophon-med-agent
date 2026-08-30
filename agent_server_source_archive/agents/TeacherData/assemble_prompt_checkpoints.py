"""Assemble prompt checkpoints, removing cross-batch duplicates before grounding."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from agents.TeacherData.generate import _attach_case_support, _validate_cases
from agents.TeacherData.medical_prompts import MedicalPromptSampler


def _key(case: dict) -> tuple[str, str]:
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(case.get("prompt", "")).lower())
    return str(case.get("category", "")), text


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 DeepSeek 提示词 checkpoint 并全局去重")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--medical-database", default="/data/structure/med_database/med_search.sqlite")
    parser.add_argument("--medical-multiturn-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    raw = []
    models = Counter()
    for path in sorted(Path(args.checkpoint_dir).glob("batch_*.json")):
        payload = json.loads(path.read_text("utf-8"))
        models[str(payload.get("model", "unknown"))] += len(payload.get("cases", []))
        raw.extend(payload.get("cases", []))
    unique = []
    seen = set()
    duplicates = []
    for case in raw:
        # Flash occasionally omits ``turns`` or echoes the whole dialogue in
        # ``prompt`` while also returning split turns.  The executable
        # contract is that prompt is the first user input; repair this local
        # structural noise before the authoritative validator runs.  No
        # medical content is synthesized here.
        prompt = str(case.get("prompt", "")).strip()
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            case["turns"] = [prompt]
        elif len(turns) == 1:
            case["turns"] = [prompt]
        elif str(turns[0]).strip() != prompt:
            case["prompt"] = str(turns[0]).strip()
        key = _key(case)
        if key in seen:
            duplicates.append({"category": key[0], "prompt": case.get("prompt")})
            continue
        seen.add(key)
        unique.append(case)
    for index, case in enumerate(unique, 1):
        case["id"] = f"v3-{index:06d}"
    sampler = MedicalPromptSampler(Path(args.medical_database), seed=args.seed)
    sampler.ground_cases(unique, multiturn_ratio=args.medical_multiturn_ratio)
    _attach_case_support(unique)
    unique = _validate_cases({"cases": unique}, len(unique))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "teacher-prompts.v1",
        "generation": {
            "assembled_from": str(Path(args.checkpoint_dir).resolve()),
            "raw_count": len(raw), "unique_count": len(unique),
            "duplicates_removed": len(duplicates), "duplicate_rows": duplicates,
            "category_counts": dict(Counter(row["category"] for row in unique)),
            "model_counts": dict(models), "medical_multiturn_ratio": args.medical_multiturn_ratio,
            "seed": args.seed,
        },
        "cases": unique,
    }, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
