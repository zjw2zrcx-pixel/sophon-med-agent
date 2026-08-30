"""Deterministic, leakage-aware train/benchmark prompt splitting."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())


def _grams(text: str) -> set[str]:
    value = _normalize(text)
    return {value[index:index + 2] for index in range(max(0, len(value) - 1))}


def _similar(left: str, right: str, threshold: float = 0.72) -> bool:
    a, b = _grams(left), _grams(right)
    if not a or not b:
        return _normalize(left) == _normalize(right)
    return len(a & b) / len(a | b) >= threshold


def _source_key(case: dict[str, Any]) -> str:
    source = case.get("medical_source") or {}
    answer_hash = str(source.get("answer_sha256", ""))
    if answer_hash:
        return "medical:" + answer_hash
    return ""


def _prompt_key(case: dict[str, Any]) -> str:
    return f"prompt:{case.get('category', '')}:{_normalize(str(case.get('prompt', '')))}"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _families(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    families: list[list[dict[str, Any]]] = []
    for case in cases:
        source_key = _source_key(case)
        paraphrase_family = str(case.get("paraphrase_family_id", "")).strip()
        prompt = str(case.get("prompt", ""))
        for family in families:
            head = family[0]
            head_paraphrase_family = str(
                head.get("paraphrase_family_id", "")
            ).strip()
            if paraphrase_family and paraphrase_family == head_paraphrase_family:
                family.append(case)
                break
            if source_key and source_key == _source_key(head):
                family.append(case)
                break
            if case.get("category") == head.get("category") and _similar(
                prompt, str(head.get("prompt", ""))
            ):
                family.append(case)
                break
        else:
            families.append([case])
    return families


def split_cases(
    cases: list[dict[str, Any]], benchmark_ratio: float = 0.2,
    seed: str = "teacher-split-v1",
    locked_source_splits: dict[str, str] | None = None,
    locked_prompt_splits: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if not 0 < benchmark_ratio < 1:
        raise ValueError("benchmark_ratio 必须在 0 和 1 之间")
    ids = [str(case.get("id", "")) for case in cases]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("case id 必须非空且唯一")
    by_category: dict[str, list[dict]] = {}
    for case in cases:
        by_category.setdefault(str(case.get("category", "")), []).append(case)

    locked_source_splits = locked_source_splits or {}
    locked_prompt_splits = locked_prompt_splits or {}
    assignment: dict[str, str] = {}
    family_rows = []
    for category, category_cases in sorted(by_category.items()):
        families = _families(category_cases)
        families.sort(key=lambda family: hashlib.sha256(
            f"{seed}:{category}:{_source_key(family[0])}:{family[0]['prompt']}".encode()
        ).hexdigest())
        target = max(1, round(len(category_cases) * benchmark_ratio))
        benchmark_count = 0
        for index, family in enumerate(families):
            remaining = sum(len(item) for item in families[index:])
            locked = {
                locked_source_splits[key]
                for item in family if (key := _source_key(item)) in locked_source_splits
            }
            locked.update(
                locked_prompt_splits[key]
                for item in family if (key := _prompt_key(item)) in locked_prompt_splits
            )
            if len(locked) > 1:
                raise ValueError("同一 family 在历史 split 中存在冲突")
            if locked:
                split = next(iter(locked))
                use_benchmark = split == "benchmark"
            else:
                use_benchmark = benchmark_count < target and (
                    benchmark_count + remaining <= target
                    or (target - benchmark_count) >= len(family)
                )
                split = "benchmark" if use_benchmark else "train"
            if use_benchmark:
                benchmark_count += len(family)
            family_id = hashlib.sha256(
                "|".join(sorted(str(item["id"]) for item in family)).encode()
            ).hexdigest()[:16]
            for case in family:
                assignment[str(case["id"])] = split
            family_rows.append({
                "family_id": family_id, "category": category, "split": split,
                "case_ids": [case["id"] for case in family],
                "paraphrase_family_id": str(
                    family[0].get("paraphrase_family_id", "")
                ) or None,
                "medical_source_key": _source_key(family[0]) or None,
            })

    train, benchmark = [], []
    for source_case in cases:
        case = copy.deepcopy(source_case)
        case["split"] = assignment[str(case["id"])]
        (benchmark if case["split"] == "benchmark" else train).append(case)
    manifest = {
        "schema_version": "teacher-split.v1", "seed": seed,
        "benchmark_ratio_requested": benchmark_ratio,
        "total": len(cases), "train": len(train), "benchmark": len(benchmark),
        "category_train": dict(Counter(case["category"] for case in train)),
        "category_benchmark": dict(Counter(case["category"] for case in benchmark)),
        "families": family_rows,
        "leakage_invariant": "one semantic/source family belongs to exactly one split",
        "source_cases_sha256": _canonical_hash(cases),
        "train_cases_sha256": _canonical_hash(train),
        "benchmark_cases_sha256": _canonical_hash(benchmark),
        "case_hashes": {
            str(case["id"]): _canonical_hash(case) for case in train + benchmark
        },
        "immutability": "create a new version instead of editing this split in place",
        "locked_medical_source_count": sum(
            bool(_source_key(case) in locked_source_splits) for case in cases
        ),
        "locked_prompt_count": sum(
            bool(_prompt_key(case) in locked_prompt_splits) for case in cases
        ),
    }
    return train, benchmark, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性拆分 train/benchmark 提示词")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark-ratio", type=float, default=0.2)
    parser.add_argument("--seed", default="teacher-split-v1")
    parser.add_argument(
        "--prior-split-dir", action="append", default=[],
        help="历史固定 split 根目录；同一医疗来源必须继承既有 train/benchmark 归属",
    )
    args = parser.parse_args()
    source_path = Path(args.prompts_file).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    locked_source_splits = {}
    locked_prompt_splits = {}
    for prior_value in args.prior_split_dir:
        prior = Path(prior_value).resolve()
        for split in ("train", "benchmark"):
            payload = json.loads((prior / split / "prompts.json").read_text("utf-8"))
            for case in payload.get("cases", []):
                key = _source_key(case)
                if key and key in locked_source_splits and locked_source_splits[key] != split:
                    raise ValueError(f"历史医疗来源 split 冲突: {key}")
                if key:
                    locked_source_splits[key] = split
                prompt_key = _prompt_key(case)
                if (prompt_key in locked_prompt_splits
                        and locked_prompt_splits[prompt_key] != split):
                    raise ValueError(f"历史提示词 split 冲突: {prompt_key}")
                locked_prompt_splits[prompt_key] = split
    train, benchmark, manifest = split_cases(
        source.get("cases", []), args.benchmark_ratio, args.seed,
        locked_source_splits, locked_prompt_splits,
    )
    output = Path(args.output_dir).resolve()
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "benchmark").mkdir(parents=True, exist_ok=True)
    base_generation = dict(source.get("generation") or {})
    for name, cases in (("train", train), ("benchmark", benchmark)):
        payload = {
            "schema_version": "teacher-prompts.v1",
            "generation": {
                **base_generation, "split": name,
                "split_source": str(source_path), "split_seed": args.seed,
            },
            "cases": cases,
        }
        (output / name / "prompts.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
