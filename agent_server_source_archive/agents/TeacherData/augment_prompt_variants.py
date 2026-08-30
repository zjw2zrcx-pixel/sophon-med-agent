"""Curated navigation paraphrase families for the training split only.

These are intentionally hand-written rather than LLM-generated.  A family
keeps one intent/target while varying politeness, question form, ellipsis and
ASR-like wording.  ``split_dataset`` keeps each family on one side of the
train/benchmark boundary to prevent paraphrase leakage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from agents.MCP.tools.navigate import HOSPITAL_LOCATIONS, LOCATION_ALIASES


VARIANTS: dict[str, tuple[str, ...]] = {
    "挂号处": (
        "请带我去挂号处。",
        "挂号处怎么走？",
        "请问挂号处怎么去？",
        "麻烦带我到挂号的地方。",
        "我想挂号，应该往哪边走？",
        "挂号窗口在哪儿？",
        "从这里到挂号处怎么走？",
        "能指引我去挂号处吗？",
        "我要先挂号，请带我过去。",
        "挂号处在几楼，怎么走？",
    ),
    "门诊大厅": (
        "请带我去门诊大厅。",
        "门诊大厅往哪边走？",
        "请问门诊大厅怎么去？",
        "我第一次来，门诊大厅在哪儿？",
        "麻烦指引我到门诊大厅。",
        "从这里去门诊大厅怎么走？",
    ),
    "收费处": (
        "请带我去收费处。",
        "收费处怎么走？",
        "我需要缴费，收费处在哪？",
        "请问到收费窗口怎么去？",
        "麻烦带我到收费处。",
        "从这儿去收费处应该怎么走？",
    ),
    "服务台": (
        "请带我去服务台。",
        "服务台在哪边？",
        "请问服务台怎么走？",
        "我想问点事情，服务台怎么去？",
        "麻烦指引我到服务台。",
        "能带我去服务台吗？",
    ),
    "药房": (
        "请带我去药房。",
        "药房怎么走？",
        "取药的药房在哪儿？",
        "请问药房要往哪边走？",
        "麻烦带我到药房。",
    ),
    "急诊科": (
        "请带我去急诊科。",
        "急诊怎么走？",
        "请问急诊科在哪里？",
        "我需要去急诊，麻烦带我过去。",
        "从这里到急诊科怎么走？",
        "能马上指引我去急诊科吗？",
    ),
}


def _safe_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value)


def build_cases(*, selected_targets: list[str] | None = None) -> list[dict[str, Any]]:
    targets = selected_targets or list(VARIANTS)
    unknown = [target for target in targets if target not in VARIANTS]
    if unknown:
        raise ValueError(f"没有人工变体定义的目的地: {unknown}")
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for target in targets:
        if target not in HOSPITAL_LOCATIONS:
            raise ValueError(f"目的地不在导航注册表中: {target}")
        family_id = f"nav-paraphrase-{_safe_component(target)}-v1"
        aliases = sorted(
            alias for alias, canonical in LOCATION_ALIASES.items()
            if canonical == target
        )
        for index, prompt in enumerate(VARIANTS[target], 1):
            normalized = re.sub(r"\s+", "", prompt).lower()
            key = (target, normalized)
            if key in seen:
                continue
            seen.add(key)
            case_id = f"aug-nav-{_safe_component(target)}-{index:02d}"
            cases.append({
                "id": case_id,
                "category": "navigation",
                "prompt": prompt,
                "turns": [prompt],
                "difficulty": "medium",
                "expected": {
                    "required_tools": ["navigate"],
                    "forbidden_tools": ["medical_consult"],
                    "navigation_target": target,
                    "notes": "同一导航意图的人工口语变体；必须调用一次 navigate。",
                },
                "risk_tags": ["navigation_paraphrase"],
                "paraphrase_family_id": family_id,
                "augmentation_source": "manual_navigation_variants_v1",
                "support_data": {
                    "schema_version": "teacher-case-support.v1",
                    "category": "navigation",
                    "required_tools": ["navigate"],
                    "forbidden_tools": ["medical_consult"],
                    "navigation": {
                        "canonical_target": target,
                        "registered": True,
                        "known_aliases": aliases,
                        "mentioned_distractors": [],
                    },
                },
            })
    return cases


def merge_cases(source: dict[str, Any], additions: list[dict[str, Any]]) -> dict[str, Any]:
    existing = list(source.get("cases") or [])
    seen = {
        (str(case.get("category", "")), re.sub(r"\s+", "", str(case.get("prompt", ""))).lower())
        for case in existing
    }
    merged = existing[:]
    for case in additions:
        key = (case["category"], re.sub(r"\s+", "", case["prompt"]).lower())
        if key not in seen:
            merged.append(case)
            seen.add(key)
    return {
        "schema_version": source.get("schema_version", "teacher-prompts.v1"),
        "generation": {
            **dict(source.get("generation") or {}),
            "manual_augmentation": "navigation_paraphrases_v1",
            "manual_augmentation_count": len(merged) - len(existing),
        },
        "cases": merged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="加入人工导航同义提示词变体")
    parser.add_argument("--output", required=True, help="输出 prompts.json")
    parser.add_argument("--prompts-file", default="", help="可选：在已有 prompts.json 后追加")
    parser.add_argument(
        "--targets", default="",
        help="逗号分隔的目的地；留空使用挂号处/门诊大厅/收费处/服务台/药房/急诊科",
    )
    args = parser.parse_args()
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    additions = build_cases(selected_targets=targets or None)
    if args.prompts_file:
        source = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
        payload = merge_cases(source, additions)
    else:
        payload = {
            "schema_version": "teacher-prompts.v1",
            "generation": {
                "model": "manual",
                "manual_augmentation": "navigation_paraphrases_v1",
                "case_count": len(additions),
                "category_counts": dict(Counter(case["category"] for case in additions)),
            },
            "cases": additions,
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "added": len(additions),
        "targets": sorted({case["expected"]["navigation_target"] for case in additions}),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
