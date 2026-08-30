#!/usr/bin/env python3
"""Build a diverse, benchmark-disjoint pool for failure-driven SFT harvesting.

The output is not training data.  A case becomes a correction candidate only
after the deployed baseline actually fails it and an independently reviewed
replacement decision is available.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from agents.TeacherData.targeted_sft_v3 import exclusion_registry
from agents.agent import Agent, AgentConfig


ROOT = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = {
    "medical": "医疗问询", "navigation": "导航",
    "general": "综合问答", "mixed": "混合意图",
}
SOURCES = (
    "teacher_trajectories/prompt_bank_v7_independent_preflight_1000/prompts.json",
    "teacher_trajectories/prompt_bank_v6_flash_novelty/prompts.json",
    "teacher_trajectories/prompt_bank_v5_preflight_1000/prompts.json",
    "teacher_trajectories/luna_prompt_reviewed_v2_160/prompts.json",
    "teacher_trajectories/prompt_bank_v2_148/train_single_turn/prompts.json",
)
HARVEST_QUOTAS = {"medical": 350, "navigation": 250, "general": 200, "mixed": 200}
HOLDOUT_QUOTAS = {"medical": 35, "navigation": 25, "general": 20, "mixed": 20}


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def family(case: dict[str, Any]) -> str:
    metadata = case.get("intent_metadata") or (case.get("final_tags") or {}).get("intent_metadata") or {}
    value = str(metadata.get("semantic_family_id", "")).strip()
    return value or "surface:" + sha(norm(str(case.get("prompt", ""))))[:20]


def live_hashes() -> dict[str, str]:
    result = {}
    for name, benchmark in (("Benchmark", True), ("Voice", False)):
        agent = Agent(AgentConfig(
            default_mode=name, benchmark_enabled=benchmark,
            trajectory_enabled=False, medical_dense_enabled=False,
            navigation_location_profile="hospital" if benchmark else "basic",
        ))
        agent.initialize()
        mode = agent.benchmark_mode if benchmark else agent.voice_mode
        result[name] = sha(mode.get_base_prompt())[:12]
    return result


def load_candidates() -> list[dict[str, Any]]:
    result = []
    for source_index, relative in enumerate(SOURCES):
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            prompt = str(case.get("prompt", "")).strip()
            turns = case.get("turns") or [prompt]
            # This harvest intentionally targets the simple production regime.
            if len(turns) != 1 or not prompt or "还没说具体名称" in prompt:
                continue
            category = str(case.get("category", ""))
            if category not in CATEGORY_NAMES:
                continue
            rank_material = "\0".join(("failure-canary-20260821", category, family(case), prompt))
            result.append({
                "case": case, "source": relative, "source_index": source_index,
                "family": family(case), "rank": sha(rank_material),
            })
    return result


def select(candidates: list[dict[str, Any]], quotas: dict[str, int], used_prompts: set[str], used_families: set[str], excluded_hashes: set[str]) -> list[dict[str, Any]]:
    selected = []
    by_category = defaultdict(list)
    for item in candidates:
        by_category[item["case"]["category"]].append(item)
    for category, quota in quotas.items():
        # Prefer the newest independent source, then deterministic rank.
        ordered = sorted(by_category[category], key=lambda x: (x["source_index"], x["rank"]))
        for item in ordered:
            case = item["case"]
            prompt = str(case["prompt"]).strip()
            prompt_norm = norm(prompt)
            if (
                prompt_norm in used_prompts
                or item["family"] in used_families
                or sha(prompt_norm) in excluded_hashes
            ):
                continue
            used_prompts.add(prompt_norm)
            used_families.add(item["family"])
            selected.append(item)
            if sum(x["case"]["category"] == category for x in selected) == quota:
                break
        actual = sum(x["case"]["category"] == category for x in selected)
        if actual != quota:
            raise RuntimeError(f"only selected {actual}/{quota} for {category}")
    return selected


def materialize(items: list[dict[str, Any]], prefix: str, split: str) -> tuple[list[dict], list[dict]]:
    sessions, manifest = [], []
    for index, item in enumerate(items, 1):
        case = item["case"]
        expected = dict(case.get("expected") or {})
        prompt = str(case["prompt"]).strip()
        category = str(case["category"])
        case_id = f"{prefix}-{index:04d}"
        turn = {
            "user": prompt,
            "required_tools": list(expected.get("required_tools") or []),
            "forbidden_tools": list(expected.get("forbidden_tools") or []),
            "navigation_target": expected.get("navigation_target"),
            "navigation_action": expected.get("navigation_action"),
            "expected_notes": expected.get("notes", ""),
            "source_case_id": case.get("id"),
            "source_version": "failure_canary",
            "difficulty": case.get("difficulty"),
        }
        sessions.append({
            "session_id": case_id, "category": CATEGORY_NAMES[category],
            "category_key": category, "source_version": "failure_canary",
            "source_case_id": case.get("id"), "difficulty": case.get("difficulty"),
            "turns": [turn],
        })
        manifest.append({
            "canary_id": case_id, "split": split, "category": category,
            "category_name": CATEGORY_NAMES[category], "prompt": prompt,
            "prompt_sha256": sha(prompt), "normalized_prompt_sha256": sha(norm(prompt)),
            "semantic_family_id": item["family"], "source_file": item["source"],
            "source_case_id": case.get("id"), "difficulty": case.get("difficulty"),
        })
    return sessions, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    excluded = exclusion_registry(ROOT)
    used_prompts: set[str] = set()
    used_families: set[str] = set(excluded["family_ids"])
    candidates = load_candidates()
    harvest = select(candidates, HARVEST_QUOTAS, used_prompts, used_families, set(excluded["prompt_hashes"]))
    holdout = select(candidates, HOLDOUT_QUOTAS, used_prompts, used_families, set(excluded["prompt_hashes"]))
    harvest_sessions, harvest_manifest = materialize(harvest, "harvest", "failure_harvest")
    holdout_sessions, holdout_manifest = materialize(holdout, "holdout", "never_train")
    counts = Counter(x["category_key"] for x in harvest_sessions)
    holdout_counts = Counter(x["category_key"] for x in holdout_sessions)
    payload = {
        "schema_version": "2.0",
        "benchmark_schema": "failure-harvest-canary.v1",
        "language": "zh-CN", "driver": "production_agent",
        "trainable": False,
        "policy": {
            "failure_harvest_only": True,
            "baseline_failure_required_before_compilation": True,
            "holdout_never_train": True,
            "semantic_family_cap": 1,
            "benchmark_prompt_exclusion": True,
        },
        "current_contract_hashes": live_hashes(),
        "requirements": {"medical_embedding": {
            "required": True, "model": "qwen3-embedding-0.6b",
            "default_url": "http://127.0.0.1:8006", "dense_retrieval_required": True,
        }},
        "sampling": {
            "seed": "failure-canary-20260821", "harvest_quotas": HARVEST_QUOTAS,
            "holdout_quotas": HOLDOUT_QUOTAS,
            "harvest_counts": dict(counts), "holdout_counts": dict(holdout_counts),
            "sources": list(SOURCES),
        },
        "sessions": harvest_sessions,
        "selection_manifest": harvest_manifest,
        "holdout_sessions": holdout_sessions,
        "holdout_manifest": holdout_manifest,
        "core_external_turns": len(harvest_sessions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"harvest": len(harvest), "holdout": len(holdout), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
