#!/usr/bin/env python3
"""Strict, reproducible training-readiness audit for agent decision SFT data.

This auditor is deliberately conservative.  It never treats an inherited
``approve`` flag as proof that a sample still matches the live agent contract.
It emits measurements and case-level quarantine reasons; it does not rewrite
or silently repair training examples.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from agents.Harness.state import TaskPlan
from agents.MCP.tools.navigate import HOSPITAL_LOCATIONS
from agents.agent import Agent, AgentConfig


TOOL_RE = re.compile(r"^\s*<tool>(.*)</tool>\s*$", re.S)
EVENT_RE = re.compile(r"<execution_state_event>(.*?)</execution_state_event>", re.S)
CURRENT_TOOLS = {
    "medical_consult", "navigate", "query", "speak", "get_time",
    "get_system_stats",
}
REQUIRED_SUHA_SLOTS = {"version", "system", "conversation", "user", "history", "attempt"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def source_name(row: dict[str, Any]) -> str:
    return str(row.get("tags", {}).get("source_run") or row.get("provenance", {}).get("generator") or "unknown")


def parse_tool_output(text: str) -> dict[str, Any]:
    match = TOOL_RE.match(text)
    if not match:
        raise ValueError("output_not_single_tool_tag")
    value = json.loads(match.group(1))
    if not isinstance(value, dict):
        raise ValueError("tool_payload_not_object")
    return value


def latest_state(history: str) -> dict[str, Any] | None:
    matches = EVENT_RE.findall(history or "")
    if not matches:
        return None
    event = json.loads(matches[-1])
    state = event.get("state")
    return state if isinstance(state, dict) else None


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text or "").lower()


def canonical_sample_hash(row: dict[str, Any]) -> str:
    value = dict(row)
    value.pop("sample_sha256", None)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def extract_user_text(row: dict[str, Any]) -> str:
    text = str(row.get("input", {}).get("prompt_slots", {}).get("user", "")).strip()
    match = re.search(
        r"(?:用户输入|当前用户输入)\s*[:：]\s*(.+?)"
        r"(?:\n(?:本轮执行约束|冻结计划|任务约束)\s*[:：]|\n<plan>|\n<instruction>|\Z)",
        text,
        re.S,
    )
    return match.group(1).strip() if match else text


def evidence_ids_from_state(state: dict[str, Any] | None) -> set[str]:
    if not state:
        return set()
    facts = state.get("known_facts", {})
    med = facts.get("medical.consultation", {}) if isinstance(facts, dict) else {}
    if not isinstance(med, dict):
        return set()
    result: set[str] = set()
    stack: list[Any] = [med]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key in ("evidence_id", "association_id"):
                evidence_id = value.get(key)
                if evidence_id:
                    result.add(str(evidence_id))
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return result


def iter_claim_maps(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    supervision = row.get("supervision", {})
    direct = supervision.get("claim_evidence_map") if isinstance(supervision, dict) else None
    if isinstance(direct, list):
        return direct
    review = supervision.get("semantic_review") if isinstance(supervision, dict) else None
    if not isinstance(review, dict):
        return ()
    maps = review.get("claim_evidence_map")
    return maps if isinstance(maps, list) else ()


def add_reason(reasons: dict[str, list[str]], case_id: str, reason: str) -> None:
    if reason not in reasons[case_id]:
        reasons[case_id].append(reason)


def collect_eval_prompts(root: Path) -> set[str]:
    """Best-effort exact prompt collection from benchmark dataset files."""
    prompts: set[str] = set()
    candidates = list((root / "docs").glob("**/dataset.json"))
    for path in candidates:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [obj]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"prompt", "user", "user_input", "query", "utterance"} and isinstance(item, str):
                        norm = normalize_text(item)
                        if norm:
                            prompts.add(norm)
                    elif isinstance(item, (dict, list)):
                        stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
    return prompts


def current_contract() -> tuple[dict[str, str], Any]:
    """Return live base-prompt hashes and the deterministic benchmark planner."""
    hashes: dict[str, str] = {}
    benchmark_mode = None
    for mode_name, benchmark in (("Benchmark", True), ("Voice", False)):
        agent = Agent(AgentConfig(
            default_mode=mode_name,
            benchmark_enabled=benchmark,
            trajectory_enabled=False,
            medical_dense_enabled=False,
            navigation_location_profile="hospital" if benchmark else "basic",
        ))
        agent.initialize()
        mode = agent.benchmark_mode if benchmark else agent.voice_mode
        prompt = mode.get_base_prompt()
        hashes[mode_name] = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if benchmark:
            benchmark_mode = mode
    return hashes, benchmark_mode


def audit(rows: list[dict[str, Any]], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: collections.Counter[str] = collections.Counter()
    by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_category: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    case_reasons: dict[str, list[str]] = collections.defaultdict(list)
    prompt_groups: dict[str, set[str]] = collections.defaultdict(set)
    output_groups: dict[str, set[str]] = collections.defaultdict(set)
    family_groups: dict[str, set[str]] = collections.defaultdict(set)
    system_hashes: collections.Counter[str] = collections.Counter()
    tool_counts: collections.Counter[str] = collections.Counter()
    plan_signatures: collections.Counter[str] = collections.Counter()
    speak_lengths: list[int] = []
    eval_prompts = collect_eval_prompts(root)
    live_hashes, benchmark_mode = current_contract()
    accepted_system_hashes = set(live_hashes.values())
    seen_hashes: set[str] = set()
    seen_decision_keys: set[tuple[str, str]] = set()

    for row in rows:
        counts["rows"] += 1
        case_id = str(row.get("case_id", ""))
        source = source_name(row)
        category = str(row.get("category", "unknown"))
        action = str(row.get("action", ""))
        sample_hash = str(row.get("sample_sha256", ""))
        if not sample_hash or sample_hash != canonical_sample_hash(row):
            counts["invalid_sample_hash"] += 1
            add_reason(case_reasons, case_id, "invalid_sample_hash")
        if sample_hash in seen_hashes:
            counts["duplicate_sample_hash"] += 1
            add_reason(case_reasons, case_id, "duplicate_sample_hash")
        seen_hashes.add(sample_hash)
        decision_key = (
            case_id,
            str(
                row.get("decision_id")
                or f"turn:{row.get('external_turn')}:iteration:{row.get('agent_iteration')}"
            ),
        )
        if decision_key in seen_decision_keys:
            counts["duplicate_case_decision_key"] += 1
            add_reason(case_reasons, case_id, "duplicate_case_decision_key")
        seen_decision_keys.add(decision_key)
        slots = row.get("input", {}).get("prompt_slots", {})
        version = str(slots.get("version", ""))
        system = str(slots.get("system", ""))
        system_hash = hashlib.sha256(system.encode()).hexdigest()[:12]
        system_hashes[system_hash] += 1
        by_source[source]["rows"] += 1
        by_category[category]["rows"] += 1

        raw_user = extract_user_text(row)
        prompt_groups[normalize_text(raw_user)].add(case_id)
        family = str(row.get("tags", {}).get("intent_metadata", {}).get("semantic_family_id", ""))
        if family:
            family_groups[family].add(case_id)
        else:
            counts["missing_semantic_family"] += 1

        if normalize_text(raw_user) in eval_prompts:
            counts["exact_eval_prompt_collision_rows"] += 1
            add_reason(case_reasons, case_id, "exact_eval_prompt_collision")

        if system_hash not in accepted_system_hashes:
            counts["stale_or_nonlive_system_prompt_rows"] += 1
            add_reason(case_reasons, case_id, "stale_or_nonlive_system_prompt")

        if version == "suha.v3":
            counts["suha_v3_rows"] += 1
            missing = REQUIRED_SUHA_SLOTS - set(slots)
            if missing:
                counts["missing_live_slots"] += 1
                add_reason(case_reasons, case_id, "missing_live_prompt_slots")
        else:
            counts["synthetic_or_nonlive_slot_rows"] += 1
            add_reason(case_reasons, case_id, "synthetic_or_nonlive_prompt_slots")

        try:
            output = parse_tool_output(str(row.get("output", "")))
        except (ValueError, json.JSONDecodeError):
            counts["invalid_tool_output"] += 1
            add_reason(case_reasons, case_id, "invalid_tool_output")
            continue

        tool_call = output.get("tool_call")
        param = output.get("param")
        if tool_call != action or not isinstance(param, dict):
            counts["action_envelope_mismatch"] += 1
            add_reason(case_reasons, case_id, "action_envelope_mismatch")
            continue

        if action == "plan":
            try:
                plan = TaskPlan.from_payload(param)
            except (TypeError, ValueError):
                counts["invalid_plan"] += 1
                add_reason(case_reasons, case_id, "invalid_plan")
                continue
            signature = tuple(step.preferred_tool or "null" for step in plan.steps)
            plan_signatures[" -> ".join(signature)] += 1
            if any(step.preferred_tool and step.preferred_tool not in CURRENT_TOOLS for step in plan.steps):
                counts["unknown_plan_tool"] += 1
                add_reason(case_reasons, case_id, "unknown_plan_tool")
            current = benchmark_mode.get_compact_workflow_plan(raw_user)
            current_signature = (
                tuple(str(step.get("preferred_tool") or "") for step in current.get("steps", []))
                if isinstance(current, dict) else None
            )
            if signature != current_signature:
                counts["plan_mismatch_current_workflow"] += 1
                correction = row.get("provenance", {}).get("correction")
                if correction:
                    counts["plan_mismatch_with_explicit_correction"] += 1
                else:
                    counts["plan_mismatch_without_correction_proof"] += 1
                    add_reason(case_reasons, case_id, "plan_mismatch_without_correction_proof")
            else:
                counts["plan_match_current_workflow"] += 1
            continue

        tool = str(param.get("tool", ""))
        arguments = param.get("arguments")
        tool_counts[tool] += 1
        if tool not in CURRENT_TOOLS or not isinstance(arguments, dict):
            counts["invalid_act_tool_or_arguments"] += 1
            add_reason(case_reasons, case_id, "invalid_act_tool_or_arguments")

        state = latest_state(str(slots.get("history", "")))
        if version == "suha.v3" and state is None:
            counts["act_missing_execution_state"] += 1
            add_reason(case_reasons, case_id, "act_missing_execution_state")
        if state is not None:
            counts["acts_with_execution_state"] += 1
            current_id = str(state.get("current_step_id") or "")
            current_detail = state.get("current_step_detail") or {}
            preferred = str(current_detail.get("preferred_tool") or "") if isinstance(current_detail, dict) else ""
            if str(param.get("step_id") or "") != current_id:
                counts["act_step_mismatch"] += 1
                add_reason(case_reasons, case_id, "act_step_mismatch")
            if preferred and tool != preferred:
                counts["act_tool_mismatch"] += 1
                add_reason(case_reasons, case_id, "act_tool_mismatch")

        if tool == "medical_consult":
            query = str(arguments.get("query", "")).strip() if isinstance(arguments, dict) else ""
            if not query or normalize_text(query) != normalize_text(raw_user):
                counts["medical_query_not_current_user_verbatim"] += 1
                add_reason(case_reasons, case_id, "medical_query_not_current_user_verbatim")
        elif tool == "navigate":
            target = str(arguments.get("target", "")) if isinstance(arguments, dict) else ""
            action_arg = str(arguments.get("action", "")) if isinstance(arguments, dict) else ""
            if action_arg == "start" and target not in HOSPITAL_LOCATIONS:
                counts["invalid_navigation_target"] += 1
                add_reason(case_reasons, case_id, "invalid_navigation_target")
        elif tool == "speak":
            text = str(arguments.get("text", "")) if isinstance(arguments, dict) else ""
            length = len(text)
            speak_lengths.append(length)
            output_groups[normalize_text(text)].add(case_id)
            if not text.strip():
                counts["empty_speak"] += 1
                add_reason(case_reasons, case_id, "empty_speak")
            if length > 90:
                counts["speak_over_90_chars"] += 1
            if length > 120:
                counts["speak_over_120_chars"] += 1

            if category in {"medical", "mixed"}:
                counts["medical_or_mixed_speak"] += 1
                maps = list(iter_claim_maps(row))
                if not maps:
                    counts["medical_speak_without_claim_map"] += 1
                    add_reason(case_reasons, case_id, "medical_speak_without_claim_map")
                else:
                    counts["medical_speak_with_claim_map"] += 1
                    allowed = evidence_ids_from_state(state)
                    for item in maps:
                        ids = item.get("evidence_ids", []) if isinstance(item, dict) else []
                        support = item.get("support", "") if isinstance(item, dict) else ""
                        if ids and (not allowed or any(str(eid) not in allowed for eid in ids)):
                            counts["claim_map_unknown_evidence"] += 1
                            add_reason(case_reasons, case_id, "claim_map_unknown_evidence")
                        if not ids and support not in {"conservative_policy", "tool_message", "uncertainty"}:
                            counts["claim_map_unbound_claim"] += 1
                            add_reason(case_reasons, case_id, "claim_map_unbound_claim")

    duplicate_prompt_groups = {k: v for k, v in prompt_groups.items() if k and len(v) > 1}
    duplicate_output_groups = {k: v for k, v in output_groups.items() if k and len(v) > 1}
    repeated_family_groups = {k: v for k, v in family_groups.items() if len(v) > 1}
    counts["exact_duplicate_prompt_groups"] = len(duplicate_prompt_groups)
    counts["cases_in_exact_duplicate_prompt_groups"] = len(set().union(*duplicate_prompt_groups.values())) if duplicate_prompt_groups else 0
    counts["exact_duplicate_speak_groups"] = len(duplicate_output_groups)
    counts["cases_in_exact_duplicate_speak_groups"] = len(set().union(*duplicate_output_groups.values())) if duplicate_output_groups else 0
    counts["repeated_semantic_family_groups"] = len(repeated_family_groups)
    counts["cases_in_repeated_semantic_families"] = len(set().union(*repeated_family_groups.values())) if repeated_family_groups else 0

    all_cases = {str(row.get("case_id", "")) for row in rows}
    quarantined = set(case_reasons)
    decisions = [
        {"case_id": case_id, "decision": "quarantine" if case_id in quarantined else "provisional_pass", "reasons": case_reasons.get(case_id, [])}
        for case_id in sorted(all_cases)
    ]
    report = {
        "schema_version": "training-readiness-audit.v1",
        "policy": {
            "case_atomicity": True,
            "inherited_approve_is_not_sufficient": True,
            "provisional_pass_is_not_authorization_to_train": True,
        },
        "totals": {
            **dict(sorted(counts.items())),
            "cases": len(all_cases),
            "quarantined_cases": len(quarantined),
            "provisional_pass_cases": len(all_cases - quarantined),
        },
        "by_source": {k: dict(sorted(v.items())) for k, v in sorted(by_source.items())},
        "by_category": {k: dict(sorted(v.items())) for k, v in sorted(by_category.items())},
        "system_prompt_hashes": dict(system_hashes.most_common()),
        "current_live_system_prompt_hashes": live_hashes,
        "tool_counts": dict(tool_counts.most_common()),
        "plan_signatures": dict(plan_signatures.most_common()),
        "speak_length": {
            "count": len(speak_lengths),
            "mean": round(sum(speak_lengths) / len(speak_lengths), 2) if speak_lengths else 0,
            "max": max(speak_lengths, default=0),
        },
        "top_duplicate_prompts": [
            {"normalized_prompt": key[:160], "case_count": len(value), "case_ids": sorted(value)[:12]}
            for key, value in sorted(duplicate_prompt_groups.items(), key=lambda item: len(item[1]), reverse=True)[:20]
        ],
        "top_duplicate_speaks": [
            {"normalized_text": key[:160], "case_count": len(value), "case_ids": sorted(value)[:12]}
            for key, value in sorted(duplicate_output_groups.items(), key=lambda item: len(item[1]), reverse=True)[:20]
        ],
        "top_semantic_families": [
            {"semantic_family_id": key, "case_count": len(value), "case_ids": sorted(value)[:12]}
            for key, value in sorted(family_groups.items(), key=lambda item: len(item[1]), reverse=True)[:20]
        ],
        "quarantine_reason_counts": dict(collections.Counter(reason for values in case_reasons.values() for reason in values).most_common()),
    }
    return report, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    root = Path(__file__).resolve().parents[2]
    report, decisions = audit(rows, root)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.cases.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
