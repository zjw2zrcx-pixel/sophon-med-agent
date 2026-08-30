#!/usr/bin/env python3
"""Compile only failure-proven, independently reviewed correction decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from agents.Harness.state import TaskPlan
from agents.TeacherData.audit_training_readiness import (
    evidence_ids_from_state, latest_state, parse_tool_output,
)


CURRENT_TOOLS = {"medical_consult", "navigate", "query", "speak", "get_time", "get_system_stats"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_hash(row: dict[str, Any]) -> str:
    value = dict(row)
    value.pop("sample_sha256", None)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def validate(row: dict[str, Any], live_hashes: set[str]) -> list[str]:
    errors = []
    corrected = str(row.get("corrected_output") or "")
    baseline = str(row.get("baseline_output") or "")
    if not row.get("failure_codes"):
        errors.append("missing_baseline_failure")
    if not corrected or corrected == baseline:
        errors.append("missing_or_unchanged_correction")
    teacher = str(row.get("teacher_model") or "")
    review = row.get("independent_review")
    if not teacher:
        errors.append("missing_teacher_model")
    if not isinstance(review, dict) or review.get("decision") != "approve":
        errors.append("independent_review_not_approved")
    elif not review.get("reviewer_model") or review.get("reviewer_model") == teacher:
        errors.append("reviewer_not_independent")
    slots = row.get("prompt_slots") or {}
    system_hash = hashlib.sha256(str(slots.get("system", "")).encode()).hexdigest()[:12]
    if system_hash not in live_hashes:
        errors.append("stale_system_prompt")
    try:
        output = parse_tool_output(corrected)
    except Exception:
        errors.append("invalid_tool_envelope")
        return errors
    tool_call = output.get("tool_call")
    param = output.get("param")
    if tool_call == "plan":
        try:
            TaskPlan.from_payload(param)
        except Exception:
            errors.append("invalid_plan")
        return errors
    if tool_call != "act" or not isinstance(param, dict):
        errors.append("not_plan_or_act")
        return errors
    if str(param.get("action_type", "")) != "CALL_TOOL":
        errors.append("unsupported_action_type")
        return errors
    tool = str(param.get("tool", ""))
    arguments = param.get("arguments")
    if tool not in CURRENT_TOOLS or not isinstance(arguments, dict):
        errors.append("invalid_tool_or_arguments")
        return errors
    state = latest_state(str(slots.get("history", "")))
    if state is None:
        errors.append("missing_execution_state")
        return errors
    detail = state.get("current_step_detail") or {}
    if str(param.get("step_id") or "") != str(state.get("current_step_id") or ""):
        errors.append("step_id_mismatch")
    if detail.get("preferred_tool") and tool != detail.get("preferred_tool"):
        errors.append("tool_step_mismatch")
    if tool == "speak":
        text = str(arguments.get("text", ""))
        if not text or len(text) > 90:
            errors.append("invalid_speak_length")
        if row.get("category") in {"medical", "mixed"}:
            maps = review.get("claim_evidence_map") if isinstance(review, dict) else None
            if not isinstance(maps, list) or not maps:
                errors.append("missing_claim_evidence_map")
            else:
                allowed = evidence_ids_from_state(state)
                for item in maps:
                    ids = item.get("evidence_ids") if isinstance(item, dict) else None
                    support = str(item.get("support", "")) if isinstance(item, dict) else ""
                    if ids:
                        if any(str(eid) not in allowed for eid in ids):
                            errors.append("unknown_claim_evidence")
                    elif support not in {"conservative_policy", "uncertainty"}:
                        errors.append("unbound_medical_claim")
    return list(dict.fromkeys(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    live_hashes = set((dataset.get("current_contract_hashes") or {}).values())
    rows = read_jsonl(args.reviewed)
    accepted, rejected = [], []
    for row in rows:
        errors = validate(row, live_hashes)
        if errors:
            rejected.append({"request_id": row.get("request_id"), "errors": errors})
            continue
        output = parse_tool_output(str(row["corrected_output"]))
        action = str(output["tool_call"])
        sample = {
            "schema_version": "agent-sft-decision.v3",
            "case_id": "failure_correction::" + str(row["canary_id"]),
            "decision_id": str(row["request_id"]),
            "category": row["category"], "action": action,
            "agent_iteration": row.get("agent_iteration"),
            "external_turn": row.get("external_turn"),
            "input": {"model": row.get("baseline_model"), "prompt_slots": row["prompt_slots"]},
            "output": row["corrected_output"],
            "gate": {"deterministic": "pass", "semantic_review": "approve", "failure_proven": "pass"},
            "provenance": {
                "generator": "failure_driven_canary.v1",
                "teacher_model": row["teacher_model"],
                "correction": {
                    "baseline_model": row.get("baseline_model"),
                    "baseline_output": row["baseline_output"],
                    "baseline_output_sha256": row["baseline_output_sha256"],
                    "failure_codes": row["failure_codes"],
                    "corrected_output": row["corrected_output"],
                },
                "independent_reviewer_model": row["independent_review"]["reviewer_model"],
            },
            "supervision": {"semantic_review": row["independent_review"]},
            "tags": {
                "conversation_type": "single_turn", "split": "train",
                "intent_metadata": {"semantic_family_id": row["semantic_family_id"]},
            },
        }
        sample["sample_sha256"] = canonical_hash(sample)
        accepted.append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, values in ((args.output, accepted), (args.rejected, rejected)):
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "failure-correction-compile-report.v1",
        "input": len(rows), "accepted": len(accepted), "rejected": len(rejected),
        "trainable": bool(accepted) and not rejected,
        "note": "Dataset-level release still requires held-out pre/post evaluation.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
