#!/usr/bin/env python3
"""Extract auditable correction requests from an actual baseline Agent run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from agents.TeacherData.audit_training_readiness import latest_state, parse_tool_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_codes(turn: dict[str, Any]) -> list[str]:
    codes = []
    if turn.get("runtime_error"):
        codes.append("runtime_error")
    for error in turn.get("contract_errors") or []:
        codes.append("contract:" + str(error))
    if not bool(turn.get("format_clean", False)):
        codes.append("invalid_model_envelope")
    terminal = str(turn.get("terminal_action") or turn.get("turn_end_reason") or "")
    if terminal not in {"query", "speak"}:
        codes.append("missing_terminal_action")
    calls = turn.get("model_call_records") or []
    if any(int(call.get("attempt_entries", 0) or 0) > 0 for call in calls):
        codes.append("required_repair_attempt")
    if any(str((call.get("benchmark") or {}).get("stop_reason", "")) == "length" for call in calls):
        codes.append("generation_truncated")
    return list(dict.fromkeys(codes))


def call_failure_codes(
    turn: dict[str, Any], calls: list[dict[str, Any]], call_index: int
) -> list[str]:
    """Attribute only errors for which this particular generation is responsible.

    Turn-level contract errors describe the terminal result and therefore belong to
    the last generation.  A repair marker is recorded on the *following* call, so
    the generation that caused the repair is the preceding one.
    """
    call = calls[call_index]
    codes: list[str] = []
    output = str(call.get("output", ""))
    try:
        parse_tool_output(output)
    except Exception:
        codes.append("invalid_model_envelope")
    if str((call.get("benchmark") or {}).get("stop_reason", "")) == "length":
        codes.append("generation_truncated")
    if call_index + 1 < len(calls):
        next_call = calls[call_index + 1]
        if int(next_call.get("attempt_entries", 0) or 0) > 0:
            codes.append("caused_repair_attempt")
            codes.extend(
                "repair:" + str(category)
                for category in (next_call.get("attempt_categories") or [])
            )
    if call_index == len(calls) - 1:
        if turn.get("runtime_error"):
            codes.append("runtime_error")
        codes.extend("contract:" + str(error) for error in (turn.get("contract_errors") or []))
        terminal = str(turn.get("terminal_action") or turn.get("turn_end_reason") or "")
        if terminal not in {"query", "speak"}:
            codes.append("missing_terminal_action")
    return list(dict.fromkeys(codes))


def failure_owner(turn: dict[str, Any], codes: list[str]) -> str:
    """Do not teach the model to violate an upstream frozen workflow."""
    calls = turn.get("model_call_records") or []
    if not calls:
        return "workflow"
    required_missing = [code.split(":", 2)[-1] for code in codes if code.startswith("contract:missing:")]
    if required_missing:
        final_call = calls[-1]
        state = latest_state(str((final_call.get("prompt_slots") or {}).get("history", "")))
        preferred = str(((state or {}).get("current_step_detail") or {}).get("preferred_tool") or "")
        try:
            actual = str(parse_tool_output(str(final_call.get("output", ""))).get("param", {}).get("tool", ""))
        except Exception:
            actual = ""
        if preferred and preferred not in required_missing:
            return "workflow"
    return "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    live_hashes = set((dataset.get("current_contract_hashes") or {}).values())
    manifests = {x["canary_id"]: x for x in dataset.get("selection_manifest", [])}
    turns = read_jsonl(args.run_dir / "turns.jsonl")
    requests = []
    workflow_failures = []
    counts = Counter()
    for turn in turns:
        counts["turns"] += 1
        session_id = str(turn.get("session_id", ""))
        canary_id = session_id.split("-0-", 1)[-1] if "-0-" in session_id else str(turn.get("scenario_id", ""))
        manifest = manifests.get(canary_id)
        if manifest is None:
            counts["unknown_case"] += 1
            continue
        calls = turn.get("model_call_records") or []
        codes = failure_codes(turn)
        if not calls and codes:
            counts["workflow_failure"] += 1
            workflow_failures.append({
                "schema_version": "failure-owner-record.v1",
                "canary_id": canary_id, "category": manifest["category"],
                "prompt": manifest["prompt"], "failure_codes": codes,
                "contract_errors": list(turn.get("contract_errors") or []),
                "model_calls": 0, "owner": "workflow", "trainable": False,
            })
            continue
        if not calls:
            counts["scheduler_only_pass"] += 1
            continue
        call_hashes = {
            hashlib.sha256(str((call.get("prompt_slots") or {}).get("system", "")).encode()).hexdigest()[:12]
            for call in calls
        }
        if not calls or not call_hashes or not call_hashes.issubset(live_hashes):
            counts["stale_or_missing_contract"] += 1
            continue
        if not codes:
            counts["baseline_pass"] += 1
            continue
        owner = failure_owner(turn, codes)
        if owner == "workflow":
            counts["workflow_failure"] += 1
            workflow_failures.append({
                "schema_version": "failure-owner-record.v1",
                "canary_id": canary_id, "category": manifest["category"],
                "prompt": manifest["prompt"], "failure_codes": codes,
                "contract_errors": list(turn.get("contract_errors") or []),
                "model_calls": len(calls), "owner": "workflow",
                "trainable": False,
            })
            continue
        counts["baseline_failure"] += 1
        attributed_calls = [
            (call_index, call, call_failure_codes(turn, calls, call_index))
            for call_index, call in enumerate(calls)
        ]
        attributed_calls = [item for item in attributed_calls if item[2]]
        if not attributed_calls:
            # Preserve a model-owned failure even if an older trace lacks enough
            # per-call metadata.  The final generation owned the terminal result.
            attributed_calls = [(len(calls) - 1, calls[-1], codes)]
        for call_index, call, attributed_codes in attributed_calls:
            output = str(call.get("output", ""))
            prompt_slots = dict(call.get("prompt_slots") or {})
            request_id = f"{canary_id}:call-{call_index + 1}"
            requests.append({
                "schema_version": "failure-correction-request.v1",
                "request_id": request_id,
                "canary_id": canary_id,
                "category": manifest["category"],
                "prompt": manifest["prompt"],
                "semantic_family_id": manifest["semantic_family_id"],
                "prompt_slots": prompt_slots,
                "baseline_model": turn.get("model"),
                "baseline_output": output,
                "baseline_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "failure_codes": attributed_codes,
                "contract_errors": list(turn.get("contract_errors") or []),
                "agent_iteration": call.get("agent_iteration"),
                "external_turn": turn.get("external_turn"),
                "teacher_requirements": {
                    "output": "exactly_one_tool_envelope",
                    "use_current_prompt_state": True,
                    "medical_claims_require_visible_evidence_ids": True,
                    "no_evidence_answer": "explicit_unknown_or_refusal",
                    "max_speak_chars": 90,
                },
                "corrected_output": None,
                "teacher_model": None,
                "independent_review": None,
                "trainable": False,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    workflow_path = args.output.with_name("workflow_failures.jsonl")
    with workflow_path.open("w", encoding="utf-8") as handle:
        for failure in workflow_failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "failure-extraction-report.v1",
        "dataset": str(args.dataset), "run_dir": str(args.run_dir),
        "counts": dict(counts), "correction_requests": len(requests),
        "workflow_failures": len(workflow_failures),
        "trainable": False,
        "blocking_requirements": [
            "teacher_corrected_output", "independent_review",
            "deterministic_compile_gate", "heldout_pre_post_eval",
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
