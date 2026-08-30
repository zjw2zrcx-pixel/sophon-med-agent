"""Validate, review and compile Luna-authored four-category workflow SFT."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _many(paths: Iterable[str]) -> list[dict[str, Any]]:
    return [row for path in paths for row in _read(path)]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse(output: str) -> tuple[dict[str, Any] | None, list[str]]:
    value = str(output or "").strip()
    if not value.startswith("<tool>") or not value.endswith("</tool>"):
        return None, ["INVALID_TOOL_ENVELOPE"]
    try:
        payload = json.loads(value[6:-7])
    except json.JSONDecodeError:
        return None, ["INVALID_TOOL_JSON"]
    return (payload, []) if isinstance(payload, dict) else (None, ["TOOL_NOT_OBJECT"])


def validate(request: dict[str, Any], output: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    payload, errors = _parse(str(output.get("output", "")))
    frame = request["decision_frame"]
    phase = request["phase"]
    text = ""
    if payload is None:
        return {"pass": False, "errors": errors, "text": text}
    param = payload.get("param")
    if not isinstance(param, dict):
        errors.append("PARAM_NOT_OBJECT")
        param = {}
    if phase == "plan":
        if payload.get("tool_call") != "plan": errors.append("EXPECTED_PLAN")
        expected_steps = frame["expected_plan"].get("steps", [])
        actual_steps = param.get("steps") if isinstance(param.get("steps"), list) else []
        expected_signature = [(x.get("step_id"), x.get("preferred_tool")) for x in expected_steps]
        actual_signature = [(x.get("step_id"), x.get("preferred_tool")) for x in actual_steps if isinstance(x, dict)]
        if actual_signature != expected_signature: errors.append("PLAN_STEP_OR_TOOL_MISMATCH")
    else:
        if payload.get("tool_call") != "act": errors.append("EXPECTED_ACT")
        if param.get("action_type") != "CALL_TOOL": errors.append("EXPECTED_CALL_TOOL")
        if str(param.get("step_id", "")) != str(frame.get("current_step_id", "")):
            errors.append("WRONG_STEP_ID")
        allowed = list(frame.get("allowed_tools") or [])
        tool = str(param.get("tool", ""))
        if len(allowed) != 1 or tool != allowed[0]: errors.append("WRONG_TOOL")
        arguments = param.get("arguments")
        if not isinstance(arguments, dict):
            errors.append("ARGUMENTS_NOT_OBJECT"); arguments = {}
        if tool == "medical_consult" and arguments != {"query": request["prompt"]}:
            errors.append("MEDICAL_QUERY_NOT_EXACT")
        if tool == "navigate":
            expected_target = str((prompt.get("expected") or {}).get("navigation_target", ""))
            if arguments.get("action") != "start" or arguments.get("target") != expected_target:
                errors.append("NAVIGATION_ARGUMENT_MISMATCH")
        if tool in {"get_time", "get_system_stats"} and arguments:
            errors.append("READ_TOOL_ARGUMENTS_NOT_EMPTY")
        if tool == "speak":
            text = str(arguments.get("text", "")).strip()
            if not text: errors.append("EMPTY_SPEAK")
            limit = 90 if frame.get("category") in {"medical", "mixed"} else 120
            if len(text) > limit: errors.append("SPEAK_TOO_LONG")
            if text and text[-1] not in "。！？!?": errors.append("INCOMPLETE_SPEAK")
    return {"pass": not errors, "errors": sorted(set(errors)), "text": text}


def _evidence_ids(observations: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in observations:
        value = row.get("observation") or {}
        for item in value.get("evidence", []) + value.get("associations", []):
            identifier = item.get("evidence_id") or item.get("association_id")
            if identifier: ids.append(str(identifier))
    return ids


def prepare(requests: list[dict[str, Any]], outputs: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = {str(x.get("request_id")): x for x in outputs}
    prompt_by_id = {str(x["id"]): x for x in prompts}
    rows, rejected = [], Counter()
    for request in requests:
        candidate = out.get(str(request["request_id"]))
        if candidate is None:
            rejected["missing_output"] += 1; continue
        result = validate(request, candidate, prompt_by_id[request["case_id"]])
        if not result["pass"]:
            rejected.update(result["errors"]); continue
        if request["phase"] != "act" or request["decision_frame"]["allowed_tools"] != ["speak"]:
            continue
        observations = request["decision_frame"].get("tool_observations", [])
        rows.append({
            "schema_version": "workflow-semantic-review-request.v1",
            "request_id": request["request_id"], "case_id": request["case_id"],
            "category": request["decision_frame"]["category"],
            "prompt": request["prompt"], "candidate_text": result["text"],
            "tool_observations": observations,
            "allowed_evidence_ids": _evidence_ids(observations),
            "instruction": "只依据工具observation审核回答；不得改写。医学事实须绑定有效证据ID；稳定常识或工具数值可标conservative_policy。",
        })
    return rows, {"requests": len(requests), "outputs": len(outputs), "semantic_review_requests": len(rows), "rejected": dict(rejected)}


def compile_rows(requests: list[dict[str, Any]], outputs: list[dict[str, Any]], prompts: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_by_id = {str(x.get("request_id")): x for x in outputs}
    review_by_id = {str(x.get("request_id")): x for x in reviews}
    prompt_by_id = {str(x["id"]): x for x in prompts}
    samples, rejected = [], Counter()
    failed_case_ids = {
        str(request["case_id"])
        for request in requests
        if request["phase"] == "act"
        and request["decision_frame"]["allowed_tools"] == ["speak"]
        and (
            not review_by_id.get(str(request["request_id"]))
            or review_by_id[str(request["request_id"])].get("decision") != "approve"
        )
    }
    for request in requests:
        if str(request["case_id"]) in failed_case_ids:
            continue
        request_id = str(request["request_id"])
        candidate = output_by_id.get(request_id)
        if candidate is None:
            rejected["missing_output"] += 1; continue
        result = validate(request, candidate, prompt_by_id[request["case_id"]])
        if not result["pass"]:
            rejected.update(result["errors"]); continue
        is_speak = request["phase"] == "act" and request["decision_frame"]["allowed_tools"] == ["speak"]
        review = review_by_id.get(request_id)
        if is_speak and (not review or review.get("decision") != "approve"):
            rejected["semantic_review_not_approved"] += 1; continue
        sample = {
            "schema_version": "agent-sft-decision.v3", "case_id": request["case_id"],
            "decision_id": request_id, "category": request["decision_frame"]["category"],
            "external_turn": 1, "agent_iteration": int(request["decision_index"]) + 1,
            "action": request["phase"], "input": {"prompt_slots": request["prompt_slots"], "model": "gpt-5.6-luna"},
            "output": str(candidate["output"]).strip(),
            "tags": {"conversation_type": "single_turn", "source_run": "luna_workflow_batch1_128", "split": "train"},
            "supervision": {"decision_frame": request["decision_frame"], "semantic_review": review},
            "provenance": {"teacher_model": candidate.get("teacher_model"), "generator": "workflow_sft_v1", "reasoning_removed": True},
            "gate": {"deterministic": "pass", "semantic_review": "approve" if is_speak else "not_required", "human_spot_review": "waived_by_user_ai_review"},
        }
        sample["sample_sha256"] = _sha(sample); samples.append(sample)
    report = {"schema_version": "workflow-sft-compile-report.v1", "request_count": len(requests), "sample_count": len(samples), "case_count": len({x['case_id'] for x in samples}), "semantic_review_excluded_cases": sorted(failed_case_ids), "semantic_review_excluded_decisions": sum(str(x["case_id"]) in failed_case_ids for x in requests), "rejected": dict(rejected), "categories": dict(Counter(x["category"] for x in samples)), "actions": dict(Counter(x["action"] for x in samples)), "trainable": bool(samples) and not rejected}
    return samples, report


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-review", "compile"):
        p = sub.add_parser(name); p.add_argument("--requests", required=True); p.add_argument("--prompts", required=True); p.add_argument("--teacher-outputs", required=True, nargs="+"); p.add_argument("--output-dir", required=True)
        if name == "compile": p.add_argument("--semantic-reviews", required=True, nargs="+")
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    requests = _read(args.requests)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["cases"]
    candidates = _many(args.teacher_outputs)
    if args.command == "prepare-review":
        rows, report = prepare(requests, candidates, prompts); _write(output / "semantic_review_requests.jsonl", rows); name = "preparation_report.json"
    else:
        rows, report = compile_rows(requests, candidates, prompts, _many(args.semantic_reviews)); _write(output / "train.jsonl", rows); _write(output / "sft_decisions.jsonl", rows); name = "manifest.json"
    (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
