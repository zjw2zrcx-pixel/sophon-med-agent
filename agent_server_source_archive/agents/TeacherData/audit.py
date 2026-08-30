"""Deterministic quality gate for generated Teacher trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from agents.medical_policy import unsupported_departments


def _commands(run: dict[str, Any]) -> list[dict[str, Any]]:
    turns = run.get("turns")
    if isinstance(turns, list) and turns:
        return [
            command
            for turn in turns if isinstance(turn, dict)
            for command in turn.get("commands", []) if isinstance(command, dict)
        ]
    return [item for item in run.get("commands", []) if isinstance(item, dict)]


def _trajectory_files(root: Path, case_id: str) -> list[str]:
    directory = root / "trajectories" / case_id
    return [
        str(path.relative_to(root))
        for path in sorted(directory.glob("*.json"))
        if path.name != "session.json"
    ]


def _trajectory_errors(root: Path, paths: list[str]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        for value in payload.get("errors", []):
            errors.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return errors


def audit_run(run: dict[str, Any], root: Path) -> dict[str, Any]:
    case_id = str(run.get("id", ""))
    category = str(run.get("category", ""))
    expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
    commands = _commands(run)
    names = [str(command.get("name", "")) for command in commands]
    reasons: list[str] = []

    if run.get("status") != "completed":
        reasons.append("RUN_NOT_COMPLETED")
    if run.get("session_end_reason") != "speak":
        reasons.append("SESSION_NOT_ENDED_BY_SPEAK")
    if not commands or names[-1] != "speak":
        reasons.append("FINAL_COMMAND_NOT_SPEAK")
    for name in expected.get("required_tools", []):
        if name not in names:
            reasons.append(f"MISSING_REQUIRED_TOOL:{name}")
    for name in expected.get("forbidden_tools", []):
        if name in names:
            reasons.append(f"FORBIDDEN_TOOL_USED:{name}")
    for command in commands:
        if command.get("success") is False:
            reasons.append(f"TOOL_FAILED:{command.get('name', '')}")
    if (run.get("token_usage") or {}).get("context_overflow"):
        reasons.append("CONTEXT_OVERFLOW_8K")

    turns = run.get("turns") if isinstance(run.get("turns"), list) else []
    for index, turn in enumerate(turns, 1):
        turn_commands = [
            item for item in turn.get("commands", []) if isinstance(item, dict)
        ]
        turn_names = [str(item.get("name", "")) for item in turn_commands]
        if not turn_names or turn_names[0] != "plan" or turn_names.count("plan") != 1:
            reasons.append(f"TURN_{index}_PLAN_PROTOCOL_ERROR")
        expected_terminal = "speak" if index == len(turns) else "query"
        if not turn_names or turn_names[-1] != expected_terminal:
            reasons.append(f"TURN_{index}_TERMINAL_EXPECTED:{expected_terminal}")
        if turn.get("end_reason") != expected_terminal:
            reasons.append(f"TURN_{index}_END_REASON_EXPECTED:{expected_terminal}")
        for command in turn_commands:
            if command.get("name") != "medical_consult":
                continue
            medical = command.get("medical") if isinstance(command.get("medical"), dict) else {}
            status = str(medical.get("status", ""))
            questions = [q for q in medical.get("questions", []) if str(q).strip()]
            if status in {"need_more_info", "ambiguous"} and questions:
                if expected_terminal != "query" or turn.get("end_reason") != "query":
                    reasons.append(f"TURN_{index}_MEDICAL_FOLLOWUP_IGNORED")
            elif turn.get("end_reason") == "query":
                reasons.append(f"TURN_{index}_QUERY_WITHOUT_TOOL_REQUEST")

    expected_target = str(expected.get("navigation_target") or "").strip()
    navigation = [c for c in commands if c.get("name") == "navigate"]
    actual_targets = [
        str(c.get("params", {}).get("target", "")).strip()
        for c in navigation if isinstance(c.get("params"), dict)
    ]
    if expected_target and expected_target not in actual_targets:
        reasons.append(f"NAVIGATION_TARGET_MISMATCH:{expected_target}")

    # Evidence alignment applies only to a medical answer. A pure navigation
    # case must be allowed to speak its explicitly labelled department target.
    if category in {"medical", "mixed"}:
        consultation: dict[str, Any] = {}
        for command in commands:
            medical = command.get("medical") if isinstance(command.get("medical"), dict) else {}
            if medical:
                consultation = medical
        unsupported = unsupported_departments(
            str(run.get("final", "")), consultation, actual_targets
        )
        if unsupported:
            reasons.append("UNSUPPORTED_DEPARTMENT:" + ",".join(unsupported))

    trajectory_files = _trajectory_files(root, case_id)
    if not trajectory_files:
        reasons.append("TRAJECTORY_FILE_MISSING")
    trajectory_errors = _trajectory_errors(root, trajectory_files)
    if trajectory_errors:
        reasons.append("TRAJECTORY_HAS_ERRORS")

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        decision = "reject"
    elif category in {"medical", "mixed"}:
        decision = "semantic_review_required"
    else:
        decision = "accept"
    return {
        "id": case_id,
        "category": category,
        "decision": decision,
        "reasons": reasons,
        "tool_sequence": names,
        "expected_navigation_target": expected_target or None,
        "actual_navigation_targets": actual_targets,
        "trajectory_files": trajectory_files,
        "trajectory_errors": trajectory_errors,
    }


def audit_runs(root: Path, runs_payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    rows = [audit_run(run, root) for run in runs_payload.get("runs", [])]
    counts = Counter(row["decision"] for row in rows)
    return {
        "schema_version": "teacher-audit.v1",
        "source": str((root / "runs.json").resolve()),
        "summary": {
            "total": len(rows),
            "accept": counts["accept"],
            "semantic_review_required": counts["semantic_review_required"],
            "reject": counts["reject"],
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 Teacher trajectory")
    parser.add_argument("run_dir", help="包含 runs.json 和 trajectories/ 的目录")
    parser.add_argument("--output", default="", help="默认写入 <run_dir>/audit.json")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    payload = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    audited = audit_runs(root, payload)
    output = Path(args.output).resolve() if args.output else root / "audit.json"
    output.write_text(json.dumps(audited, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
