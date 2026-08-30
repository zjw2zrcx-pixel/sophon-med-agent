"""Project audited trajectories into compact PLAN/ACT supervised examples."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def _sample_hash(sample: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def project(
    run_dir: Path, decisions: set[str], semantic_approved: dict[str, dict] | None = None,
) -> tuple[list[dict], dict]:
    root = Path(run_dir).resolve()
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    prompts_path = root / "prompts.json"
    prompt_rows: dict[str, dict] = {}
    if prompts_path.is_file():
        prompt_payload = json.loads(prompts_path.read_text(encoding="utf-8"))
        prompt_rows = {
            str(row.get("id")): row for row in prompt_payload.get("cases", [])
        }
    runs_path = root / "runs.json"
    run_rows: dict[str, dict] = {}
    teacher_model = ""
    prompt_model = ""
    if runs_path.is_file():
        run_payload = json.loads(runs_path.read_text(encoding="utf-8"))
        run_rows = {str(row.get("id")): row for row in run_payload.get("runs", [])}
        teacher_model = str(run_payload.get("teacher_model", ""))
        prompt_model = str(run_payload.get("prompt_model", ""))
    semantic_approved = semantic_approved or {}
    allowed = {
        str(row["id"]): row for row in audit.get("cases", [])
        if row.get("decision") in decisions and (
            row.get("decision") != "semantic_review_required"
            or str(row["id"]) in semantic_approved
        )
    }
    samples = []
    skipped_overflow = 0
    for case_id, audit_row in allowed.items():
        prompt_row = prompt_rows.get(case_id, {})
        run_row = run_rows.get(case_id, {})
        category = str(
            prompt_row.get("category") or run_row.get("category") or "unknown"
        )
        scripted_turns = list(prompt_row.get("turns") or [prompt_row.get("prompt", "")])
        actual_turns = list(run_row.get("turns") or [])
        conversation_turns = max(
            1, len(actual_turns),
            max((int(call.get("external_turn") or 0)
                 for relative in audit_row.get("trajectory_files", [])
                 for call in json.loads((root / relative).read_text(encoding="utf-8"))
                    .get("model_calls", [])), default=1),
        )
        expected = prompt_row.get("expected") or run_row.get("expected") or {}
        case_tags = {
            "category": category,
            "difficulty": str(prompt_row.get("difficulty") or "unknown"),
            "conversation_type": "multi_turn" if conversation_turns > 1 else "single_turn",
            "conversation_turns": conversation_turns,
            "scripted_turns": max(1, len(scripted_turns)),
            "required_tools": sorted(str(x) for x in expected.get("required_tools", [])),
            "risk_tags": sorted(str(x) for x in prompt_row.get("risk_tags", [])),
            "teacher_model": teacher_model,
            "prompt_model": prompt_model,
            "source_run": root.name,
            "medical_grounded": category in {"medical", "mixed"},
        }
        intent_metadata = prompt_row.get("intent_metadata") or run_row.get("intent_metadata")
        if isinstance(intent_metadata, dict):
            # Keep provenance in the SFT row so independent-intent data can be
            # counted without reopening the prompt bank.  Do not infer a
            # paraphrase flag from wording; only the audited source may set it.
            case_tags["intent_metadata"] = dict(intent_metadata)
        elif prompt_row or run_row:
            case_tags["intent_metadata"] = {
                "schema_version": "teacher-intent-origin.v1",
                "origin": "original_independent_legacy",
                "independent_intent": True,
                "semantic_family_id": f"legacy:{root.name}:{case_id}",
                "dialogue_mode": "multi_turn" if conversation_turns > 1 else "single_turn",
                "is_paraphrase": False,
                "is_pronunciation_variant": False,
            }
        for relative in audit_row.get("trajectory_files", []):
            trajectory = json.loads((root / relative).read_text(encoding="utf-8"))
            correction = trajectory.get("correction")
            for call in trajectory.get("model_calls", []):
                context = call.get("context_stats") or {}
                if context.get("context_overflow") or context.get("overflow_detected_from_error"):
                    skipped_overflow += 1
                    continue
                output = str(call.get("output", "")).strip()
                if not output.startswith("<tool>"):
                    continue
                action = "plan" if '"tool_call":"plan"' in output else (
                    "act" if '"tool_call":"act"' in output else "other"
                )
                if action == "other":
                    continue
                sample = {
                    "schema_version": "agent-sft-decision.v2",
                    "case_id": case_id,
                    "category": category,
                    "tags": case_tags,
                    "task_id": call.get("task_id"),
                    "external_turn": call.get("external_turn"),
                    "agent_iteration": call.get("agent_iteration"),
                    "action": action,
                    "input": {
                        "prompt_slots": call.get("prompt_slots", {}),
                        "model": call.get("model"),
                    },
                    "output": output,
                    "token_usage": call.get("usage", {}),
                    "context_stats": context,
                    "provenance": {
                        "trajectory": relative,
                        "audit_decision": audit_row.get("decision"),
                        "correction": correction,
                        "semantic_review": semantic_approved.get(case_id),
                        "reasoning_removed": True,
                    },
                }
                sample["sample_sha256"] = _sample_hash(sample)
                samples.append(sample)
    samples.sort(key=lambda row: (
        row["case_id"], int(row.get("external_turn") or 0),
        int(row.get("agent_iteration") or 0), row["sample_sha256"],
    ))
    manifest = {
        "schema_version": "agent-sft-projection.v2",
        "source": str(root), "allowed_decisions": sorted(decisions),
        "case_count": len({sample["case_id"] for sample in samples}),
        "sample_count": len(samples), "skipped_context_overflow_calls": skipped_overflow,
        "ordering": "case_id, external_turn, agent_iteration",
        "loss_contract": "input is masked; calculate loss only on output XML PLAN/ACT",
        "original_trajectories_mutated": False,
        "semantic_approved_case_ids": sorted(semantic_approved),
        "category_cases": dict(Counter(
            sample["category"] for sample in {
                row["case_id"]: row for row in samples
            }.values()
        )),
        "category_samples": dict(Counter(sample["category"] for sample in samples)),
        "conversation_type_cases": dict(Counter(
            sample["tags"]["conversation_type"] for sample in {
                row["case_id"]: row for row in samples
            }.values()
        )),
    }
    return samples, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="将审计通过轨迹重排成逐决策 SFT 样本")
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-decision", action="append", default=["accept"],
        help="默认只允许 accept；医疗 semantic_review_required 须人工审阅后显式加入",
    )
    parser.add_argument(
        "--semantic-review-file", default="",
        help="人工医疗语义复审 JSON；只导出其中 decision=approve 的 case",
    )
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    semantic_approved = {}
    if args.semantic_review_file:
        review = json.loads(Path(args.semantic_review_file).read_text(encoding="utf-8"))
        semantic_approved = {
            str(row["id"]): row for row in review.get("cases", [])
            if row.get("decision") == "approve"
        }
    samples, manifest = project(
        Path(args.run_dir), set(args.allow_decision), semantic_approved
    )
    with (output / "sft_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
