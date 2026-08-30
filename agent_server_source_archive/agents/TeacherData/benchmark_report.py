"""Summarize an untouched benchmark trajectory run into comparable metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[rank], 6)


def build_report(
    run_dir: Path, semantic_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    runs_payload = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    audit_payload = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    runs = runs_payload.get("runs", [])
    audit_rows = audit_payload.get("cases", [])
    audit_by_id = {str(row.get("id")): row for row in audit_rows}
    latencies = [float(row["elapsed_seconds"]) for row in runs if row.get("elapsed_seconds") is not None]
    turns: list[int] = []
    categories: dict[str, Counter] = {}
    tool_errors = state_errors = context_overflow_cases = 0
    provider_total_overflow_cases = 0
    token_totals = Counter()
    peak_prompt_tokens = 0

    for run in runs:
        usage = run.get("token_usage") or {}
        for key in ("prompt_tokens_sum", "completion_tokens_sum", "total_tokens_sum"):
            token_totals[key] += int(usage.get(key, 0) or 0)
        peak_prompt_tokens = max(peak_prompt_tokens, int(usage.get("peak_prompt_tokens", 0) or 0))
        context_overflow_cases += int(bool(usage.get("context_overflow")))
        provider_total_overflow_cases += int(
            int(usage.get("provider_total_overflow_calls", 0) or 0) > 0
        )
        audit = audit_by_id.get(str(run.get("id")), {})
        errors = [str(value) for value in audit.get("trajectory_errors", [])]
        tool_errors += sum('"category": "tool"' in value for value in errors)
        state_errors += sum('"category": "state"' in value for value in errors)
        trajectory_turns = set()
        for relative in audit.get("trajectory_files", []):
            path = root / relative
            if not path.is_file():
                continue
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            turn = trajectory.get("external_turn")
            if turn is not None:
                trajectory_turns.add(int(turn))
        turns.append(max(trajectory_turns) if trajectory_turns else int(run.get("turn_count", 1) or 1))
        category = str(run.get("category", "unknown"))
        counter = categories.setdefault(category, Counter())
        counter["total"] += 1
        counter[str(audit.get("decision", "unreviewed"))] += 1

    decisions = Counter(str(row.get("decision", "unreviewed")) for row in audit_rows)
    total = len(runs)
    completed = sum(row.get("status") == "completed" for row in runs)
    rejected = decisions["reject"]
    report = {
        "schema_version": "teacher-benchmark-report.v1",
        "source": str(root),
        "dataset_policy": "benchmark prompts and trajectories must remain uncorrected and excluded from SFT",
        "counts": {
            "total": total, "completed": completed,
            "run_errors": total - completed,
            "accept": decisions["accept"],
            "semantic_review_required": decisions["semantic_review_required"],
            "reject": rejected,
        },
        "quality": {
            "strict_accept_rate": round(decisions["accept"] / total, 6) if total else 0.0,
            "non_reject_rate": round((total - rejected) / total, 6) if total else 0.0,
            "run_error_rate": round((total - completed) / total, 6) if total else 0.0,
            "audit_reject_rate": round(rejected / total, 6) if total else 0.0,
            "tool_error_count": tool_errors, "state_error_count": state_errors,
        },
        "turns": {
            "mean": round(sum(turns) / len(turns), 6) if turns else None,
            "p50": _percentile([float(value) for value in turns], 0.5),
            "p90": _percentile([float(value) for value in turns], 0.9),
            "max": max(turns) if turns else None,
        },
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 6) if latencies else None,
            "p50": _percentile(latencies, 0.5), "p90": _percentile(latencies, 0.9),
            "max": round(max(latencies), 6) if latencies else None,
        },
        "tokens": {
            **dict(token_totals), "peak_prompt_tokens": peak_prompt_tokens,
            "context_window_tokens": 8192,
            "context_overflow_cases": context_overflow_cases,
            "provider_total_overflow_cases": provider_total_overflow_cases,
        },
        "by_category": {key: dict(value) for key, value in sorted(categories.items())},
        "interpretation": {
            "result_quality": "strict_accept_rate uses deterministic audit; medical semantic quality remains a separate human-review dimension",
            "error_rate": "report both execution errors and audit rejection; do not merge them",
        },
    }
    if semantic_review is not None:
        reviewed = {
            str(row.get("id")): row for row in semantic_review.get("cases", [])
        }
        approved = sum(row.get("decision") == "approve" for row in reviewed.values())
        human_rejected = sum(row.get("decision") == "reject" for row in reviewed.values())
        unresolved = sum(
            row.get("decision") == "semantic_review_required"
            and str(row.get("id")) not in reviewed for row in audit_rows
        )
        report["human_semantic_quality"] = {
            "reviewed_medical_cases": len(reviewed),
            "approved": approved, "rejected": human_rejected,
            "unresolved_semantic_reviews": unresolved,
            "overall_pass_count": decisions["accept"] + approved,
            "overall_pass_rate": round(
                (decisions["accept"] + approved) / total, 6
            ) if total else 0.0,
            "policy": "review annotates benchmark quality only; it must not mutate or enter SFT",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总未修正 benchmark 的轮数、延迟、质量与错误率")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="")
    parser.add_argument("--semantic-review-file", default="")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    output = Path(args.output).resolve() if args.output else root / "benchmark_report.json"
    semantic_review = (
        json.loads(Path(args.semantic_review_file).read_text(encoding="utf-8"))
        if args.semantic_review_file else None
    )
    output.write_text(json.dumps(
        build_report(root, semantic_review), ensure_ascii=False, indent=2
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
