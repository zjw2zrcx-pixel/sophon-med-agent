"""Apply deterministic runtime gates after an LLM semantic review."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def filter_review(review_path: Path, run_dir: Path, output: Path) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    runs = {
        str(row.get("id")): row
        for row in json.loads((run_dir / "runs.json").read_text(encoding="utf-8")).get("runs", [])
    }
    prompts = {
        str(row.get("id")): row
        for row in json.loads((run_dir / "prompts.json").read_text(encoding="utf-8")).get("cases", [])
    }
    approved = []
    rejected = []
    reasons = collections.Counter()
    for row in review.get("cases", []):
        if row.get("decision") != "approve":
            rejected.append(row)
            continue
        case_id = str(row.get("id"))
        run = runs.get(case_id, {})
        prompt = prompts.get(case_id, {})
        reject_reasons = []
        if run.get("status") != "completed":
            reject_reasons.append("RUN_NOT_COMPLETED")
        usage = run.get("token_usage") or {}
        if usage.get("context_overflow"):
            reject_reasons.append("CONTEXT_OVERFLOW")
        if not str(run.get("final", "")).strip():
            reject_reasons.append("EMPTY_FINAL")
        if prompt.get("category") == "mixed":
            target = str((prompt.get("expected") or {}).get("navigation_target") or "")
            executed = {
                str(command.get("params", {}).get("target", ""))
                for command in run.get("commands", [])
                if command.get("name") == "navigate" and command.get("success")
            }
            if target not in executed:
                reject_reasons.append("NAVIGATION_NOT_EXECUTED")
        if reject_reasons:
            rejected.append({
                **row, "decision": "reject",
                "deterministic_reject_reasons": reject_reasons,
            })
            for reason in reject_reasons:
                reasons[reason] += 1
        else:
            approved.append(row)
    result = {
        "schema_version": "teacher-medical-semantic-review.v2-filtered",
        "reviewer": review.get("reviewer"),
        "policy": "evidence_only_conservative_v2_plus_runtime_gates",
        "source_review": str(review_path.resolve()),
        "human_spot_review_required": True,
        "summary": {
            "reviewed": len(review.get("cases", [])),
            "llm_approve": sum(row.get("decision") == "approve" for row in review.get("cases", [])),
            "deterministic_approve": len(approved),
            "reject": len(review.get("cases", [])) - len(approved),
            "deterministic_reject_reasons": dict(reasons),
        },
        "cases": sorted(approved + rejected, key=lambda row: str(row.get("id"))),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="过滤语义复审结果中的运行时错误样本")
    parser.add_argument("--review", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = filter_review(
        Path(args.review).resolve(), Path(args.run_dir).resolve(), Path(args.output).resolve()
    )
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
