"""Recompute stored-sequence versus provider token overflow without mutation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.API.api import API


def main() -> None:
    parser = argparse.ArgumentParser(description="重算隐藏思考剔除后的 8K 序列占用")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    api = API(context_window_tokens=8192)
    cases = []
    for directory in sorted((root / "trajectories").iterdir()):
        calls = []
        for path in sorted(directory.glob("[0-9]*_*.json")):
            payload = json.loads(path.read_text("utf-8"))
            for call in payload.get("model_calls", []):
                calls.append(api._context_stats(
                    call.get("usage") or {}, str(call.get("output", ""))
                ))
        cases.append({
            "id": directory.name, "model_call_count": len(calls),
            "peak_prompt_tokens": max((x["prompt_tokens"] for x in calls), default=0),
            "peak_training_sequence_tokens_estimate": max(
                (x["training_sequence_tokens_estimate"] for x in calls), default=0
            ),
            "peak_provider_total_tokens": max(
                (x["provider_total_tokens"] for x in calls), default=0
            ),
            "training_sequence_overflow": any(x["context_overflow"] for x in calls),
            "provider_total_overflow": any(x["provider_total_overflow"] for x in calls),
        })
    report = {
        "schema_version": "context-reanalysis.v1", "source": str(root),
        "context_window_tokens": 8192,
        "method": "provider prompt tokens + conservative visible XML token estimate; hidden reasoning excluded",
        "summary": {
            "case_count": len(cases),
            "training_sequence_overflow_cases": sum(x["training_sequence_overflow"] for x in cases),
            "provider_total_overflow_cases": sum(x["provider_total_overflow"] for x in cases),
            "peak_prompt_tokens": max((x["peak_prompt_tokens"] for x in cases), default=0),
            "peak_training_sequence_tokens_estimate": max(
                (x["peak_training_sequence_tokens_estimate"] for x in cases), default=0
            ),
        },
        "cases": cases,
    }
    output = Path(args.output).resolve() if args.output else root / "context_reanalysis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
