"""Create an auditable derived prompt subset without mutating a fixed bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="按轮数筛选固定提示词集")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-turns", type=int, default=1)
    args = parser.parse_args()
    source_path = Path(args.prompts_file).resolve()
    source = json.loads(source_path.read_text("utf-8"))
    cases = [
        case for case in source.get("cases", [])
        if len(case.get("turns") or [case.get("prompt", "")]) <= args.max_turns
    ]
    generation = dict(source.get("generation") or {})
    generation.update({
        "derived_from": str(source_path), "selection": {"max_turns": args.max_turns},
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selected_count": len(cases),
    })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": source.get("schema_version", "teacher-prompts.v1"),
        "generation": generation, "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
