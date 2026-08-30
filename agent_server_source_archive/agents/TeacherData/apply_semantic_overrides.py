"""Apply auditable human overrides to an LLM semantic-review file."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="合并语义审阅与人工 override")
    parser.add_argument("--review", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review = json.loads(Path(args.review).read_text("utf-8"))
    overrides_payload = json.loads(Path(args.overrides).read_text("utf-8"))
    overrides = {str(row["id"]): row for row in overrides_payload.get("cases", [])}
    cases = []
    for source in review.get("cases", []):
        row = dict(source)
        override = overrides.get(str(row["id"]))
        if override:
            row["original_decision"] = row.get("decision")
            row["original_notes"] = row.get("notes")
            row["decision"] = override["decision"]
            row["notes"] = override["notes"]
            row["human_override"] = True
            row["human_reviewer"] = overrides_payload.get("reviewer")
        cases.append(row)
    counts = Counter(row.get("decision") for row in cases)
    output = {
        **{key: value for key, value in review.items() if key not in {"cases", "summary"}},
        "schema_version": "teacher-medical-semantic-review.v3",
        "human_overrides_source": str(Path(args.overrides).resolve()),
        "summary": {"reviewed": len(cases), "approve": counts["approve"],
                    "reject": counts["reject"], "human_overrides": len(overrides)},
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
