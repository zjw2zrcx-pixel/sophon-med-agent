"""Export completed, overflow and error case indexes from a Teacher run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(root: Path) -> dict:
    payload = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    rows = payload.get("runs", [])
    completed = []
    overflow = []
    errors = []
    for row in rows:
        usage = row.get("token_usage") or {}
        item = {
            "id": row.get("id"), "category": row.get("category"),
            "status": row.get("status"),
            "context_overflow": bool(usage.get("context_overflow")),
            "provider_total_overflow_calls": int(
                usage.get("provider_total_overflow_calls", 0) or 0
            ),
            "error": row.get("error"),
            "session_end_reason": row.get("session_end_reason"),
            "token_usage": usage,
        }
        if row.get("status") == "completed":
            completed.append(item)
        if item["context_overflow"]:
            overflow.append(item)
        if row.get("status") != "completed":
            errors.append(item)
    result = {
        "schema_version": "teacher-run-sets.v1",
        "source": str(root),
        "counts": {
            "total": len(rows), "completed": len(completed),
            "errors": len(errors), "context_overflow": len(overflow),
        },
        "completed": completed,
        "context_overflow": overflow,
        "errors": errors,
    }
    (root / "run_sets.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Teacher 运行分组清单")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    result = export(Path(args.run_dir).resolve())
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
