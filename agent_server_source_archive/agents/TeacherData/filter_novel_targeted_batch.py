"""Remove semantic-family collisions from a targeted teacher batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_many(paths: Iterable[str]) -> list[dict[str, Any]]:
    return [row for path in paths for row in _read(path)]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--exclude-scenario-dirs", required=True, nargs="+")
    parser.add_argument("--teacher-outputs", required=True, nargs="+")
    parser.add_argument("--semantic-reviews", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scenarios = _read(Path(args.scenario_dir) / "scenarios.jsonl")
    excluded_families = {
        str(row.get("semantic_family_id"))
        for directory in args.exclude_scenario_dirs
        for row in _read(Path(directory) / "scenarios.jsonl")
    }
    kept = [
        row for row in scenarios
        if str(row.get("semantic_family_id")) not in excluded_families
    ]
    kept_ids = {str(row["case_id"]) for row in kept}
    outputs = [
        row for row in _read_many(args.teacher_outputs)
        if str(row.get("case_id")) in kept_ids
    ]
    reviews = [
        row for row in _read_many(args.semantic_reviews)
        if str(row.get("case_id")) in kept_ids
    ]
    if len(kept_ids) != len(kept):
        raise RuntimeError("duplicate case_id in kept scenarios")
    if {str(row.get("case_id")) for row in outputs} != kept_ids:
        raise RuntimeError("teacher outputs do not exactly cover kept scenarios")
    if {str(row.get("case_id")) for row in reviews} != kept_ids:
        raise RuntimeError("semantic reviews do not exactly cover kept scenarios")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "scenarios.jsonl", kept)
    _write(output / "teacher_outputs.jsonl", outputs)
    _write(output / "semantic_reviews.jsonl", reviews)
    report = {
        "schema_version": "targeted-novel-filter-report.v1",
        "input_scenarios": len(scenarios),
        "excluded_semantic_family_collisions": len(scenarios) - len(kept),
        "kept_scenarios": len(kept),
        "teacher_outputs": len(outputs),
        "semantic_reviews": len(reviews),
        "exclude_scenario_dirs": args.exclude_scenario_dirs,
    }
    (output / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
