#!/usr/bin/env python3
"""Materialize a resumable slice of the failure-harvest candidate pool."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()
    source = json.loads(args.dataset.read_text(encoding="utf-8"))
    sessions = list(source.get("sessions") or [])
    if not 0 <= args.index < args.num_batches:
        raise ValueError("index must be within num-batches")
    by_category = defaultdict(list)
    for session in sessions:
        by_category[session["category_key"]].append(session)
    selected = []
    for category in ("medical", "navigation", "general", "mixed"):
        selected.extend(by_category[category][args.index::args.num_batches])
    if not selected:
        raise ValueError("selected batch is empty")
    ids = {item["session_id"] for item in selected}
    manifests = [item for item in source.get("selection_manifest", []) if item["canary_id"] in ids]
    value = {
        **source,
        "benchmark_schema": "failure-harvest-batch.v1",
        "sessions": selected,
        "selection_manifest": manifests,
        "holdout_sessions": [], "holdout_manifest": [],
        "core_external_turns": len(selected),
        "parent_dataset": str(args.dataset),
        "batch": {
            "index": args.index, "num_batches": args.num_batches,
            "count": len(selected),
            "category_counts": {
                category: sum(item["category_key"] == category for item in selected)
                for category in ("medical", "navigation", "general", "mixed")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value["batch"]))


if __name__ == "__main__":
    main()
