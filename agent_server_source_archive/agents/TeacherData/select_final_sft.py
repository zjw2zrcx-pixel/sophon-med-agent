"""Freeze a balanced ~1K case SFT release from reviewed projections."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


TARGET = {
    ("medical", "multi_turn"): 500,
    ("mixed", "single_turn"): 200,
    ("navigation", "single_turn"): 200,
    ("general", "single_turn"): 100,
}


def _rank(case_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结约千条严格审阅通过的平衡 SFT case")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="sft-final-v7")
    args = parser.parse_args()
    root = Path(args.input_dir).resolve()
    index_rows = [json.loads(line) for line in (root / "case_index.jsonl").read_text("utf-8").splitlines() if line.strip()]
    by_group: dict[tuple[str, str], list[dict]] = {}
    for row in index_rows:
        tags = row.get("tags") or {}
        group = (str(row.get("category", "")), str(tags.get("conversation_type", "")))
        by_group.setdefault(group, []).append(row)
    selected_ids: set[str] = set()
    selection: dict[str, int] = {}
    for group, target in TARGET.items():
        pool = sorted(by_group.get(group, []), key=lambda row: _rank(str(row["case_id"]), args.seed))
        if len(pool) < target:
            raise ValueError(f"{group} 可用 case 不足: {len(pool)} < {target}")
        selected_ids.update(str(row["case_id"]) for row in pool[:target])
        selection[f"{group[0]}:{group[1]}"] = target
    if len(selected_ids) != sum(TARGET.values()):
        raise ValueError("选择结果 case id 重复")
    samples = []
    for line in (root / "train.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("case_id")) in selected_ids:
            samples.append(row)
    missing = selected_ids - {str(row.get("case_id")) for row in samples}
    if missing:
        raise ValueError(f"选中 case 没有对应 SFT 样本: {sorted(missing)[:5]}")
    samples.sort(key=lambda row: (str(row["case_id"]), int(row.get("external_turn") or 0), int(row.get("agent_iteration") or 0)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples]
    for filename in ("train.jsonl", "sft_decisions.jsonl"):
        (output / filename).write_text("".join(serialized), encoding="utf-8")
    selected_index = [row for row in index_rows if str(row["case_id"]) in selected_ids]
    selected_index.sort(key=lambda row: str(row["case_id"]))
    (output / "case_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_index),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "agent-sft-final.v1",
        "source": str(root), "selection_seed": args.seed,
        "selection_policy": "medical multi_turn 500 + mixed single_turn 200 + navigation single_turn 200 + general single_turn 100",
        "case_count": len(selected_index), "sample_count": len(samples),
        "actions": dict(Counter(str(row.get("action")) for row in samples)),
        "category_cases": dict(Counter(str(row.get("category")) for row in selected_index)),
        "category_samples": dict(Counter(str(row.get("category")) for row in samples)),
        "conversation_type_cases": dict(Counter(str((row.get("tags") or {}).get("conversation_type")) for row in selected_index)),
        "intent_origins": dict(Counter(str(((row.get("tags") or {}).get("intent_metadata") or {}).get("origin")) for row in selected_index)),
        "selection": selection,
        "reasoning_removed": True,
        "context_overflow_included": False,
        "quality_gate": "only audited accept or evidence-only semantic approve; rejected/error/overflow cases excluded",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
