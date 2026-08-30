"""Compile case-level augmented surfaces into PLAN/ACT rows.

Every variant is rendered from the same root SFT decision rows.  The root
trajectory is not copied or edited; provenance records its immutable hash and
the compiler only changes the external user surface plus exact medical query
argument text where required by the tool contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from agents.TeacherData.augment_pipeline import _canonical, _replace_surface, _sha


def main() -> None:
    parser = argparse.ArgumentParser(description="将 case-level semantic/ASR 变体编译为独立 SFT pilot")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--augmentation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root_dir = Path(args.root_dir).resolve()
    aug_dir = Path(args.augmentation_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root_prompts = {
        str(row["id"]): row
        for row in json.loads((root_dir / "prompts.json").read_text(encoding="utf-8")).get("cases", [])
    }
    root_rows: dict[str, list[dict[str, Any]]] = {}
    for line in (root_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        root_rows.setdefault(str(row.get("case_id")), []).append(row)
    variants: list[dict[str, Any]] = []
    for line in (aug_dir / "augmented_cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            variants.append(json.loads(line))
    lineage = {
        str(row["case_id"]): row
        for line in (aug_dir / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() for row in [json.loads(line)]
    }
    compiled: list[dict[str, Any]] = []
    missing_roots: list[str] = []
    for variant in variants:
        variant_id = str(variant["case_id"])
        root_id = str(variant["root_case_id"])
        rows = root_rows.get(root_id, [])
        root_case = root_prompts.get(root_id)
        if not rows or not root_case:
            missing_roots.append(root_id)
            continue
        variant_turns = list(variant.get("turns") or [variant.get("prompt", "")])
        variant_lineage = lineage.get(variant_id, {
            "case_id": variant_id, "root_case_id": root_id,
            "variant_type": variant.get("variant_type", "original"),
            "trajectory_hash": variant.get("trajectory_hash"),
        })
        for row in rows:
            if variant_id == root_id:
                rendered = json.loads(json.dumps(row, ensure_ascii=False))
                rendered.setdefault("tags", {})["augmentation"] = variant_lineage
                rendered.setdefault("provenance", {})["augmentation"] = variant_lineage
                rendered["sample_sha256"] = _sha({k: v for k, v in rendered.items() if k != "sample_sha256"})
            else:
                rendered = _replace_surface(
                    row, root_case, variant_turns, variant_id,
                    str(variant_lineage.get("trajectory_hash") or ""), variant_lineage,
                )
            compiled.append(rendered)
    if missing_roots:
        raise RuntimeError(f"variants refer to missing root SFT cases: {sorted(set(missing_roots))[:10]}")
    # A repeated sample can only happen if the source SFT itself repeats a
    # hash; deduplicate after rendering while retaining deterministic order.
    by_hash = {str(row["sample_sha256"]): row for row in compiled}
    compiled = sorted(by_hash.values(), key=lambda row: (
        str(row.get("case_id", "")), int(row.get("external_turn") or 0),
        int(row.get("agent_iteration") or 0), str(row.get("sample_sha256", "")),
    ))
    (output / "train.jsonl").write_text(
        "".join(_canonical(row) + "\n" for row in compiled), encoding="utf-8"
    )
    (output / "sft_decisions.jsonl").write_text(
        "".join(_canonical(row) + "\n" for row in compiled), encoding="utf-8"
    )
    case_index: dict[str, dict[str, Any]] = {}
    for row in compiled:
        case_id = str(row["case_id"])
        item = case_index.setdefault(case_id, {
            "case_id": case_id, "root_case_id": str(row.get("original_case_id", case_id)),
            "category": row.get("category"), "tags": row.get("tags") or {},
            "sample_count": 0, "actions": Counter(), "external_turns": set(),
        })
        item["sample_count"] += 1
        item["actions"][str(row.get("action"))] += 1
        item["external_turns"].add(int(row.get("external_turn") or 0))
    with (output / "case_index.jsonl").open("w", encoding="utf-8") as handle:
        for case_id in sorted(case_index):
            row = case_index[case_id]
            row["actions"] = dict(row["actions"])
            row["external_turns"] = sorted(row["external_turns"])
            handle.write(_canonical(row) + "\n")
    metadata = json.loads((aug_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "agent-sft-augmented.v1",
        "root_dir": str(root_dir), "augmentation_dir": str(aug_dir),
        "case_count": len(case_index), "sample_count": len(compiled),
        "actions": dict(Counter(str(row.get("action")) for row in compiled)),
        "variant_type_cases": dict(Counter(
            str(((row.get("tags") or {}).get("augmentation") or {}).get("variant_type", "unknown"))
            for row in case_index.values()
        )),
        "sampling_policy": "root_case_uniform_then_variant",
        "trajectory_reuse": True,
        "trajectory_hashes_preserved": True,
        "reasoning_removed": True,
        "source_augmentation_metadata": metadata,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
