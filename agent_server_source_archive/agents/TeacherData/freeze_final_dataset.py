"""Materialize prompts and audited trajectories for the frozen SFT release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


def _prompt_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in root.glob("*/prompts.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for case in payload.get("cases", []):
            case_id = str(case.get("id", ""))
            if case_id:
                index.setdefault((path.parent.name, case_id), case)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结最终提示词集与对应 audited trajectory")
    parser.add_argument("--sft-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-root", default="teacher_trajectories")
    args = parser.parse_args()
    sft = Path(args.sft_dir).resolve()
    root = Path(args.teacher_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts_out = []
    trajectory_sources = []
    missing_prompts = []
    missing_trajectories = []
    prompt_lookup = _prompt_index(root)
    index_rows = [json.loads(line) for line in (sft / "case_index.jsonl").read_text("utf-8").splitlines() if line.strip()]
    for row in index_rows:
        final_id = str(row["case_id"])
        original_id = str(row.get("original_case_id", ""))
        tags = row.get("tags") or {}
        source_run = str(tags.get("source_run", ""))
        source_path = root / source_run
        prompt = prompt_lookup.get((source_run, original_id))
        if prompt is None:
            missing_prompts.append({"final_case_id": final_id, "source_run": source_run, "original_case_id": original_id})
        else:
            prompt = json.loads(json.dumps(prompt, ensure_ascii=False))
            prompt["id"] = final_id
            prompt["original_case_id"] = original_id
            prompt["final_tags"] = tags
            if not isinstance(prompt.get("intent_metadata"), dict):
                intent = tags.get("intent_metadata")
                if isinstance(intent, dict):
                    prompt["intent_metadata"] = intent
            prompts_out.append(prompt)
        trajectory_dir = source_path / "trajectories" / original_id
        destination = output / "trajectories" / _safe(final_id)
        if trajectory_dir.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(trajectory_dir, destination, dirs_exist_ok=True)
            trajectory_sources.append({
                "final_case_id": final_id, "source_run": source_run,
                "original_case_id": original_id,
                "source": str(trajectory_dir), "materialized": str(destination.relative_to(output)),
            })
        else:
            missing_trajectories.append({"final_case_id": final_id, "source_run": source_run, "original_case_id": original_id, "expected": str(trajectory_dir)})
    if missing_prompts or missing_trajectories:
        raise RuntimeError(json.dumps({"missing_prompts": missing_prompts[:10], "missing_trajectories": missing_trajectories[:10], "missing_prompt_count": len(missing_prompts), "missing_trajectory_count": len(missing_trajectories)}, ensure_ascii=False))
    source_payload = json.loads((sft / "manifest.json").read_text(encoding="utf-8"))
    (output / "prompts.json").write_text(json.dumps({
        "schema_version": "teacher-final-prompts.v1",
        "generation": {
            "frozen_from_sft": str(sft),
            "case_count": len(prompts_out),
            "quality_gate": source_payload.get("quality_gate"),
            "intent_origins": source_payload.get("intent_origins"),
            "category_cases": source_payload.get("category_cases"),
            "conversation_type_cases": source_payload.get("conversation_type_cases"),
        },
        "cases": sorted(prompts_out, key=lambda case: str(case["id"])),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(sft / "manifest.json", output / "sft_manifest.json")
    for filename in ("train.jsonl", "sft_decisions.jsonl", "case_index.jsonl"):
        shutil.copy2(sft / filename, output / filename)
    (output / "trajectory_sources.json").write_text(json.dumps({
        "schema_version": "teacher-final-trajectories.v1",
        "case_count": len(trajectory_sources),
        "sources": trajectory_sources,
        "quality_gate": source_payload.get("quality_gate"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "prompt_cases": len(prompts_out), "trajectory_cases": len(trajectory_sources)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
