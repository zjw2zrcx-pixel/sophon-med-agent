"""Create auditable human-corrected copies of navigation trajectories."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from agents.TeacherData.audit import audit_runs


def _replace_text(value: str, old: str, new: str) -> str:
    patterns = (
        (f'"target":"{old}"', f'"target":"{new}"'),
        (f'"target": "{old}"', f'"target": "{new}"'),
        (f'\\"target\\":\\"{old}\\"', f'\\"target\\":\\"{new}\\"'),
        (f'"navigation.target":"{old}"', f'"navigation.target":"{new}"'),
        (f'"navigation.target": "{old}"', f'"navigation.target": "{new}"'),
        (f'\\"navigation.target\\":\\"{old}\\"',
         f'\\"navigation.target\\":\\"{new}\\"'),
        (f"导航至{old}", f"导航至{new}"),
        (f"导航到{old}", f"导航到{new}"),
        (f"前往{old}", f"前往{new}"),
        (f"带您去{old}", f"带您去{new}"),
        (f"带您前往{old}", f"带您前往{new}"),
        (f"目标为{old}", f"目标为{new}"),
        (f"目标是{old}", f"目标是{new}"),
    )
    for source, target in patterns:
        value = value.replace(source, target)
    return value


def _repair_value(value: Any, old: str, new: str, key: str = "") -> Any:
    if isinstance(value, dict):
        repaired = {}
        for child_key, child_value in value.items():
            if child_key in {"target", "navigation.target"} and child_value == old:
                repaired[child_key] = new
            else:
                repaired[child_key] = _repair_value(child_value, old, new, child_key)
        return repaired
    if isinstance(value, list):
        return [_repair_value(item, old, new, key) for item in value]
    if isinstance(value, str):
        return _replace_text(value, old, new)
    return value


def correct_run(
    run: dict[str, Any], *, expected_target: str, reviewer: str, source: str,
) -> tuple[dict[str, Any], str]:
    navigation = [
        command for turn in run.get("turns", [])
        for command in turn.get("commands", [])
        if command.get("name") == "navigate"
    ]
    if len(navigation) != 1:
        raise ValueError(f"{run.get('id')} 必须恰好包含一次 navigate")
    old_target = str(navigation[0].get("params", {}).get("target", "")).strip()
    if not old_target or old_target == expected_target:
        raise ValueError(f"{run.get('id')} 没有可校正的错误 target")
    repaired = _repair_value(copy.deepcopy(run), old_target, expected_target)
    repaired["correction"] = {
        "type": "human_navigation_target_correction",
        "reviewer": reviewer,
        "source_run": source,
        "old_target": old_target,
        "new_target": expected_target,
        "scope": [
            "plan", "act proposal", "navigate params/result", "execution state",
            "final speak", "model-call history",
        ],
    }
    return repaired, old_target


def main() -> None:
    parser = argparse.ArgumentParser(description="校正唯一确定的导航目标错误并保留来源")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-ids", required=True, help="逗号分隔；只处理目标不一致项")
    parser.add_argument("--reviewer", default="human-review")
    args = parser.parse_args()
    source_root = Path(args.run_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_payload = json.loads((source_root / "runs.json").read_text(encoding="utf-8"))
    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    source_by_id = {str(run["id"]): run for run in source_payload.get("runs", [])}
    missing = sorted(wanted - source_by_id.keys())
    if missing:
        raise ValueError(f"未找到 case: {', '.join(missing)}")

    corrected_rows = []
    corrections = []
    for case_id in sorted(wanted):
        run = source_by_id[case_id]
        expected_target = str(
            (run.get("expected") or {}).get("navigation_target") or ""
        ).strip()
        if not expected_target:
            raise ValueError(f"{case_id} 缺少人工确认的 expected navigation_target")
        corrected, old_target = correct_run(
            run, expected_target=expected_target, reviewer=args.reviewer,
            source=str((source_root / "runs.json").resolve()),
        )
        corrected_rows.append(corrected)
        source_case_dir = source_root / "trajectories" / case_id
        output_case_dir = output_root / "trajectories" / case_id
        output_case_dir.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(source_case_dir.glob("*.json")):
            payload = json.loads(source_file.read_text(encoding="utf-8"))
            repaired = _repair_value(payload, old_target, expected_target)
            repaired["correction"] = corrected["correction"]
            (output_case_dir / source_file.name).write_text(
                json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        corrections.append({
            "id": case_id, "old_target": old_target, "new_target": expected_target,
            "reviewer": args.reviewer,
        })

    source_prompts = source_root / "prompts.json"
    if source_prompts.is_file():
        prompts = json.loads(source_prompts.read_text(encoding="utf-8"))
        prompts["cases"] = [
            case for case in prompts.get("cases", []) if str(case.get("id")) in wanted
        ]
        (output_root / "prompts.json").write_text(
            json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    runs_payload = {
        "schema_version": source_payload.get("schema_version", "teacher-pilot.v1"),
        "teacher_model": source_payload.get("teacher_model"),
        "prompt_model": source_payload.get("prompt_model"),
        "correction_provenance": {
            "type": "human_corrected", "source_dir": str(source_root),
            "reviewer": args.reviewer, "cases": corrections,
        },
        "runs": corrected_rows,
    }
    (output_root / "runs.json").write_text(
        json.dumps(runs_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "audit.json").write_text(
        json.dumps(audit_runs(output_root, runs_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
