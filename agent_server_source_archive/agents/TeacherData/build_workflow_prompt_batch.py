"""Assemble a deterministic, answer-free workflow prompt bank.

Medical rows are hydrated from the read-only facts table; no medical answer is
generated here.  The resulting file is suitable for the trajectory generator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .medical_prompts import MIXED_SAFE_TARGETS, MedicalPromptSampler


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt(row: dict[str, Any]) -> str:
    value = row.get("prompt") or row.get("question")
    if not value and row.get("turns"):
        value = row["turns"][0]
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not value:
        raise ValueError(f"缺少 prompt/question: {row.get('id', row.get('case_id'))}")
    return value


def _medical_rows(path: Path, database: Path, medical_count: int, mixed_count: int) -> list[dict[str, Any]]:
    source_rows = _rows(path)
    need = medical_count + mixed_count
    if len(source_rows) < need:
        raise ValueError(f"targeted 场景不足: 需要 {need}，实际 {len(source_rows)}")
    out: list[dict[str, Any]] = []
    seen_source: set[int] = set()
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        for index, original in enumerate(source_rows[:need]):
            source = original.get("source") or {}
            row_id = int(source.get("id", source.get("row_id", 0)) or 0)
            if not row_id or row_id in seen_source:
                raise ValueError(f"facts row_id 非法或重复: {row_id}")
            seen_source.add(row_id)
            found = db.execute(
                "SELECT subject,aspect,answer,source FROM facts WHERE id=?", (row_id,)
            ).fetchone()
            if not found:
                raise ValueError(f"facts/{row_id} 不存在")
            subject, aspect, answer, dataset = (str(value).strip() for value in found)
            if not dataset.endswith("/train"):
                raise ValueError(f"facts/{row_id} 不是 train 来源")
            declared = str(source.get("answer_sha256", ""))
            actual = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            if declared and declared != actual:
                raise ValueError(f"facts/{row_id} answer_sha256 不匹配")
            question = f"{subject}的{aspect}是什么？"
            category = "medical" if index < medical_count else "mixed"
            case_id = f"workflow-{category}-{original.get('case_id', original.get('id', index + 1))}"
            case: dict[str, Any] = {
                "id": case_id,
                "category": category,
                "prompt": question,
                "turns": [question],
                "difficulty": "medium",
                "risk_tags": ["evidence_grounded"],
                "medical_source": {
                    "database_table": "facts", "row_id": row_id,
                    "source": dataset, "question": question,
                    "reference_answer": answer[:600], "answer_sha256": actual,
                },
                "expected": {
                    "required_tools": ["medical_consult"],
                    "forbidden_tools": ["navigate"],
                },
            }
            if category == "mixed":
                case["expected"]["required_tools"] = ["medical_consult", "navigate"]
                case["expected"]["forbidden_tools"] = []
                case["expected"]["navigation_target"] = MIXED_SAFE_TARGETS[row_id % len(MIXED_SAFE_TARGETS)]
            out.append(case)
    # This is deliberately the existing normalizer: it deterministically
    # rebuilds mixed prompts and labels from the hydrated source metadata.
    return MedicalPromptSampler.normalize_existing_cases(out)


def _simple_rows(
    path: Path, category: str, prefix: str, count: int | None = None,
) -> list[dict[str, Any]]:
    out = []
    source_rows = _rows(path)
    if count is not None:
        if count < 0 or count > len(source_rows):
            raise ValueError(f"{category} count 超出范围: {count}/{len(source_rows)}")
        if category == "general":
            tool_rows = [
                row for row in source_rows
                if (row.get("expected") or {}).get("required_tools")
            ]
            plain_rows = [row for row in source_rows if row not in tool_rows]
            # Keep scarce time/system-tool decisions, then fill with stable
            # no-tool knowledge tasks.  Preserve the authored order in each.
            source_rows = (tool_rows + plain_rows)[:count]
        else:
            source_rows = source_rows[:count]
    for index, original in enumerate(source_rows, 1):
        case = copy.deepcopy(original)
        raw_id = case.get("id", case.get("case_id", index))
        case["id"] = f"workflow-{prefix}-{raw_id}"
        case["category"] = category
        case["prompt"] = _prompt(case)
        case["turns"] = [case["prompt"]]
        expected = case.setdefault("expected", {})
        expected["required_tools"] = ["navigate"] if category == "navigation" else list(expected.get("required_tools", []))
        if category == "navigation" and not expected.get("navigation_target"):
            raise ValueError(f"navigation 缺少 navigation_target: {case['id']}")
        out.append(case)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    medical = _medical_rows(Path(args.scenarios), Path(args.database), args.medical_count, args.mixed_count)
    cases = (
        medical
        + _simple_rows(
            Path(args.navigation), "navigation", "nav", args.navigation_count,
        )
        + _simple_rows(Path(args.general), "general", "gen", args.general_count)
    )
    ids = [str(c["id"]) for c in cases]
    prompts = [re.sub(r"\s+", " ", str(c["prompt"])).strip() for c in cases]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("case id 或规范化 prompt 重复")
    allowed = {"medical_consult", "navigate", "query", "speak", "get_time", "get_system_stats"}
    for case in cases:
        tools = case.get("expected", {}).get("required_tools", [])
        if any(tool not in allowed for tool in tools):
            raise ValueError(f"非法工具: {case['id']}")
        if case["category"] == "navigation" and len(tools) != 1:
            raise ValueError(f"navigation required_tools 必须仅含 navigate: {case['id']}")
    payload = {"schema_version": "teacher-prompts.v1", "generation": {"generator": "build_workflow_prompt_batch", "answer_free": True}, "cases": cases}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": "workflow-prompt-manifest.v1", "output": str(output), "counts": {k: sum(c["category"] == k for c in cases) for k in ("medical", "mixed", "navigation", "general")}, "case_count": len(cases), "case_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    output.with_name("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="构建医疗/混合/导航/通用统一 prompts.json（不生成医学答案）")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--navigation", required=True)
    parser.add_argument("--general", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--medical-count", type=int, required=True)
    parser.add_argument("--mixed-count", type=int, required=True)
    parser.add_argument("--navigation-count", type=int)
    parser.add_argument("--general-count", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.medical_count < 0 or args.mixed_count < 0:
        parser.error("count 不能为负数")
    print(json.dumps(build(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
