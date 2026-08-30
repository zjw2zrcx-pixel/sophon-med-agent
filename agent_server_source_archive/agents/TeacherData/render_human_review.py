"""Render targeted SFT candidates as a human-readable review packet."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

from agents.TeacherData.targeted_sft_v3 import parse_target_output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_many(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(Path(path)))
    return rows


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _quoted(value: Any) -> str:
    text = _one_line(value)
    return "> " + (text or "（空）")


def _priority(scenario: dict[str, Any]) -> tuple[str, list[str]]:
    supervision = scenario.get("supervision") or {}
    medical = scenario.get("medical_context") or {}
    aspect = str(supervision.get("source_aspect", ""))
    reasons: list[str] = []
    if supervision.get("evidence_gap"):
        reasons.append("evidence-gap")
    if aspect in {"治疗", "手术治疗"}:
        reasons.append(aspect)
    if medical.get("red_flags") or medical.get("urgency") not in {"", "routine"}:
        reasons.append("高风险/非日常紧急度")
    if medical.get("medication_notice"):
        reasons.append("用药相关")
    return ("必须100%复审", reasons) if reasons else ("普通抽检", [])


def _evidence_lines(medical: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in medical.get("evidence", []):
        evidence_id = item.get("evidence_id", "无ID")
        labels = [str(item.get("type", "evidence"))]
        if item.get("subject"):
            labels.append(f"实体={_one_line(item['subject'])}")
        if item.get("aspect"):
            labels.append(f"方面={_one_line(item['aspect'])}")
        lines.append(f"- `{evidence_id}`（{'；'.join(labels)}）")
        if item.get("question"):
            lines.append(f"  - 文档问题：{_one_line(item['question'])}")
        if item.get("text"):
            lines.append(f"  - 内容：{_one_line(item['text'])}")
        source = _one_line(item.get("source"))
        source_line = item.get("source_line")
        if source or source_line not in (None, ""):
            lines.append(
                f"  - 来源：{source or '未标注'}"
                + (f"，行 {source_line}" if source_line not in (None, "") else "")
            )
    for item in medical.get("associations", []):
        association_id = item.get("association_id", "无ID")
        lines.append(
            f"- `{association_id}`（association）"
            f"{_one_line(item.get('matched'))} —[{_one_line(item.get('relation'))}]→ "
            f"{_one_line(item.get('related'))}"
        )
        if item.get("source"):
            lines.append(f"  - 来源：{_one_line(item['source'])}")
    if not lines:
        lines.append("- 无保留证据；该答案只能明确陈述本地资料缺口，不得补充医学事实。")
    return lines


def _claim_lines(review: dict[str, Any] | None) -> list[str]:
    if not review:
        return ["- 无自动语义审核记录。"]
    mappings = review.get("claim_evidence_map") or []
    if not mappings:
        return ["- 无 claim 映射。"]
    lines: list[str] = []
    for index, item in enumerate(mappings, 1):
        ids = ", ".join(f"`{value}`" for value in item.get("evidence_ids", []))
        lines.append(
            f"- Claim {index}：{_one_line(item.get('claim'))}\n"
            f"  - 类型：`{item.get('kind')}`；支持：`{item.get('support')}`；"
            f"证据：{ids or '无（保守策略）'}"
        )
    return lines


def render_packet(
    scenarios: list[dict[str, Any]], outputs: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    output_by_id = {str(row.get("case_id")): row for row in outputs}
    review_by_id = {str(row.get("case_id")): row for row in reviews}
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        output = output_by_id.get(case_id)
        if output is None:
            continue
        parsed, errors = parse_target_output(str(output.get("output", "")))
        answer = parsed["text"] if parsed and not errors else str(output.get("output", ""))
        review = review_by_id.get(case_id)
        priority, reasons = _priority(scenario)
        records.append({
            "scenario": scenario, "answer": answer, "review": review,
            "priority": priority, "priority_reasons": reasons,
            "automated_status": (
                "approved" if review and review.get("decision") == "approve" else "rejected"
            ),
        })
    records.sort(key=lambda row: (
        0 if row["priority"] == "必须100%复审" else 1,
        0 if row["automated_status"] == "rejected" else 1,
        str(row["scenario"]["case_id"]),
    ))
    counts = Counter(row["automated_status"] for row in records)
    priorities = Counter(row["priority"] for row in records)
    lines = [
        "# GPT Luna Batch 0 人工医学复审包",
        "",
        "本文件按证据审核，不要求审阅者补充外部医学知识。请判断 Luna 答案是否严格回答当前疾病和方面，且每个事实都被下方证据直接支持。",
        "",
        "## 汇总",
        "",
        f"- 待复审案例：{len(records)}",
        f"- 自动语义审核通过：{counts['approved']}；自动拒绝：{counts['rejected']}",
        f"- 必须 100% 复审：{priorities['必须100%复审']}；普通抽检：{priorities['普通抽检']}",
        "- 人工决定只能填写：`approve`、`reject` 或 `edit`。选择 edit 时必须填写 corrected_text。",
        "- 自动拒绝案例保留在包内用于核对，但默认不会进入编译训练集。",
        "",
        "## 通用检查项",
        "",
        "1. 疾病/对象是否与用户问题完全一致？",
        "2. 是否只回答 requested aspect，没有用邻近方面填充？",
        "3. 每个医学事实能否由列出的 evidence/association 直接支持？",
        "4. 是否添加了证据外疾病、药物、剂量、疗程、检查或科室？",
        "5. gap 回答是否只说明本地资料未覆盖，没有暗示不存在该医学事实？",
        "6. 表达是否简短、完整、无夸大且适合语音播报？",
        "",
    ]
    current_section = ""
    template: list[dict[str, Any]] = []
    for position, record in enumerate(records, 1):
        scenario = record["scenario"]
        medical = scenario.get("medical_context") or {}
        supervision = scenario.get("supervision") or {}
        review = record["review"]
        if record["priority"] != current_section:
            current_section = record["priority"]
            lines.extend([f"## {current_section}", ""])
        reasons = "、".join(record["priority_reasons"]) or "普通 grounded"
        lines.extend([
            f"### {position}. `{scenario['case_id']}`",
            "",
            f"- 自动状态：`{record['automated_status']}`",
            f"- 复审级别：{record['priority']}（{reasons}）",
            f"- split：`{scenario.get('split')}`；源方面：`{supervision.get('source_aspect')}`；"
            f"检索 intent：`{supervision.get('intent_aspect')}`；evidence-gap：`{str(bool(supervision.get('evidence_gap'))).lower()}`",
            f"- 源实体：{_one_line((scenario.get('source') or {}).get('subject'))}",
            "",
            "用户问题：",
            "",
            _quoted(scenario.get("prompt")),
            "",
            "Luna 候选答案：",
            "",
            _quoted(record["answer"]),
            "",
            "本地证据：",
            "",
            *_evidence_lines(medical),
            "",
            "自动 claim→evidence 审核：",
            "",
            f"- 决定：`{review.get('decision') if review else 'missing'}`",
            *(
                [f"- 错误：{', '.join(review.get('errors', []))}"]
                if review and review.get("errors") else []
            ),
            *_claim_lines(review),
            "",
            "人工复审：",
            "",
            "- [ ] 实体一致",
            "- [ ] aspect 一致",
            "- [ ] 所有事实均有直接证据",
            "- [ ] 无证据外医学内容",
            "- [ ] 安全、简短、完整",
            "- 决定：`pending`",
            "- 修改后答案（仅 edit 时）：",
            "- 备注：",
            "",
            "---",
            "",
        ])
        template.append({
            "schema_version": "targeted-human-review.v1",
            "case_id": scenario["case_id"],
            "automated_status": record["automated_status"],
            "review_priority": record["priority"],
            "human_decision": "pending",
            "checks": {
                "entity_correct": None,
                "aspect_correct": None,
                "all_claims_supported": None,
                "no_external_medical_content": None,
                "safe_concise_complete": None,
            },
            "corrected_text": "",
            "comments": "",
            "reviewer": "",
            "reviewed_at": "",
        })
    return "\n".join(lines), template


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染 GPT Luna 人工医学复审包")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--teacher-outputs", required=True, nargs="+")
    parser.add_argument("--semantic-reviews", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    scenario_dir = Path(args.scenario_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet, template = render_packet(
        _read_jsonl(scenario_dir / "scenarios.jsonl"),
        _read_many(args.teacher_outputs),
        _read_many(args.semantic_reviews),
    )
    (output_dir / "HUMAN_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (output_dir / "human_review_decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in template),
        encoding="utf-8",
    )
    print(json.dumps({
        "case_count": len(template),
        "packet": str(output_dir / "HUMAN_REVIEW_PACKET.md"),
        "decision_template": str(output_dir / "human_review_decisions.jsonl"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
