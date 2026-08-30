"""Evidence-only semantic review for scalable medical Teacher trajectories."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx


SYSTEM = """你是医疗问答训练数据的证据对齐审阅器。你不是重新回答问题，只判断候选回答能否作为 SFT 正样本。
只允许使用输入中的 MEDICAL_EVIDENCE 和 EXECUTED_NAVIGATION 作为医疗/导航事实依据。请先判断用户问题的意图，再逐项核对回答。

可以 approve 的情况：
1. 回答确实回答了用户问题，核心疾病/药物/症状与证据中的 subject、question 或 normalized_terms 对齐；
2. 核心结论能在一条或多条证据的 text/message/associations 中找到直接支持，允许对同一证据做简洁改写和合并；
3. 证据有限时，回答明确说“资料有限/不能替代医生/建议就医”，而不是编造结论；这类谨慎建议本身不需要额外证据；
4. “可能、常见、可见、需要医生评估”等不确定表达只要不超出证据即可；
5. mixed 任务只有在 EXECUTED_NAVIGATION 包含目标地点时，才可 approve 导航部分。

必须 reject 的情况：
1. 添加证据没有支持的药名、剂量、疗程、温度、治疗方案、检查、病因、症状、风险、科室或确定诊断；
2. 把检索到的相邻疾病/不同 subject 的证据当作当前疾病结论，或把搜索跑偏后的内容包装成答案；
3. 回答遗漏用户的核心问题、给出危险的自行处理步骤，或在证据不足时装作确定；
4. mixed 任务声称已经导航，但 EXECUTED_NAVIGATION 没有相应目标；
5. 明显与证据矛盾。

不要因为答案很长、语言流畅或包含普通免责声明而放宽标准。对于“资料不足并建议专业医生”的短答案，如果没有虚构事实，应 approve。只输出 JSON：
{"decision":"approve|reject","confidence":0到1,"reason_codes":["..."],"notes":"简短中文理由"}。"""


def _compact_medical(value: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    used = 0
    for row in value.get("evidence", []):
        compact = {
            key: row.get(key) for key in
            ("type", "aspect", "subject", "question", "text", "source")
            if row.get(key) not in (None, "")
        }
        encoded = json.dumps(compact, ensure_ascii=False)
        if used + len(encoded) > 14000:
            break
        evidence.append(compact)
        used += len(encoded)
    return {
        "status": value.get("status"), "intent": value.get("intent"),
        "message": value.get("message"), "normalized_terms": value.get("normalized_terms", []),
        "red_flags": value.get("red_flags", []), "departments": value.get("departments", []),
        "associations": value.get("associations", []), "evidence": evidence,
    }


def _case_payload(root: Path, audit_row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    medical = []
    for relative in audit_row.get("trajectory_files", []):
        trajectory = json.loads((root / relative).read_text("utf-8"))
        fact = (trajectory.get("execution_state") or {}).get("facts", {}).get(
            "medical.consultation"
        )
        value = fact.get("value") if isinstance(fact, dict) else None
        if isinstance(value, dict):
            compact = _compact_medical(value)
            if compact not in medical:
                medical.append(compact)
    navigation = [
        command.get("params", {}).get("target") for command in run.get("commands", [])
        if command.get("name") == "navigate" and command.get("success")
    ]
    return {
        "case_id": audit_row.get("id"), "category": run.get("category"),
        "user_turns": [row.get("input") for row in run.get("turns", [])],
        "final_answer": run.get("final"), "medical_evidence": medical,
        "executed_navigation": navigation,
    }


async def review(
    root: Path, output: Path, workers: int, model: str,
    reasoning_effort: str = "low", max_tokens: int = 2400,
) -> dict[str, Any]:
    audit = json.loads((root / "audit.json").read_text("utf-8"))
    runs_payload = json.loads((root / "runs.json").read_text("utf-8"))
    runs = {str(row.get("id")): row for row in runs_payload.get("runs", [])}
    candidates = [
        row for row in audit.get("cases", [])
        if row.get("decision") == "semantic_review_required"
    ]
    checkpoint = output.with_suffix(".checkpoint.json")
    completed: dict[str, dict] = {}
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text("utf-8"))
        completed = {str(row["id"]): row for row in saved.get("cases", [])}
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY 未定义")
    gate = asyncio.Semaphore(workers)
    write_lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=300.0) as client:
        async def one(row: dict[str, Any]) -> dict[str, Any]:
            case_id = str(row["id"])
            if case_id in completed:
                return completed[case_id]
            payload = _case_payload(root, row, runs[case_id])
            body = {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM}, {
                    "role": "user", "content": json.dumps(payload, ensure_ascii=False),
                }],
                "thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort,
                "response_format": {"type": "json_object"}, "stream": False,
                # V4 Pro counts hidden reasoning in the same budget.  Leave
                # enough room for the short visible JSON verdict.
                "max_tokens": max_tokens,
            }
            last_error = ""
            result = None
            for attempt in range(3):
                try:
                    async with gate:
                        response = await client.post(
                            "https://api.deepseek.com/chat/completions",
                            headers={"Authorization": f"Bearer {key}"}, json=body,
                        )
                    response.raise_for_status()
                    data = response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    break
                except Exception as exc:  # checkpointed batch; preserve error detail
                    last_error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(2 ** attempt)
            if not isinstance(result, dict):
                result = {"decision": "reject", "confidence": 0,
                          "reason_codes": ["REVIEW_ERROR"], "notes": last_error}
            decision = str(result.get("decision", "reject"))
            confidence = float(result.get("confidence", 0) or 0)
            if decision != "approve" or confidence < 0.8:
                decision = "reject"
            reviewed = {
                "id": case_id, "decision": decision, "confidence": confidence,
                "reason_codes": list(result.get("reason_codes") or []),
                "notes": str(result.get("notes", "")), "review_model": model,
                "review_method": "evidence_only_llm_judge_v2",
            }
            async with write_lock:
                completed[case_id] = reviewed
                checkpoint.write_text(json.dumps({
                    "schema_version": "teacher-medical-semantic-review.v2",
                    "cases": [completed[key] for key in sorted(completed)],
                }, ensure_ascii=False, indent=2), "utf-8")
            return reviewed

        await asyncio.gather(*(one(row) for row in candidates))
    cases = [completed[str(row["id"])] for row in candidates]
    result = {
        "schema_version": "teacher-medical-semantic-review.v2",
        "reviewer": model, "policy": "evidence_only_conservative_v2",
        "human_spot_review_required": True,
        "summary": {
            "reviewed": len(cases),
            "approve": sum(row["decision"] == "approve" for row in cases),
            "reject": sum(row["decision"] != "approve" for row in cases),
        },
        "cases": cases,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="使用独立 DeepSeek 审阅医疗轨迹证据对齐")
    parser.add_argument("run_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--reasoning-effort", default="low", choices=("low", "high", "max")
    )
    parser.add_argument("--max-tokens", type=int, default=2400)
    args = parser.parse_args()
    asyncio.run(review(
        Path(args.run_dir).resolve(), Path(args.output).resolve(), args.workers,
        args.model, args.reasoning_effort, args.max_tokens,
    ))


if __name__ == "__main__":
    main()
