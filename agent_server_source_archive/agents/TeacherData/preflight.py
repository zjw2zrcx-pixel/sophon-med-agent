"""Preflight prompt cases against the production medical query contract."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any

from agents.MCP.base import ToolContext
from agents.MCP.tools.medconsult import MedicalConsultTool


async def preflight_cases(cases: list[dict[str, Any]], workers: int = 3) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers 必须大于 0")
    tool = MedicalConsultTool(dense_enabled=True)
    gate = asyncio.Semaphore(workers)

    async def inspect(case: dict[str, Any]) -> dict[str, Any]:
        category = str(case.get("category", ""))
        if category not in {"medical", "mixed"}:
            return {"id": case["id"], "decision": "eligible", "checks": []}
        turns = list(case.get("turns") or [case.get("prompt", "")])
        support_replies = [
            str(value).strip()
            for value in (case.get("followup_support") or {}).get("user_replies", [])
            if str(value).strip()
        ]
        checks = []
        async with gate:
            queries = list(turns)
            for turn in queries:
                result = await tool.call({"query": str(turn)}, ToolContext())
                payload = {}
                if result.data:
                    try:
                        payload = json.loads(result.data)
                    except json.JSONDecodeError:
                        payload = {}
                checks.append({
                    "query": str(turn),
                    "tool_success": result.success,
                    "status": payload.get("status"),
                    "intent": payload.get("intent"),
                    "questions": payload.get("questions", []),
                    "dense_used": (payload.get("retrieval") or {}).get("dense_used"),
                    "error": result.error,
                })

            # A nominally single-turn prompt may safely become multi-turn when
            # its database-grounded support contains a pre-authored reply.
            # Materialize at most two replies so the production session limit
            # remains three external user turns.
            if len(turns) == 1:
                while (
                    checks[-1]["status"] in {"need_more_info", "ambiguous"}
                    and support_replies and len(queries) < 3
                ):
                    reply = support_replies.pop(0)
                    queries.append(reply)
                    result = await tool.call({"query": reply}, ToolContext())
                    payload = json.loads(result.data) if result.data else {}
                    checks.append({
                        "query": reply, "tool_success": result.success,
                        "status": payload.get("status"), "intent": payload.get("intent"),
                        "questions": payload.get("questions", []),
                        "dense_used": (payload.get("retrieval") or {}).get("dense_used"),
                        "error": result.error,
                    })
                turns = queries

        reasons: list[str] = []
        if any(not check["tool_success"] for check in checks):
            reasons.append("MEDICAL_TOOL_FAILED")
        terminal_statuses = {"ok", "urgent"}
        if len(turns) == 1:
            if checks[0]["status"] not in terminal_statuses:
                reasons.append(f"SINGLE_TURN_NOT_ANSWERABLE:{checks[0]['status']}")
        else:
            if checks[0]["status"] not in {"need_more_info", "ambiguous"}:
                reasons.append(f"FIRST_TURN_DOES_NOT_REQUEST_FOLLOWUP:{checks[0]['status']}")
            if not checks[0]["questions"]:
                reasons.append("FIRST_TURN_HAS_NO_QUESTION")
            if checks[-1]["status"] not in terminal_statuses:
                reasons.append(f"FINAL_TURN_NOT_ANSWERABLE:{checks[-1]['status']}")
        return {
            "id": case["id"],
            "decision": "eligible" if not reasons else "reject",
            "reasons": reasons,
            "checks": checks,
            "materialized_turns": turns,
        }

    rows = list(await asyncio.gather(*(inspect(case) for case in cases)))
    decisions = {row["id"]: row for row in rows}
    eligible = []
    for source_case in cases:
        row = decisions[source_case["id"]]
        if row["decision"] != "eligible":
            continue
        case = copy.deepcopy(source_case)
        materialized = list(row.get("materialized_turns") or case.get("turns") or [])
        if len(materialized) > 1:
            case["turns"] = materialized
            case["prompt"] = materialized[0]
            required = list((case.get("expected") or {}).get("required_tools", []))
            if "query" not in required:
                required.append("query")
            case.setdefault("expected", {})["required_tools"] = required
            support = case.get("support_data")
            if isinstance(support, dict):
                support["required_tools"] = list(required)
            case["preflight_materialized_followup"] = True
        eligible.append(case)
    counts = Counter(row["decision"] for row in rows)
    return {
        "schema_version": "teacher-preflight.v1",
        "summary": {
            "total": len(rows), "eligible": counts["eligible"], "reject": counts["reject"],
        },
        "cases": rows,
        "eligible_cases": eligible,
    }


async def async_main(args) -> None:
    source_path = Path(args.prompts_file).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cases = source.get("cases")
    if not isinstance(cases, list):
        raise ValueError("prompts_file 缺少 cases")
    result = await preflight_cases(cases, workers=args.workers)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(json.dumps({
        key: value for key, value in result.items() if key != "eligible_cases"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    generation = dict(source.get("generation") or {})
    generation.update({
        "preflight_source": str(source_path),
        "preflight_summary": result["summary"],
    })
    (output_dir / "prompts.json").write_text(json.dumps({
        "schema_version": "teacher-prompts.v1",
        "generation": generation,
        "cases": result["eligible_cases"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用真实医疗工具预检提示词集")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
