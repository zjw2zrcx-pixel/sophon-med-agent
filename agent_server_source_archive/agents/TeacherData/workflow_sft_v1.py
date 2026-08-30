"""Offline workflow materializer for the four-category prompt bank.

This deliberately does not call a model or move a robot.  It runs the local
read-only adapters, applies their observations to :class:`ExecutionState`,
and emits one teacher request for each plan/act decision point.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from agents.Harness.state import ExecutionState, PlanStep, TaskPlan, ToolMetadata
from agents.Harness.state import project_medical_consultation
from agents.Modes.benchmark import BenchmarkMode
from agents.TeacherData.targeted_sft_v3 import _call_preloaded_medical_tool_sync
from agents.agent import Agent, AgentConfig


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tool_result(case: dict[str, Any], tool: str, args: dict[str, Any], medical_tool: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (facts, observation) from deterministic local adapters."""
    if tool == "medical_consult":
        if medical_tool is None:
            raise RuntimeError("medical_consult 必须使用本地工具实例")
        result = _call_preloaded_medical_tool_sync(medical_tool, str(args["query"]))
        if not result.success:
            raise RuntimeError(f"medical_consult failed: {result.error_type}")
        raw = result.facts.get("medical.consultation", {})
        projected = project_medical_consultation(raw)
        return ({"medical.consultation": projected,
                 "dialogue.followup_required": bool(projected.get("followup_required")),
                 "dialogue.followup_questions": projected.get("followup_questions", [])}, projected)
    if tool == "navigate":
        target = str(args.get("target", ""))
        return ({"navigation.status": "navigating", "navigation.target": target}, {"status": "navigating", "target": target, "adapter": "dummy_hospital_profile"})
    if tool == "get_time":
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        return ({"system.time": now}, {"status": "ok", "time": now, "adapter": "local_clock"})
    if tool == "get_system_stats":
        usage = shutil.disk_usage("/")
        value = {"disk_total": usage.total, "disk_used": usage.used, "disk_free": usage.free}
        return ({"system.stats": value}, {"status": "ok", "stats": value, "adapter": "local_system"})
    if tool == "speak":
        text = str(args.get("text", ""))
        return ({"speech.last_text": text}, {"status": "ok", "text": text, "adapter": "local_speak"})
    raise ValueError(f"不支持的离线工具: {tool}")


def _plan(case: dict[str, Any], mode: BenchmarkMode) -> TaskPlan:
    payload = mode.get_compact_workflow_plan(str(case.get("prompt", "")))
    if payload is None:
        raise ValueError(f"BenchmarkMode 无法为 case 构造 compact plan: {case.get('id')}")
    plan = TaskPlan.from_payload(payload)
    category = str(case.get("category", ""))
    required = list((case.get("expected") or {}).get("required_tools", []))
    expected = required + ([] if category == "navigation" else ["speak"])
    observed = [step.preferred_tool for step in plan.steps]
    if category in {"navigation", "general"} and observed != expected:
        steps = tuple(
            PlanStep(
                step_id=f"s{index}", goal=f"执行{tool}",
                preferred_tool=tool, allowed_tools=(tool,),
                depends_on=(f"s{index - 1}",) if index > 1 else (),
            )
            for index, tool in enumerate(expected, 1)
        )
        plan = TaskPlan(
            plan_id=f"corrective-{case['id']}", revision=1,
            goal=f"完成{category}任务",
            goal_description=str(case.get("prompt", "")),
            success_conditions=(), steps=steps,
            done_when="完成预定工具动作并结束当前轮次",
        )
    return plan


def _request(
    case: dict[str, Any], state: ExecutionState, phase: str, allowed: list[str],
    completed: list[str], index: int, mode: BenchmarkMode, include_plan: bool,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    history = [] if not include_plan else [dict(item.__dict__) for item in state.tool_history]
    slots = {"version": "workflow-sft.v1", "system": mode.get_system_prompt(),
             "user": str(case.get("prompt", "")), "history": json.dumps(history, ensure_ascii=False),
             "plan": json.dumps(state.plan.to_dict(), ensure_ascii=False) if include_plan else ""}
    instruction = ("严格遵循当前 decision_frame；只输出一个 <tool> JSON。"
                   "plan 阶段只调用 plan，act 阶段只调用 act；不得输出推理或裸文本。")
    request_id = f"{case['id']}:{index}"
    return {"schema_version": "workflow-teacher-request.v1", "request_id": request_id, "case_id": case["id"],
            "decision_index": index, "phase": phase, "prompt": case["prompt"],
            "instruction": instruction,
            "prompt_slots": slots, "decision_frame": {"current_step_id": state.active_step_id,
            "completed_tools": list(completed), "allowed_tools": list(allowed),
            "forbidden_tools": list(
                (case.get("expected") or {}).get("forbidden_tools", [])
            ),
            "category": case["category"], "expected_plan": state.plan.to_dict(),
            "tool_observations": list(observations)},
            "provenance": {"generator": "workflow_sft_v1", "offline": True}}


def build(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("prompts.json.cases 必须是数组")
    scenarios, requests = [], []
    agent = Agent(AgentConfig(default_mode="Benchmark", benchmark_enabled=True, trajectory_enabled=False, emit_agent_events=False))
    agent.initialize()
    mode = agent.benchmark_mode
    assert mode is not None
    medical_tool = agent.mcp.tools.get("medical_consult")
    if any(str(c.get("category")) in {"medical", "mixed"} for c in cases) and medical_tool is not None:
        medical_tool._retriever(medical_tool._index_path())
    for case in cases:
        if str(case.get("category")) not in {"medical", "mixed", "navigation", "general"}:
            raise ValueError(f"非法 category: {case.get('id')}")
        plan = _plan(case, mode)
        state = ExecutionState.create(plan)
        completed: list[str] = []
        observations = []
        requests.append(_request(
            case, state, "plan", ["plan"], completed, 0, mode, False,
            observations,
        ))
        for i in range(1, len(plan.steps) + 1):
            step = state.active_step
            if step is None:
                break
            tool = step.preferred_tool
            args = {}
            if tool == "medical_consult": args = {"query": case["prompt"]}
            elif tool == "navigate": args = {"action": "start", "target": case["expected"]["navigation_target"]}
            elif tool == "speak": args = {"text": "依据本地工具结果完成本轮答复。"}
            requests.append(_request(
                case, state, "act", [tool], completed, i, mode, True,
                observations,
            ))
            facts, observation = _tool_result(case, tool, args, medical_tool)
            metadata = ToolMetadata(effect="READ", terminal=(tool == "speak"), turn_terminal=(tool == "speak"), session_terminal=(tool == "speak"))
            state.apply_tool_result(tool=tool, arguments=args, metadata=metadata, success=True, facts=facts, observation=observation)
            completed.append(tool)
            observations.append({"tool": tool, "arguments": args, "observation": observation})
        scenarios.append({"schema_version": "workflow-scenario.v1", "case_id": case["id"], "category": case["category"], "prompt": case["prompt"], "medical_context": {"medical_source": case.get("medical_source")} if case["category"] in {"medical", "mixed"} else {}, "expected": {"required_tools": completed}, "decision_frame": {"tool_sequence": completed}, "execution_state": state.to_dict(), "observations": observations, "provenance": {"generator": "workflow_sft_v1", "offline": True, "input_sha256": hashlib.sha256(json.dumps(case, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}})
    return scenarios, requests


def main() -> None:
    p = argparse.ArgumentParser(description="离线构造 workflow SFT scenarios/teacher_requests")
    p.add_argument("--prompts", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    scenarios, requests = build(_read(Path(args.prompts)))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scenarios.jsonl", scenarios), ("teacher_requests.jsonl", requests)):
        (out / name).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(scenarios), "requests": len(requests), "offline": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
