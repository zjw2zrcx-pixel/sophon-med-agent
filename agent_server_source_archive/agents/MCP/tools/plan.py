"""Visible task-planning tool; stores no hidden reasoning."""
from __future__ import annotations

from typing import Any, Dict

from ...Harness import TaskPlan
from ..base import Tool, ToolContext, ToolResult


class PlanTool(Tool):
    name = "plan"
    description = (
        "为当前用户任务建立简短、可执行的显式计划。每个任务只能在第一次调用中使用；"
        "它不是思维链，不要写分析过程。"
    )
    param_schema = {
        "goal": "必填，最终语义目标，不要写成工具调用序列",
        "goal_description": "可选，面向人的目标说明",
        "success_conditions": (
            "可选，事实条件数组；字段为 fact/operator/value/require_valid。operator 只能是 "
            "eq/neq/in/not_in/gt/gte/lt/lte/exists/not_exists；exists/not_exists 不写 value。"
            "条件必须描述完整用户目标，不能把中间医疗检索成功当最终完成。可能 query 或 speak 的"
            "医疗任务建议留空；导航并播报可同时要求 navigation.status eq navigating 与 "
            "speech.last_text exists"
        ),
        "steps": (
            "必填，1到8个实际执行步骤；包含 step_id/goal/preferred_tool/depends_on/"
            "condition/verification。不要增加‘规划任务’或‘结束任务’等无工具空步骤"
        ),
        "done_when": "可选，一句话完成标准",
    }
    modes = ["Voice", "Benchmark"]

    async def call(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        try:
            plan = TaskPlan.from_payload(params)
        except (TypeError, ValueError) as exc:
            return ToolResult(
                success=False,
                error=f"plan 无效: {exc}",
                error_type="INVALID_PLAN",
                retryable=True,
                recovery_hint="请输出1到8个有序步骤；依赖必须引用已有 step_id。",
            )
        value = plan.to_dict()
        if not value["done_when"]:
            value["done_when"] = "满足成功条件或完成全部步骤并向用户给出最终答复"
        import json
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > 6000:
            return ToolResult(
                success=False,
                error="plan 内容过长",
                error_type="INVALID_PLAN",
                retryable=True,
                recovery_hint="只保留完成任务所需的关键步骤。",
            )
        return ToolResult(success=True, data=encoded)
