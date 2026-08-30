"""Model-visible execution decision envelope; intercepted by the Harness."""
from __future__ import annotations

from typing import Any, Dict

from ..base import Tool, ToolContext, ToolResult


class ActTool(Tool):
    name = "act"
    description = (
        "执行阶段唯一允许的控制调用。选择 CURRENT STEP 的下一项实际动作；"
        "Harness 会验证 step_id、工具、参数、重试和事实状态后再执行。"
    )
    param_schema = {
        "step_id": "必填，最后一个 execution_state_event 中的 current_step_id",
        "action_type": "CALL_TOOL、CALL_SKILL 或 FINISH",
        "tool": "CALL_TOOL 时必填，实际业务工具名",
        "skill": "CALL_SKILL 时必填，实际 Skill 名",
        "arguments": "必填对象，传给实际工具或 Skill 的参数",
        "response": "FINISH 时必填，最终答复文本",
    }
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "READ", "idempotent": False,
        "produces": [], "invalidates": [],
        "retry": {"max_attempts": 1},
    }

    async def call(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        del params, context
        return ToolResult(
            success=False,
            error_type="HARNESS_CONTROL_ONLY",
            error="act 必须由 Harness Controller 解包，不能作为业务工具直接执行。",
        )
