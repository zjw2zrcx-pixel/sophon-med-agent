from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class MCPManager:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.ros_bridge: Optional[object] = None

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name} (modes: {tool.modes})")

    def register_many(self, tools: List[Tool]):
        for tool in tools:
            self.register(tool)

    def get_tool_description(self, mode: str) -> str:
        available = [
            tool for tool in self.get_tools_for_mode(mode)
            if tool.name not in {"plan", "act"}
        ]
        if not available:
            return ""

        lines = [
            "## 可用工具",
            "",
            '调用格式: <tool>{"tool_call":"工具名","param":{...}}</tool>',
            "<tool> 内必须是严格 JSON；不要在标签内外添加解释。",
            "每轮只直接调用一个业务工具；不得调用 plan 或 act。",
            '示例: <tool>{"tool_call":"get_time","param":{}}</tool>',
            '结束示例: <tool>{"tool_call":"speak","param":{"text":"任务已完成"}}</tool>',
            "模型不得声明步骤完成、修改事实或改写旧状态，Harness 会根据工具结果推进状态。",
            "",
        ]
        for tool in available:
            lines.append(tool.get_description_text())
            lines.append("")

        return "\n".join(lines)

    def semantic_view(self, mode: str) -> List[Dict[str, Any]]:
        """Small planner-facing view; execution-only metadata stays internal."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "produces": list(tool.get_harness_metadata().produces),
                "effect": tool.get_harness_metadata().effect,
            }
            for tool in self.get_tools_for_mode(mode)
            if tool.name not in {"plan", "act"}
        ]

    def get_tools_for_mode(self, mode: str) -> List[Tool]:
        return [t for t in self.tools.values() if mode in t.modes]

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            logger.error(f"未知工具: {tool_name}")
            return ToolResult(
                success=False,
                error=f"未知工具: {tool_name}",
                data=f"[错误: 未知工具 '{tool_name}']",
            )

        if context is None:
            context = ToolContext(ros_bridge=self.ros_bridge)

        logger.info(f"执行工具: {tool_name} 参数: {params}")
        try:
            result = await tool.call(params, context)
            metadata = tool.get_harness_metadata()
            if result.success and not result.facts and len(metadata.produces) == 1:
                result.facts = {metadata.produces[0]: result.data}
            logger.info(f"工具结果: {tool_name} success={result.success}")
            return result
        except Exception as e:
            logger.error(f"工具执行异常: {tool_name}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                error_type="TOOL_EXCEPTION",
                data=f"[工具执行失败: {e}]",
            )

    @staticmethod
    def format_result(tool_name: str, result: ToolResult) -> str:
        status = "成功" if result.success else "失败"
        # Message.to_api_dict() adds the source header for tool_result messages.
        # Keep the stored content header-free to avoid duplicating it in the LLM
        # context as "[工具结果: name]" twice.
        parts = [f"状态: {status}"]
        if result.data:
            parts.append(result.data)
        if result.error and not result.success:
            parts.append(f"错误: {result.error}")
        return "\n".join(parts)
