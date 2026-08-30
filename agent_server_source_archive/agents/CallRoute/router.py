from __future__ import annotations

import logging
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ..MCP.base import ToolContext

from .parser.parser import Command, FuzzyCommandParser
from .safety import should_execute, SafetyDecision

logger = logging.getLogger(__name__)


def _repair_truncated_json_object(value: str) -> Optional[dict]:
    """Repair only missing trailing object braces from small-model output."""
    # Observed Qwen 4B variant: it closes ``param`` before the final steps
    # array (``...}}]}``) instead of after it (``...}]}}``).
    if value.endswith("}}]}"):
        try:
            repaired = json.loads(value[:-4] + "}]}}")
        except json.JSONDecodeError:
            pass
        else:
            return repaired if isinstance(repaired, dict) else None
    curly = 0
    square = 0
    in_string = False
    escaped = False
    for char in value:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            curly += 1
        elif char == "}":
            curly -= 1
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
        if curly < 0 or square < 0:
            return None
    if in_string or square != 0 or curly not in {1, 2}:
        return None
    try:
        repaired = json.loads(value + ("}" * curly))
    except json.JSONDecodeError:
        return None
    return repaired if isinstance(repaired, dict) else None


class CallStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CallResult:
    type: str
    data: str = ""
    success: bool = True
    error: str = ""
    error_type: str = ""
    duration_ms: float = 0.0
    cpu_ms: float = 0.0
    empty: bool = False
    retryable: bool = False
    recovery_hint: str = ""
    facts: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    transient: dict = field(default_factory=dict, repr=False)


@dataclass
class ParsedResponse:
    text: str
    commands: List[Command]
    parse_failures: List[str] = field(default_factory=list)


class CallRouter:
    def __init__(self, mcp: "MCPManager" = None, skill_manager: "SkillManager" = None):
        from ..MCP.manager import MCPManager  # noqa: F811
        from ..Skill.manager import SkillManager  # noqa: F811

        self.mcp = mcp
        self.skill_manager = skill_manager
        self._tools = list(mcp.tools.keys()) if mcp else []
        self._skills = list(skill_manager.loader.skills.keys()) if skill_manager else []
        self.parser = FuzzyCommandParser(tools=self._tools, skills=self._skills)
        self.pending_calls: dict = {}

    def update_registered_names(self):
        self._tools = list(self.mcp.tools.keys()) if self.mcp else []
        self._skills = (
            list(self.skill_manager.loader.skills.keys())
            if self.skill_manager else []
        )
        self.parser.update_registered_names(self._tools, self._skills)

    def parse_response(self, full_text: str) -> ParsedResponse:
        """
        对模型的完整响应文本做全量解析。

        策略：从完整文本中提取所有 {...} 命令块，
        剩余部分作为纯文本。对每个命令块用 FuzzyCommandParser 三层解析。
        """
        commands: List[Command] = []
        text_parts: List[str] = []
        parse_failures: List[str] = []

        def parse_xml_tool(match) -> str:
            raw = match.group(0)
            inner = match.group(1).strip()
            try:
                try:
                    payload = json.loads(inner)
                except json.JSONDecodeError:
                    payload = _repair_truncated_json_object(inner)
                    if payload is None:
                        raise
                if not isinstance(payload, dict):
                    raise ValueError("tool payload must be an object")
                command_keys = [key for key in ("tool_call", "skill_call", "judge") if key in payload]
                if len(command_keys) != 1:
                    raise ValueError("tool payload requires exactly one command key")
                command_type = command_keys[0]
                name = payload[command_type]
                params = payload.get("param", {})
                if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
                    raise ValueError("invalid tool name or param object")
                commands.append(Command(
                    type=command_type,
                    name=name.strip(),
                    params=params,
                    raw=raw,
                    confidence=1.0,
                ))
            except (json.JSONDecodeError, ValueError, TypeError):
                parse_failures.append(raw)
            return ""

        # suha.v2 exact path. Remove wrappers before applying legacy fuzzy parsing.
        remaining_text = re.sub(
            r"<tool>\s*(.*?)\s*</tool>", parse_xml_tool, full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        def record_invalid_command(candidate: str) -> None:
            if re.search(r"tool\s*_?\s*call|skill\s*_?\s*call", candidate, re.I):
                parse_failures.append(candidate)

        # 逐个提取花括号块
        i = 0
        while i < len(remaining_text):
            brace_start = remaining_text.find('{', i)
            if brace_start == -1:
                text_parts.append(remaining_text[i:])
                break

            # 纯文本部分
            if brace_start > i:
                text_parts.append(remaining_text[i:brace_start])

            # 找匹配的右花括号（支持嵌套）
            j = brace_start + 1
            depth = 1
            while j < len(remaining_text) and depth > 0:
                if remaining_text[j] == '{':
                    depth += 1
                elif remaining_text[j] == '}':
                    depth -= 1
                j += 1

            if depth == 0:
                # 找到了匹配的 }
                command_text = remaining_text[brace_start:j]
                cmd = self.parser.parse(command_text)
                if cmd.is_valid:
                    commands.append(cmd)
                else:
                    # 解析失败，回退为文本
                    record_invalid_command(command_text)
                    text_parts.append(command_text)
                i = j
            else:
                # 没有匹配的 }，把剩余内容作为可能的不完整命令
                # 尝试解析，失败则作为文本
                remaining = remaining_text[brace_start:]
                cmd = self.parser.parse(remaining)
                if cmd.is_valid:
                    commands.append(cmd)
                    i = len(remaining_text)
                else:
                    record_invalid_command(remaining)
                    text_parts.append(remaining)
                    i = len(remaining_text)

        clean_text = "".join(text_parts).strip()
        return ParsedResponse(
            text=clean_text,
            commands=commands,
            parse_failures=parse_failures,
        )

    async def execute_command(
        self,
        command: Command,
        context: Optional[ToolContext] = None,
    ) -> CallResult:
        if command.type == "tool_call":
            decision = should_execute(
                command.name, command.confidence, is_tool=True
            )
            if not decision.allowed:
                logger.warning(f"工具执行被拒绝: {decision.reason}")
                return CallResult(
                    type="tool_call",
                    success=False,
                    error=decision.reason,
                    data=f"[执行被拒绝: {decision.reason}]",
                )

            if decision.needs_confirm:
                logger.info(f"工具需要确认: {command.name} (置信度: {command.confidence:.2f})")

            if self.mcp is None:
                return CallResult(
                    type="tool_call",
                    success=False,
                    error="MCP manager not initialized",
                    data="[MCP 未初始化]",
                )
            try:
                started = time.monotonic()
                cpu_started = time.process_time()
                tool_result = await self.mcp.execute(
                    command.name, command.params, context=context
                )
                logger.info(
                    "PERF: tool_%s=%.2fs",
                    command.name,
                    time.monotonic() - started,
                )
                return CallResult(
                    type="tool_call",
                    success=tool_result.success,
                    data=tool_result.data,
                    error=tool_result.error if not tool_result.success else "",
                    error_type=tool_result.error_type,
                    duration_ms=(time.monotonic() - started) * 1000,
                    cpu_ms=(time.process_time() - cpu_started) * 1000,
                    empty=tool_result.empty,
                    retryable=tool_result.retryable,
                    recovery_hint=tool_result.recovery_hint,
                    facts=tool_result.facts,
                    diagnostics=tool_result.diagnostics,
                    transient=tool_result.transient,
                )
            except Exception as e:
                logger.error(f"工具执行异常: {command.name}: {e}")
                return CallResult(
                    type="tool_call",
                    success=False,
                    error=str(e),
                    data=f"[工具执行失败: {e}]",
                )

        elif command.type == "skill_call":
            # 查找技能的安全级别
            skill_safety = "normal"
            if self.skill_manager is not None:
                skill_def = self.skill_manager.loader.skills.get(command.name)
                if skill_def is not None:
                    skill_safety = skill_def.safety_level
            decision = should_execute(
                command.name, command.confidence, is_tool=False,
                skill_safety_level=skill_safety,
            )
            if not decision.allowed:
                return CallResult(
                    type="skill_call",
                    success=False,
                    error=decision.reason,
                    data=f"[技能执行被拒绝: {decision.reason}]",
                )

            if self.skill_manager is None:
                return CallResult(
                    type="skill_call",
                    success=False,
                    error="Skill manager not initialized",
                    data="[Skill 管理器未初始化]",
                )
            try:
                content = self.skill_manager.activate(command.name)
                return CallResult(
                    type="skill_call",
                    success=True,
                    data=content,
                )
            except Exception as e:
                logger.error(f"技能执行异常: {command.name}: {e}")
                return CallResult(
                    type="skill_call",
                    success=False,
                    error=str(e),
                    data=f"[技能执行失败: {e}]",
                )

        elif command.type == "judge":
            return CallResult(type="judge", success=True, data=command.name)

        return CallResult(
            type="unknown",
            success=False,
            error=f"未知命令类型: {command.type}",
        )

    def get_pending_status(self, call_id: str) -> CallStatus:
        return self.pending_calls.get(call_id, CallStatus.PENDING)
