from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

DANGEROUS_TOOLS: FrozenSet[str] = frozenset({
    "navigate",
})
SAFE_TOOLS: FrozenSet[str] = frozenset({
    "plan", "get_system_stats", "get_time", "medical_consult", "speak",
})
SKILL_SAFETY_LEVELS: FrozenSet[str] = frozenset({
    "dangerous", "strict", "normal",
})
MIN_CONFIDENCE_DANGEROUS = 0.7
MIN_CONFIDENCE_SAFE = 0.5
MIN_CONFIDENCE_UNKNOWN = 0.5
MIN_CONFIDENCE_SKILL_DANGEROUS = 0.8
MIN_CONFIDENCE_SKILL_STRICT = 0.7
MIN_CONFIDENCE_SKILL_NORMAL = 0.5


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""
    needs_confirm: bool = False


def should_execute(
    command_name: str,
    command_confidence: float,
    is_tool: bool = True,
    skill_safety_level: str = "normal",
) -> SafetyDecision:
    if not is_tool:
        level = skill_safety_level.lower()
        if level == "dangerous":
            if command_confidence >= MIN_CONFIDENCE_SKILL_DANGEROUS:
                return SafetyDecision(allowed=True, needs_confirm=True)
            return SafetyDecision(
                allowed=False,
                reason=f"危险技能置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_SKILL_DANGEROUS}",
            )
        elif level == "strict":
            if command_confidence >= MIN_CONFIDENCE_SKILL_STRICT:
                return SafetyDecision(allowed=True)
            return SafetyDecision(
                allowed=False,
                reason=f"严格技能置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_SKILL_STRICT}",
            )
        else:  # normal
            if command_confidence >= MIN_CONFIDENCE_SKILL_NORMAL:
                return SafetyDecision(allowed=True)
            return SafetyDecision(
                allowed=False, reason=f"技能置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_SKILL_NORMAL}",
            )

    name = command_name

    if name in DANGEROUS_TOOLS:
        if command_confidence >= MIN_CONFIDENCE_DANGEROUS:
            return SafetyDecision(allowed=True, needs_confirm=True)
        return SafetyDecision(
            allowed=False,
            reason=f"危险工具 '{name}' 置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_DANGEROUS}",
        )

    if name in SAFE_TOOLS:
        if command_confidence >= MIN_CONFIDENCE_SAFE:
            return SafetyDecision(allowed=True)
        return SafetyDecision(
            allowed=False,
            reason=f"安全工具 '{name}' 置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_SAFE}",
        )

    if command_confidence >= MIN_CONFIDENCE_UNKNOWN:
        return SafetyDecision(allowed=True, needs_confirm=True)
    return SafetyDecision(
        allowed=False,
        reason=(
            f"未知工具 '{name}' 置信度不足: {command_confidence:.2f} < {MIN_CONFIDENCE_UNKNOWN}。"
            '正确格式: <tool>{"tool_call":"工具名","param":{"参数名":"参数值"}}</tool>'
        ),
    )
