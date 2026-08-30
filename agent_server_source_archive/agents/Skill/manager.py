from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .loader import SkillDef, SkillLoader

logger = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, loader: SkillLoader):
        self.loader = loader
        self.active_skills: Dict[str, SkillDef] = {}

    def get_skills_description(self, mode: str) -> str:
        available = self.get_skills_for_mode(mode)
        if not available:
            return ""

        lines = [
            "## 可用技能",
            "",
            '执行阶段通过 act 激活技能：<tool>{"tool_call":"act","param":'
            '{"step_id":"s1","action_type":"CALL_SKILL","skill":"技能名",'
            '"arguments":{}}}</tool>',
            "技能激活后，系统会告诉你操作步骤。",
            "",
        ]
        for skill in available:
            lines.append(skill.get_skills_list_text())
            lines.append("")

        return "\n".join(lines)

    def get_skills_for_mode(self, mode: str) -> List[SkillDef]:
        return [s for s in self.loader.skills.values() if s.mode == mode]

    def activate(self, skill_name: str) -> str:
        if skill_name not in self.loader.skills:
            logger.warning(f"未知技能: {skill_name}")
            raise KeyError(f"未知技能: {skill_name}")

        skill = self.loader.skills[skill_name]
        self.active_skills[skill_name] = skill
        logger.info(f"激活技能: {skill_name}")
        return skill.get_activation_prompt()

    def deactivate(self, skill_name: str):
        if skill_name in self.active_skills:
            del self.active_skills[skill_name]
            logger.info(f"停用技能: {skill_name}")

    def deactivate_all(self):
        self.active_skills.clear()

    def get_active_skill_prompt(self) -> str:
        if not self.active_skills:
            return ""
        parts = []
        for skill in self.active_skills.values():
            parts.append(skill.get_activation_prompt())
        return "\n\n".join(parts)

    def get_active_allowed_tools(self) -> List[str]:
        tools = []
        for skill in self.active_skills.values():
            tools.extend(skill.allowed_tools)
        return list(dict.fromkeys(tools))

    def is_skill_active(self, skill_name: str) -> bool:
        return skill_name in self.active_skills
