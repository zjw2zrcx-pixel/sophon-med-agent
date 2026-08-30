from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillDef:
    name: str
    description: str
    when_to_use: str = ""
    mode: str = "Voice"
    safety_level: str = "normal"
    allowed_tools: List[str] = field(default_factory=list)
    content: str = ""

    def get_skills_list_text(self) -> str:
        return (
            f"### {self.name}\n"
            f"描述: {self.description}\n"
            f"使用时机: {self.when_to_use}"
        )

    def get_activation_prompt(self) -> str:
        lines = [
            f"## 已激活技能: {self.name}",
            f"描述: {self.description}",
            f"适用模式: {self.mode}",
        ]
        if self.allowed_tools:
            lines.append(f"可用工具: {', '.join(self.allowed_tools)}")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)


class SkillLoader:
    def __init__(self, skill_dir: str):
        self.skill_dir = Path(skill_dir)
        self.skills: Dict[str, SkillDef] = {}

    def load_all(self) -> Dict[str, SkillDef]:
        self.skills.clear()
        if not self.skill_dir.exists():
            logger.warning(f"技能目录不存在: {self.skill_dir}")
            return self.skills

        for skill_path in self.skill_dir.iterdir():
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if skill_md.exists():
                skill_def = self._parse_skill_md(skill_md)
                if skill_def:
                    self.skills[skill_def.name] = skill_def
                    logger.info(f"加载技能: {skill_def.name}")

        return self.skills

    def _parse_skill_md(self, path: Path) -> Optional[SkillDef]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"读取技能文件失败 {path}: {e}")
            return None

        frontmatter: Dict[str, str] = {}
        content = text

        fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            content = fm_match.group(2).strip()
            pending_list_key: Optional[str] = None
            for line in fm_text.split("\n"):
                line = line.strip()
                if line.startswith("-") and pending_list_key:
                    frontmatter[pending_list_key] = (
                        frontmatter.get(pending_list_key, "")
                        + (", " if frontmatter.get(pending_list_key) else "")
                        + line[1:].strip().strip('"').strip("'")
                    )
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip().lower().replace(" ", "_")
                    val = val.strip().strip('"').strip("'")
                    frontmatter[key] = val
                    pending_list_key = key if not val else None

        name = frontmatter.get("name", path.parent.name)
        description = frontmatter.get("description", "")
        when_to_use = frontmatter.get("when_to_use", "")
        mode = frontmatter.get("mode", "Voice")
        safety_level = frontmatter.get("safety_level", "normal")

        allowed_tools: List[str] = []
        if "allowed_tools" in frontmatter:
            tools_str = frontmatter["allowed_tools"]
            allowed_tools = [
                t.strip().strip("-").strip()
                for t in tools_str.split(",")
                if t.strip()
            ]

        return SkillDef(
            name=name,
            description=description,
            when_to_use=when_to_use,
            mode=mode,
            safety_level=safety_level,
            allowed_tools=allowed_tools,
            content=content,
        )
