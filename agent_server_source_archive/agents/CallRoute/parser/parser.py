from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

COMMAND_TYPE_ALIASES: Dict[str, List[str]] = {
    "tool_call": [
        "toolcall", "tool_call", "tool-call", "toolcal", "tol_call",
        "toll_call", "call_tool", "tool",
    ],
    "skill_call": [
        "skillcall", "skill_call", "skill-call", "skil_call", "skillcal",
        "skilcal", "invoke_skill",
    ],
    "judge": ["judg", "judgement", "judg_call", "decision"],
}

_STRICT_PATTERN = re.compile(
    r'\{\s*"?\s*(tool_call|skill_call|judge)\s*"?\s*:\s*"([^"]+)"\s*'
    # Small models frequently emit JSON-ish variants such as
    # {"tool_call":"navigate" "param"{"action":"start"}} or
    # {"tool_call":"speak" "param{text":"..."}}.  Accept optional quotes
    # and a missing ':' around the param key while keeping the command name
    # and parameter values quoted.
    r'(?:["\']?\s*param\s*["\']?\s*[:=]?\s*\{([^}]*)\})?\s*\}',
    re.IGNORECASE,
)

_LOOSE_PATTERN = re.compile(
    r'\{\s*(\w[\w\s_\-]*)\s*[:=]\s*["\']*([\w][\w\-]*)["\']*\s*'
    r'(?:\s*param\s*[\{\(\[]\s*([^\}\)\]]*?)\s*[\}\)\]])?\s*[,;]?\s*\}',
    re.IGNORECASE | re.DOTALL,
)

_PARAM_STRICT_RE = re.compile(r'(\w+)"?\s*[:=]\s*"([^"]*)"')
_PARAM_LOOSE_RE = re.compile(r'(\w+)"?\s*[:=]\s*["\']*([^"\'}\s,]+)["\']*')
_PARAM_ARRAY_RE = re.compile(r'(\w+)"?\s*[:=]\s*(\[[^\]]*\])')


registered_tools: List[str] = []
registered_skills: List[str] = []


def set_registered_names(tools: List[str], skills: List[str]):
    global registered_tools, registered_skills
    registered_tools = list(tools)
    registered_skills = list(skills)


@dataclass
class Command:
    type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    confidence: float = 0.0

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.name.strip())
            and self.confidence >= 0.5
            and self.type in ("tool_call", "skill_call", "judge")
        )


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _normalize_command_type(raw: str) -> Optional[str]:
    raw_clean = raw.lower().replace(' ', '_').replace('-', '')
    for canonical, aliases in COMMAND_TYPE_ALIASES.items():
        for alias in aliases:
            if raw_clean == alias.replace('_', '').replace('-', ''):
                return canonical
            if raw_clean == alias:
                return canonical
    best_match: Optional[str] = None
    best_dist = float('inf')
    for canonical, aliases in COMMAND_TYPE_ALIASES.items():
        for alias in aliases:
            alias_clean = alias.replace('_', '').replace('-', '')
            dist = _edit_distance(raw_clean, alias_clean)
            if dist < best_dist and dist <= 2:
                best_dist = dist
                best_match = canonical
    return best_match


def _fuzzy_match_name(name: str, cmd_type: str) -> str:
    candidates = (
        registered_tools if cmd_type == "tool_call"
        else registered_skills if cmd_type == "skill_call"
        else []
    )
    if not candidates:
        return name
    if name in candidates:
        return name
    best_match = name
    best_dist = float('inf')
    for candidate in candidates:
        dist = _edit_distance(name.lower(), candidate.lower())
        if dist < best_dist:
            best_dist = dist
            best_match = candidate
    if best_dist <= 2:
        return best_match
    return name


def _parse_params_strict(params_str: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # Strip leading/trailing quotes (JSON format)
    if params_str.startswith('"'):
        params_str = params_str.lstrip('"')
    for m in _PARAM_STRICT_RE.finditer(params_str):
        result[m.group(1)] = m.group(2)
    # A command may match the strict outer syntax while the model omits quotes
    # around one or more parameter values. Merge the loose parser's result so a
    # high-confidence outer match never silently discards valid parameters.
    for key, value in _parse_params_loose(params_str).items():
        result.setdefault(key, value)
    for match in _PARAM_ARRAY_RE.finditer(params_str):
        try:
            result[match.group(1)] = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
    return result


def _parse_params_loose(params_str: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    filter_words = {
        'tool_call', 'skill_call', 'judge',
        'param', 'name', 'tool', 'skill',
    }
    if params_str.startswith('"'):
        params_str = params_str.lstrip('"')
    for m in _PARAM_LOOSE_RE.finditer(params_str):
        key, value = m.group(1), m.group(2)
        if key.lower() not in filter_words:
            result[key] = value
    return result


class FuzzyCommandParser:
    def __init__(
        self,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ):
        self._tools = tools or []
        self._skills = skills or []
        set_registered_names(self._tools, self._skills)

    def parse(self, text: str) -> Command:
        result = self._try_strict_parse(text)
        if result and result.confidence >= 0.9:
            return result

        result = result or self._try_loose_parse(text)
        if result and result.confidence >= 0.7:
            return result

        result = result or self._try_heuristic_parse(text)
        if result:
            return result

        return Command(type="unknown", name="", params={}, raw=text, confidence=0.0)

    def update_registered_names(self, tools: List[str], skills: List[str]):
        self._tools = tools
        self._skills = skills
        set_registered_names(tools, skills)

    def _try_strict_parse(self, text: str) -> Optional[Command]:
        match = _STRICT_PATTERN.search(text)
        if not match:
            return None
        cmd_type = match.group(1).lower().replace(' ', '_').replace('-', '_')
        cmd_name = match.group(2)
        params = _parse_params_strict(match.group(3)) if match.group(3) else _parse_params_loose(match.group(3)) if match.group(3) else {}
        normalized = _normalize_command_type(cmd_type)
        if normalized is None:
            return None
        return Command(
            type=normalized,
            name=cmd_name,
            params=params,
            raw=text,
            confidence=0.95,
        )

    def _try_loose_parse(self, text: str) -> Optional[Command]:
        match = _LOOSE_PATTERN.search(text)
        if not match:
            return None

        raw_type = match.group(1).lower().replace(' ', '_').replace('-', '')
        cmd_type = _normalize_command_type(raw_type)
        if cmd_type is None:
            return None

        cmd_name = match.group(2).strip()
        params = _parse_params_loose(match.group(3)) if match.group(3) else {}

        cmd_name = _fuzzy_match_name(cmd_name, cmd_type)

        confidence = 0.75
        if cmd_type == "tool_call" and cmd_name in registered_tools:
            confidence = 0.85
        elif cmd_type == "skill_call" and cmd_name in registered_skills:
            confidence = 0.85

        return Command(
            type=cmd_type, name=cmd_name, params=params,
            raw=text, confidence=confidence,
        )

    def _try_heuristic_parse(self, text: str) -> Optional[Command]:
        text_lower = text.lower()

        cmd_type: Optional[str] = None
        best_dist = float('inf')
        for canonical, aliases in COMMAND_TYPE_ALIASES.items():
            for alias in aliases:
                alias_clean = alias.replace('_', '').replace('-', '')
                for i in range(max(1, len(text_lower) - len(alias_clean) + 1)):
                    segment = text_lower[i:i + len(alias_clean)]
                    dist = _edit_distance(segment, alias_clean)
                    if dist < best_dist and dist <= 2:
                        best_dist = dist
                        cmd_type = canonical

        if cmd_type is None:
            return None

        cmd_name = ""
        quoted = re.findall(r'["\']([^"\'}]+)["\']', text)
        # Skip quoted strings that match command type aliases or known keys
        _skip_keys = {'tool_call', 'skill_call', 'judge', 'param', 'text', 'name'}
        for _q in quoted:
            if _q not in _skip_keys:
                cmd_name = _q
                break
        if not cmd_name and quoted:
            cmd_name = quoted[0]
        if not cmd_name:
            after_sep = re.search(r'[:=]\s*(\w[\w\-]*)', text)
            if after_sep:
                cmd_name = after_sep.group(1)

        cmd_name = _fuzzy_match_name(cmd_name, cmd_type)

        params: Dict[str, str] = {}
        param_block = text
        for m in re.finditer(r'(\w+)\s*[:=]\s*["\']*([^"\'}\s,]+)["\']*', param_block):
            key, value = m.group(1), m.group(2)
            if key.lower() not in {
                'tool_call', 'skill_call', 'judge',
                'param', 'name',
            }:
                params[key] = value

        confidence = max(0.3, 0.6 - best_dist * 0.1)
        if cmd_name and cmd_name in set(registered_tools) | set(registered_skills):
            confidence += 0.2

        return Command(
            type=cmd_type, name=cmd_name, params=params,
            raw=text, confidence=min(confidence, 0.9),
        )
