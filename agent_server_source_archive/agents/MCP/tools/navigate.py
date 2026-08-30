"""Navigation tool: drive robot to hospital locations via HTTP."""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any, Dict, Iterable, Mapping

from ..base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

NAV_SCRIPT = "/home/linaro/.zeroclaw/workspace/navigate/test_start.py"
STOP_SCRIPT = "/home/linaro/.zeroclaw/workspace/navigate/stop_test.py"

LOCATIONS = ["医院大门", "急诊科", "药房"]
HOSPITAL_LOCATIONS = [
    "医院大门", "门诊大厅", "挂号处", "收费处", "服务台", "急诊科", "药房",
    "检验科", "影像科", "超声科", "输液室", "住院部", "体检中心", "卫生间",
    "呼吸内科", "心血管内科", "消化内科", "神经内科", "内分泌科", "肾内科",
    "血液科", "风湿免疫科", "普通外科", "骨科", "泌尿外科", "神经外科",
    "胸外科", "妇科", "产科", "儿科", "眼科", "耳鼻喉科", "口腔科",
    "皮肤科", "精神心理科", "康复医学科", "中医科",
]

LOCATION_ALIASES = {
    "医院门口": "医院大门", "门口": "医院大门",
    "门诊": "门诊大厅", "门诊部": "门诊大厅",
    "挂号": "挂号处", "挂号台": "挂号处", "挂号窗口": "挂号处",
    "缴费": "收费处", "交费": "收费处", "收费窗口": "收费处",
    "导诊台": "服务台", "咨询台": "服务台",
    "急诊": "急诊科", "急诊室": "急诊科", "看急诊": "急诊科",
    "取药": "药房", "拿药": "药房", "领药": "药房", "配药": "药房",
    "化验室": "检验科", "抽血": "检验科", "抽血处": "检验科", "抽血化验": "检验科",
    "抽血的地方": "检验科", "抽血的那个屋": "检验科", "抽血那个屋": "检验科",
    "放射科": "影像科", "CT": "影像科", "CT室": "影像科", "核磁室": "影像科",
    "拍个片子": "影像科", "拍个片": "影像科",
    "拍片子": "影像科", "拍片": "影像科",
    "B超室": "超声科", "打点滴": "输液室",
    "厕所": "卫生间", "洗手间": "卫生间",
    "呼吸科": "呼吸内科", "心内科": "心血管内科", "消化科": "消化内科",
    "神内科": "神经内科", "内分泌内科": "内分泌科", "肾病科": "肾内科",
    "风湿科": "风湿免疫科", "普外科": "普通外科", "泌尿科": "泌尿外科",
    "看骨头的那科": "骨科", "看骨头那个科": "骨科", "看骨头": "骨科",
    "脑外科": "神经外科", "妇产科": "妇科", "小儿科": "儿科",
    "生小孩那个科": "产科", "生小孩的科": "产科", "生孩子那个科": "产科",
    "五官科": "耳鼻喉科", "牙科": "口腔科", "心理科": "精神心理科",
    "看眼睛的地方": "眼科", "看眼睛": "眼科",
    "康复科": "康复医学科",
}

_STOP_PATTERN = re.compile(r"停止导航|取消导航|停下|别走|不要走")
_INTENT_PATTERN = re.compile(
    r"导航|前往|回到|返回|回去|怎么走|怎么去|"
    r"(?:带|领|送|陪|指引|引导+)(?:着)?我(?:去|到|前往|回到|过去)|"
    r"(?:带|领|送|陪|指引|引导+)(?:着)?我(?:再)?回(?:一下|去|到)?|"
    r"(?:带路|导航)(?:去|到|至)|去|到"
)
_ALIAS_PATTERN = re.compile(
    r"急诊(?:科|室)?|看急诊|取药|拿药|领药|配药|去拿药"
)


def match_location_keyword(
    text: str,
    locations: Iterable[str] = LOCATIONS,
    aliases: Mapping[str, str] = LOCATION_ALIASES,
) -> str:
    """Return the canonical registered destination mentioned in *text*."""
    allowed = tuple(locations)
    # Multi-stop utterances commonly use “先去 A，再带我去 B”.  The active
    # destination is normally the last explicitly mentioned place, not the
    # first item in the registry.
    candidates = [(location, location) for location in allowed]
    candidates.extend((alias, target) for alias, target in aliases.items() if target in allowed)
    matches = [
        (text.rfind(phrase), len(phrase), target)
        for phrase, target in candidates
        if phrase and phrase in text
    ]
    if not matches:
        return ""
    return max(matches)[2]


def match_navigation_request(
    text: str,
    locations: Iterable[str] = LOCATIONS,
    aliases: Mapping[str, str] = LOCATION_ALIASES,
) -> tuple[str, str] | None:
    """Recognize only an explicit physical navigation request.

    Exact registered-place matching is intentionally deterministic.  This
    function is shared by the prompt policy and the tool's parameter recovery
    so the model and the execution layer cannot disagree about a destination.
    """
    utterance = text.splitlines()[-1].strip()
    if not utterance:
        return None
    if _STOP_PATTERN.search(utterance):
        return "stop", ""

    if not _INTENT_PATTERN.search(utterance):
        return None
    target = match_location_keyword(utterance, locations, aliases)
    if not target and re.search(r"回到|返回|回去", utterance):
        target = "医院大门"
    if not target:
        return None
    return "start", target


class NavigateTool(Tool):
    name = "navigate"
    description = (
        "引导机器人移动至指定科室或地点。"
        "当用户说带我、引导我、引导导我、领我、送我、陪我、带路、指引我、"
        "去、到、找、前往、回到、返回或导航至某地点时，优先使用此工具；"
        "用户说取药、拿药、领药或配药时也必须调用，并统一传入 target=药房；"
        "急诊/急诊室统一传入 target=急诊科；返回/回去统一传入 target=医院大门。"
        "工具成功前不得声称已经开始导航。"
    )
    param_schema = {
        "action": "操作类型: start=启动导航, stop=停止导航",
        "target": "目标地点（仅start时需要）。可选: " + ", ".join(LOCATIONS),
        "announcement": "可选：导航启动成功后由导航工具播报的提示语",
    }
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "WRITE", "idempotent": True,
        "produces": ["navigation.status", "navigation.target"],
        "invalidates": ["navigation.status", "navigation.target"],
        "retry": {
            "max_attempts": 2,
            "allowed_errors": ["TIMEOUT", "TEMPORARY_UNAVAILABLE"],
        },
    }

    def __init__(self, execution_mode: str = "real", location_profile: str = "basic") -> None:
        if execution_mode not in {"real", "mock"}:
            raise ValueError("navigation execution_mode must be real or mock")
        self.execution_mode = execution_mode
        if location_profile not in {"basic", "hospital"}:
            raise ValueError("navigation location_profile must be basic or hospital")
        self.location_profile = location_profile
        self.allowed_locations = tuple(
            HOSPITAL_LOCATIONS if location_profile == "hospital" else LOCATIONS
        )
        self.param_schema = {
            "action": "操作类型: start=启动导航, stop=停止导航",
            "target": "目标地点（仅start时需要）。可选: " + ", ".join(self.allowed_locations),
            "announcement": "可选：导航启动成功后由导航工具播报的提示语",
        }

    def match_location(self, text: str) -> str:
        return match_location_keyword(text, self.allowed_locations)

    def match_request(self, text: str) -> tuple[str, str] | None:
        return match_navigation_request(text, self.allowed_locations)

    @staticmethod
    def _latest_user_text(context: ToolContext) -> str:
        session = context.session
        history = getattr(session, "history", ()) if session is not None else ()
        for message in reversed(history):
            if getattr(message, "role", "") == "user":
                return str(getattr(message, "content", "") or "")
        return ""

    def _recover_missing_params(
        self,
        action: str,
        target: str,
        context: ToolContext,
    ) -> tuple[str, str]:
        """Recover only an explicit navigation command from the current turn.

        This is deliberately narrow: it never guesses a destination from tool
        history or assistant text.  Recovery requires a navigation/stop verb in
        the latest external user message and one of the registered locations.
        """
        text = self._latest_user_text(context)
        if not text:
            return action, target

        request = self.match_request(text)
        if request is None:
            return action, target
        matched_action, matched_target = request
        recovered_action = action or matched_action
        recovered_target = target or matched_target
        logger.info(
            "Recovered navigate params from current user request: action=%s target=%s",
            recovered_action,
            recovered_target,
        )
        return recovered_action, recovered_target

    async def _run_script(self, script: str, *args: str, timeout: float) -> ToolResult:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return ToolResult(
                success=False, error=f"导航脚本超时({timeout:g}s)",
                error_type="TIMEOUT", retryable=True,
            )
        except (OSError, Exception) as exc:
            return ToolResult(success=False, error=f"导航脚本启动失败: {exc}")

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error=f"导航脚本退出码 {proc.returncode}: {err or out}",
                data=out,
            )
        if err:
            logger.warning("navigation stderr: %s", err)
        return ToolResult(success=True, data=out)

    async def _announce(
        self, announcement: str, context: ToolContext
    ) -> tuple[str, dict[str, Any]]:
        """Play an already-accepted navigation announcement when possible.

        Navigation is the committed physical side effect.  A speaker outage
        therefore must not turn a successful movement command into a failed
        navigation result; the failure is recorded as an observation instead.
        """
        if not announcement:
            return "", {}
        facts: dict[str, Any] = {"speech.last_text": announcement}
        remote_audio = context.extra.get("remote_audio") if context.extra else None
        if remote_audio is None:
            return "", facts
        try:
            operation_id = await remote_audio.speak(announcement)
            facts["speech.operation_id"] = operation_id
            return "", facts
        except Exception as exc:
            logger.warning("Navigation announcement failed after start: %s", exc)
            facts["speech.announcement_error"] = str(exc)
            return f"（导航播报未送达: {exc}）", facts

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        action = str(params.get("action", "") or "").strip().lower()
        target = str(params.get("target", "") or "").strip()
        announcement = str(params.get("announcement", "") or "").strip()
        if not action or (action == "start" and not target):
            action, target = self._recover_missing_params(action, target, context)

        if action == "stop":
            if self.execution_mode == "mock":
                return ToolResult(
                    success=True, data="[mock] 导航已停止。",
                    facts={"navigation.status": "stopped"},
                )
            result = await self._run_script(STOP_SCRIPT, timeout=15)
            if result.success:
                result.data = f"导航已停止。{result.data}"
                result.facts = {"navigation.status": "stopped"}
            return result

        if action == "start":
            if not target:
                return ToolResult(
                    success=False, error=f"未指定目标地点",
                    data=f"可用地点: {', '.join(self.allowed_locations)}"
                )
            # Keep ASR typo tolerance narrow.  A permissive distance here can
            # turn an unsupported place into a different real destination.
            best = None
            best_dist = 999
            for loc in self.allowed_locations:
                dist = _edit_distance(target, loc)
                if dist < best_dist:
                    best_dist = dist
                    best = loc
            if best_dist > 1:
                return ToolResult(
                    success=False, error=f"未找到地点 '{target}'",
                    data=f"可用地点: {', '.join(self.allowed_locations)}"
                )
            if self.execution_mode == "mock":
                note, speech_facts = await self._announce(announcement, context)
                return ToolResult(
                    success=True,
                    data=f"[mock] 参数校验通过，已开始导航至{best}。{note}",
                    facts={
                        "navigation.status": "navigating",
                        "navigation.target": best,
                        **speech_facts,
                    },
                )
            result = await self._run_script(NAV_SCRIPT, best, timeout=30)
            if result.success:
                note, speech_facts = await self._announce(announcement, context)
                result.data = f"已开始导航至{best}。{result.data}{note}"
                result.facts = {
                    "navigation.status": "navigating",
                    "navigation.target": best,
                    **speech_facts,
                }
            return result

        return ToolResult(
            success=False, error=f"未知操作 '{action}'",
            data="可用操作: start, stop"
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
