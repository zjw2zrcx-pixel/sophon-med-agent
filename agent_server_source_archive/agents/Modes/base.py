from __future__ import annotations

import logging
import asyncio
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..API.session import Session
from ..API.api import API
from ..MCP.manager import MCPManager
from ..MCP.base import ToolContext
from ..Skill.manager import SkillManager
from ..CallRoute.router import CallRouter, ParsedResponse
from ..CallRoute.parser.parser import Command
from ..Harness import ActionValidator, ExecutionState, TaskPlan, ToolMetadata
from ..medical_policy import unsupported_departments

logger = logging.getLogger(__name__)


@dataclass
class ModeConfig:
    name: str
    max_turns: int = 15
    image_freq: str = "on_demand"
    safety_level: str = "normal"
    auto_compact_threshold: float = 0.8
    history_visible_entries: int = 8


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "Voice": ModeConfig(
        name="Voice", max_turns=12, image_freq="always",
        safety_level="normal", auto_compact_threshold=1.0,
    ),
    "Benchmark": ModeConfig(
        name="Benchmark", max_turns=8, image_freq="never",
        safety_level="normal", auto_compact_threshold=1.0,
    ),
}


@dataclass
class LoopResult:
    text: str = ""
    commands: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    turn_end_reason: str = ""
    session_ended: bool = False
    continuation_pcm: bytes = field(default=b"", repr=False)
    continuation_audio: dict = field(default_factory=dict)


class ModeBase(ABC):
    name: str = ""

    def __init__(
        self,
        api: API,
        mcp: MCPManager,
        skill_manager: SkillManager,
        call_router: CallRouter,
    ):
        self.api = api
        self.mcp = mcp
        self.skills = skill_manager
        self.call_router = call_router
        self.session: Optional[Session] = None
        self.config = MODE_CONFIGS.get(self.name, ModeConfig(name=self.name))
        self._cached_base_prompt: Optional[str] = None
        self.action_validator = ActionValidator()

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def get_max_turns(self) -> int:
        return self.config.max_turns

    # These hooks deliberately live outside the model prompt.  A mode may
    # deterministically own its workflow while retaining the normal Harness,
    # tool events, and trajectory format.
    def get_compact_workflow_plan(self, user_input: str) -> Optional[dict]:
        return None

    def get_scheduled_workflow_command(
        self, user_input: str
    ) -> Optional[Command]:
        return None

    def adapt_compact_terminal_action(
        self, command: Command, execution_state: ExecutionState
    ) -> Optional[Command]:
        """Normalize a complete final response into the required speak action.

        Small models frequently emit ``act(FINISH)`` after producing the right
        answer even though Voice sessions must terminate through TTS.  Treat
        this as serialization repair only when the workflow is already at its
        speak step (or all planned work is satisfied); never use it to skip
        pending business tools.
        """
        if command.type != "tool_call" or command.name != "act":
            return None
        params = command.params
        if str(params.get("action_type", "")).upper() != "FINISH":
            return None
        text = str(params.get("response", "") or "").strip()
        step = execution_state.active_step
        terminal_ready = bool(
            (step is not None and step.preferred_tool == "speak")
            or (
                step is None
                and execution_state.status == "GOAL_SATISFIED"
                and not execution_state.active_step_id
            )
        )
        if not text or not terminal_ready:
            return None
        return Command(
            type="tool_call", name="speak", params={"text": text},
            raw="[agent_action_repaired:finish_to_speak]\n" + command.raw,
            confidence=command.confidence,
        )

    def adapt_compact_plain_text(
        self, response_text: str, execution_state: ExecutionState
    ) -> Optional[Command]:
        return None

    def adapt_compact_truncated_terminal_output(
        self, response_text: str, execution_state: ExecutionState
    ) -> Optional[Command]:
        """Recover a cut-off final speak envelope in Voice and Benchmark."""
        step = execution_state.active_step
        if step is None or step.preferred_tool != "speak":
            return None
        if "<tool>" not in response_text or "</tool>" in response_text:
            return None
        if not re.search(r'"(?:tool_call|action_type)"\s*:', response_text):
            return None
        match = re.search(r'"text"\s*:\s*"(.*)$', response_text, re.DOTALL)
        if match is None:
            return None
        text = match.group(1).rstrip()
        text = re.sub(r'(?<!\\)"\s*\}+\s*$', '', text)
        text = re.sub(r'\\(?:u[0-9a-fA-F]{0,3})?$', '', text)
        text = text.replace(r'\n', '\n').replace(r'\"', '"').replace(r'\\', '\\')
        text = text.strip().rstrip('，、；：')
        if len(text) < 4:
            return None
        return Command(
            type="tool_call", name="speak", params={"text": text},
            raw="[agent_action_repaired:truncated_terminal_to_speak]\n" + response_text,
            confidence=1.0,
        )

    def is_compact_navigation_announcement_terminal(
        self, command: Command, call_result: Any, execution_state: Any
    ) -> bool:
        """Whether a successful navigation command also ends this turn.

        The default preserves normal voice-mode behavior.  Benchmark mode
        opts in only for its single-action navigation workflow.
        """
        return False

    def adapt_grounded_speak_command(
        self, command: Command, execution_state: Optional[ExecutionState]
    ) -> Optional[Command]:
        """Remove unsupported department names without another model retry."""
        if (
            execution_state is None
            or command.type != "tool_call"
            or command.name != "speak"
        ):
            return None
        consultation = execution_state.facts.get("medical.consultation")
        if consultation is None or not consultation.valid:
            return None
        navigation = execution_state.facts.get("navigation.target")
        allowed_navigation = (
            [str(navigation.value)]
            if navigation is not None and navigation.valid else []
        )
        text = str(command.params.get("text", ""))
        unsupported = unsupported_departments(
            text, consultation.value, allowed_navigation
        )
        if not unsupported:
            return None
        repaired = text
        for department in unsupported:
            repaired = repaired.replace(department, "相关专科")
        return Command(
            type="tool_call", name="speak", params={"text": repaired},
            raw="[agent_action_repaired:unsupported_department_removed]\n" + command.raw,
            confidence=command.confidence,
        )

    def get_plain_response_rejection(
        self,
        user_input: str,
        response_text: str,
        successful_tools: set,
        attempted_tools: Optional[set] = None,
    ) -> str:
        """Return a correction when plain text cannot satisfy this request."""
        return self._navigation_response_rejection(
            user_input,
            response_text,
            successful_tools,
            attempted_tools or set(),
        )

    def get_command_rejection(
        self,
        user_input: str,
        command,
        successful_tools: set,
        attempted_tools: Optional[set] = None,
    ) -> str:
        """Return a correction when a command would violate mode policy."""
        attempted_tools = attempted_tools or set()
        if str(getattr(command, "raw", "")).startswith(
            "[agent_workflow_blocked_terminal]"
        ):
            return ""
        if (
            command.type == "tool_call"
            and command.name == "plan"
        ):
            return "plan 只能在当前任务的第一次模型调用中执行，现有 plan 已固定在 user 中。"
        if (
            command.type == "tool_call"
            and command.name in {"medical_consult", "get_time", "get_system_stats"}
            and command.name in successful_tools
        ):
            return (
                f"{command.name} 在当前 task 已成功执行，禁止重复查询。"
                "请直接依据 history 中的成功结果调用 speak。"
            )
        if (
            command.type == "tool_call"
            and command.name == "navigate"
            and "navigate" in attempted_tools
        ):
            return (
                "navigate 在本轮已经调用，禁止重复执行物理导航。"
                "请依据已有工具结果向用户报告。"
            )
        if command.type == "tool_call" and command.name == "query":
            session = self.session
            current_turn = len(session.conversation_turns) + 1 if session else 1
            if current_turn >= (session.conversation_max_turns if session else 3):
                return (
                    "SESSION_TURN_LIMIT: 当前已是本 session 第三轮，禁止继续 query；"
                    "请调用 speak 给出诚实答复并结束 session。"
                )
            followup = None
            if session is not None and session.execution_state is not None:
                followup = session.execution_state.facts.get("dialogue.followup_required")
            if followup is None or not followup.valid or followup.value is not True:
                return (
                    "QUERY_NOT_ALLOWED: 当前没有业务工具明确要求补充信息；"
                    "不得无必要调用 query。"
                )
        if command.type == "tool_call" and command.name == "speak":
            session = self.session
            state = session.execution_state if session is not None else None
            current_turn = len(session.conversation_turns) + 1 if session else 1
            max_turns = session.conversation_max_turns if session else 3
            if state is not None:
                followup = state.facts.get("dialogue.followup_required")
                if (
                    followup is not None and followup.valid and followup.value is True
                    and current_turn < max_turns
                ):
                    return (
                        "MEDICAL_FOLLOWUP_REQUIRED: 医疗工具明确要求补充信息；"
                        "当前尚未到第三轮，必须调用 query 询问一个最重要的问题，"
                        "不得直接 speak 结束 session。"
                    )
                consultation = state.facts.get("medical.consultation")
                if consultation is not None and consultation.valid:
                    navigation = state.facts.get("navigation.target")
                    allowed_navigation = (
                        [str(navigation.value)]
                        if navigation is not None and navigation.valid else []
                    )
                    unsupported = unsupported_departments(
                        str(command.params.get("text", "")),
                        consultation.value,
                        allowed_navigation,
                    )
                    if unsupported:
                        return (
                            "MEDICAL_DEPARTMENT_UNSUPPORTED: medical_consult 未返回科室 "
                            + "、".join(unsupported)
                            + "；删除这些科室，只依据结构化医疗证据回答。"
                        )
            return self._navigation_response_rejection(
                user_input,
                str(command.params.get("text", "")),
                successful_tools,
                attempted_tools,
            )
        return ""

    def get_turn_policy_prompt(
        self,
        user_input: str,
        successful_tools: set,
        attempted_tools: set,
    ) -> str:
        """Build an immutable execution rule for the current user task."""
        navigate = self.mcp.tools.get("navigate")
        location = navigate.match_location(user_input) if navigate else ""
        request = navigate.match_request(user_input) if navigate else None
        if request is None:
            if not location:
                return ""
            return (
                "## 运行时地点识别提示（高优先级）\n"
                f"关键词“{location}”与 navigate 工具注册的真实地点完全匹配，"
                "这很可能是在提及一个物理地点。请优先判断用户是否要求机器人移动；"
                "若存在移动意图，必须调用 navigate，不能用自然语言假装执行。"
            )

        action, target = request
        if action == "stop":
            command = (
                '<tool>{"tool_call":"navigate","param":{"action":"stop"}}</tool>'
            )
            place_note = "用户明确要求停止物理导航。"
        else:
            command = (
                '<tool>{"tool_call":"navigate","param":'
                f'{{"action":"start","target":"{target}"}}}}</tool>'
            )
            place_note = (
                f"执行策略已确认“{target}”是 navigate 注册的真实地点，"
                "且用户表达了物理移动意图。"
            )
        return (
            "## 本轮导航执行约束（最高优先级）\n"
            + place_note
            + "若 history 中尚无 navigate 成功记录，且 attempt 中尚无 navigate 失败记录，"
            + "必须且只能执行以下工具调用：\n"
            + command
            + "\n若 history 已记录成功，禁止重复调用，只能依据结果播报，且未确认到达时"
            + "不得声称已经到达。若 attempt 已记录失败，禁止假装已经行动，也不得重复"
            + "同一失败调用；应依据失败原因修正参数或如实报告。"
        )

    @staticmethod
    def _record_failed_tool_attempt(command, call_result) -> bool:
        """Return False only for an empty result from a query-style tool."""
        if call_result.success:
            return False
        if getattr(call_result, "empty", False) or getattr(call_result, "retryable", False):
            return True
        query_tools = {"medical_consult", "get_time", "get_system_stats"}
        if command.name not in query_tools:
            return True
        detail = f"{call_result.data}\n{call_result.error}".strip()
        if not detail:
            return False
        return not bool(re.search(r"未找到|无结果|没有(?:可用)?证据|结果为空", detail))

    @staticmethod
    def _unwrap_act_command(command: Command, execution_state: ExecutionState):
        """Convert one model ACT decision into the actual business command."""
        if command.type != "tool_call" or command.name != "act":
            return None, (
                "ACT_WRAPPER_REQUIRED",
                "执行轮必须通过 act 提出动作，禁止直接调用业务工具。",
            )
        params = command.params
        action_type = str(params.get("action_type", "CALL_TOOL") or "CALL_TOOL").upper()
        if action_type not in {"CALL_TOOL", "CALL_SKILL", "FINISH"}:
            return None, (
                "ACT_TYPE_INVALID",
                "act.action_type 只能是 CALL_TOOL、CALL_SKILL 或 FINISH。",
            )
        step_id = str(params.get("step_id", "") or "").strip()
        name_key = "tool" if action_type == "CALL_TOOL" else "skill"
        name = str(params.get(name_key) or params.get("name") or "").strip()
        post_goal_speak = (
            action_type == "CALL_TOOL"
            and name == "speak"
            and execution_state.status in {"GOAL_SATISFIED", "BLOCKED", "FAILED"}
            and not execution_state.active_step_id
        )
        if execution_state.active_step_id and step_id != execution_state.active_step_id:
            return None, (
                "ACT_STEP_MISMATCH",
                f"act.step_id 必须等于当前步骤 {execution_state.active_step_id or '<none>'}。",
            )
        if not execution_state.active_step_id and step_id:
            return None, (
                "ACT_STEP_MISMATCH",
                "当前没有活动步骤，FINISH 时 step_id 必须为空。",
            )
        if action_type == "FINISH":
            response = str(params.get("response", "") or "").strip()
            if not response:
                return None, (
                    "ACT_PAYLOAD_INVALID", "act(FINISH) 必须包含非空 response。"
                )
            return Command(
                type="judge", name="finish", params={"response": response},
                raw=command.raw, confidence=command.confidence,
            ), None
        if not step_id and not post_goal_speak:
            return None, (
                "ACT_STEP_MISMATCH", "CALL_TOOL/CALL_SKILL 需要活动 step_id。"
            )
        arguments = params.get("arguments", params.get("params", {}))
        if not name or not isinstance(arguments, dict):
            return None, (
                "ACT_PAYLOAD_INVALID",
                f"act 必须包含 {name_key} 和 arguments 对象。",
            )
        return Command(
            type="tool_call" if action_type == "CALL_TOOL" else "skill_call",
            name=name,
            params=arguments,
            raw=command.raw,
            confidence=command.confidence,
        ), None

    def get_policy_fallback_command(
        self,
        user_input: str,
        successful_tools: set,
        attempted_tools: set,
    ):
        """Return a deterministic command after the model ignores policy."""
        if "navigate" in successful_tools or "navigate" in attempted_tools:
            return None
        from ..CallRoute.parser.parser import Command

        navigate = self.mcp.tools.get("navigate")
        request = navigate.match_request(user_input) if navigate else None
        if request is None:
            return None
        action, target = request
        params = {"action": action}
        if target:
            params["target"] = target
        return Command(
            type="tool_call",
            name="navigate",
            params=params,
            raw="[execution_policy_fallback]",
            confidence=1.0,
        )

    def _navigation_response_rejection(
        self,
        user_input: str,
        response_text: str,
        successful_tools: set,
        attempted_tools: set,
    ) -> str:
        navigate = self.mcp.tools.get("navigate")
        if navigate is None or navigate.match_request(user_input) is None:
            return ""

        arrival_mentioned = re.search(r"到达|抵达", response_text)
        arrival_denied = re.search(
            r"(?:尚未|还未|未|没有|并未)(?:成功)?(?:到达|抵达)",
            response_text,
        )
        claimed_arrival = bool(arrival_mentioned and not arrival_denied)
        if "navigate" in successful_tools:
            if claimed_arrival:
                return (
                    "navigate 只确认已开始导航，并未确认到达。"
                    "禁止声称已经到达；请严格依据工具结果播报。"
                )
            return ""

        if "navigate" not in attempted_tools:
            return (
                "当前用户要求执行物理导航，但 navigate 工具尚未调用。"
                "该地点已由执行策略确认；禁止用自然语言声称正在导航，必须立即调用 navigate。"
            )

        false_success = re.search(
            r"正在(?:导航|前往|返回)|已(?:经)?(?:开始|出发)|导航中|开始前往|"
            r"马上(?:前往|出发)",
            response_text,
        )
        if false_success or claimed_arrival:
            return (
                "navigate 执行失败。禁止声称正在导航、已经出发或已经到达；"
                "必须如实说明工具返回的失败原因。"
            )
        honest_failure = re.search(
            r"失败|无法|未能|没有成功|出错|错误|异常|超时|未找到|不支持",
            response_text,
        )
        if not honest_failure:
            return "navigate 执行失败；必须明确、如实地向用户报告失败。"
        return ""

    def get_available_tools_description(self) -> str:
        return self.mcp.get_tool_description(self.name)

    def get_available_skills_description(self) -> str:
        return self.skills.get_skills_description(self.name)

    def get_base_prompt(self) -> str:
        if self._cached_base_prompt is not None:
            return self._cached_base_prompt
        parts = [self.get_system_prompt().strip()]
        tools_desc = self.get_available_tools_description()
        if tools_desc:
            parts.append(tools_desc.strip())
        skills_desc = self.get_available_skills_description()
        if skills_desc:
            parts.append(skills_desc.strip())
        self._cached_base_prompt = "\n\n".join(parts)
        return self._cached_base_prompt

    def build_full_prompt(
        self,
        session: Session,
        context: str = "",
        sensor_data: str = "",
    ) -> str:
        parts = [self.get_base_prompt()]
        active_skill_prompt = self.skills.get_active_skill_prompt()
        if active_skill_prompt:
            parts.append(active_skill_prompt)
        if context:
            parts.append("[CONTEXT]\n" + context)
        if sensor_data:
            parts.append("[SENSOR]\n" + sensor_data)
        return "\n\n".join(parts)

    async def loop(
        self,
        user_input: str,
        image: Optional[str] = None,
        context: str = "",
        sensor_data: str = "",
        tool_context_extra: Optional[dict] = None,
    ) -> LoopResult:
        loop_wall_started = time.monotonic()
        loop_cpu_started = time.thread_time()
        loop_process_cpu_started = time.process_time()
        prompt_cpu_started = time.thread_time()
        if self.session is None:
            self.session = Session(mode=self.name)
            self.session.add_message("system", self.get_base_prompt())

        actual_user_input = user_input
        continuation = self.session.consume_pending_dialogue(actual_user_input)
        workflow_user_input = actual_user_input
        if continuation is not None:
            answers = [
                str(item).strip() for item in continuation.get("answers", [])
                if str(item).strip()
            ]
            resume_input = str(
                continuation.get("resume_user_input")
                or continuation.get("root_user_input", "")
            ).strip()
            workflow_user_input = (
                "待续医疗查询：" + resume_input
                + "\n上轮追问：" + str(continuation.get("question", "")).strip()
                + "\n合并医疗查询："
                + resume_input
                + "；补充回答：" + "；".join(answers)
            )
        conversation_context = self.session.render_conversation_context()
        self.session.begin_external_turn(
            actual_user_input,
            input_metadata=dict((tool_context_extra or {}).get("input_metadata", {})),
        )
        self.session.prompt_slots.history.set_max_visible(
            self.config.history_visible_entries
        )
        task_policy = self.get_turn_policy_prompt(workflow_user_input, set(), set())
        self.session.prompt_slots.start_task(
            system=self.get_base_prompt(),
            user_input=actual_user_input,
            image=image,
            conversation_context=conversation_context,
            runtime_context=context,
            sensor_data=sensor_data,
            task_policy=task_policy,
            pending_dialogue=continuation,
        )
        self.session.add_message("user", actual_user_input, image=image)
        # From this point routing and tool arguments use the reconstructed
        # semantic request.  Conversation storage and user-visible events keep
        # the actual utterance above.
        user_input = workflow_user_input
        prompt_cpu_ms = (time.thread_time() - prompt_cpu_started) * 1000

        trace_id = uuid.uuid4().hex
        trace_started = time.monotonic()
        trace_status = "completed"
        final_emitted = False
        trajectory_events: List[dict] = []

        async def emit(event_type: str, payload: Optional[dict] = None):
            event = {
                "type": event_type,
                "timestamp": time.time(),
                "payload": payload or {},
            }
            trajectory_events.append(event)
            emitter = getattr(self.api, "emit_agent_event", None)
            if emitter is None:
                return
            await emitter(
                trace_id,
                event_type,
                session_id=self.session.id,
                mode=self.name,
                payload=payload or {},
            )

        display_input = actual_user_input
        if display_input.startswith("[语音识别结果") and "\n" in display_input:
            display_input = display_input.split("\n", 1)[1]
        await emit("turn_start", {
            "prompt": display_input,
            "has_image": image is not None,
            "input": dict((tool_context_extra or {}).get("input_metadata", {})),
        })

        result = LoopResult()
        result.metrics = {
            "prompt_context_cpu_ms": prompt_cpu_ms,
            "parse_cpu_ms": 0.0,
            "policy_cpu_ms": 0.0,
            "session_commit_cpu_ms": 0.0,
            "tool_wall_ms": 0.0,
            "tool_cpu_ms": 0.0,
        }
        compact_plan = self.get_compact_workflow_plan(user_input)
        if compact_plan is not None:
            task_plan = TaskPlan.from_payload(compact_plan)
            self.session.prompt_slots.set_plan(compact_plan)
            self.session.execution_state = ExecutionState.create(task_plan)
            self.session.prompt_slots.append_execution_event(
                self.session.execution_state.append_event("AGENT_WORKFLOW_ATTACHED")
            )
            result.metrics["compact_workflow"] = True
            await emit("harness_state", {
                "event": "AGENT_WORKFLOW_ATTACHED",
                "state": self.session.execution_state.projection(),
            })
        blocked_duplicate_counts = {}
        total_failed_calls = 0
        policy_rejection_count = 0
        planning_failure_count = 0
        repeated_failure = False
        successful_tools = set()
        attempted_tools = set()

        try:
            for turn in range(self.get_max_turns()):

                execution_state = self.session.execution_state
                scheduled_command = self.get_scheduled_workflow_command(user_input)
                if execution_state is not None:
                    if (
                        scheduled_command is None
                        and execution_state.status in {"BLOCKED", "FAILED"}
                        and not execution_state.active_step_id
                    ):
                        observation = execution_state.last_observation or {}
                        detail = str(observation.get("data") or "").strip()
                        if len(detail) > 160:
                            detail = detail[:160].rstrip() + "…"
                        message = "操作未能完成"
                        if detail:
                            message += "：" + detail
                        scheduled_command = Command(
                            type="tool_call", name="speak",
                            params={"text": message + "。"},
                            raw="[agent_workflow_blocked_terminal]",
                            confidence=1.0,
                        )
                        trace_status = "error"
                    # Deterministic workflow actions are Agent work, rather
                    # than an additional model retry.
                    if scheduled_command is None:
                        execution_state.begin_model_turn()
                    if execution_state.status == "BUDGET_EXHAUSTED":
                        result.text = "任务动作预算已耗尽，已停止执行。"
                        trace_status = "error"
                        self.session.compact_current_turn(result.text)
                        break

                logger.info(f"[{self.name}] Loop turn {turn + 1}/{self.get_max_turns()}")
                model_generated = scheduled_command is None
                if model_generated:
                    await emit("model_start", {"iteration": turn + 1})
                    _turn_t0 = time.time()
                    full_response = await self.api.chat_complete(self.session)
                    logger.info(f"PERF: llm_turn_{turn+1}={time.time() - _turn_t0:.2f}s")
                    logger.info(f"[{self.name}] LLM_RAW_OUTPUT ({len(full_response)} chars): {full_response[:500]}")
                    await emit("model_output", {
                        "iteration": turn + 1,
                        "text": full_response,
                    })
                    if not full_response.strip():
                        logger.warning(f"[{self.name}] LLM returned empty response")
                        await emit("error", {"message": "模型返回了空响应"})
                        trace_status = "error"
                        break
                    parse_cpu_started = time.thread_time()
                    parsed = self.call_router.parse_response(full_response)
                    result.metrics["parse_cpu_ms"] += (
                        time.thread_time() - parse_cpu_started
                    ) * 1000
                    self.session.add_message("assistant", full_response)
                else:
                    full_response = scheduled_command.raw
                    parsed = ParsedResponse(text="", commands=[scheduled_command])
                    await emit("agent_scheduled_action", {
                        "iteration": turn + 1,
                        "name": scheduled_command.name,
                        "params": scheduled_command.params,
                    })
                result.text = parsed.text

                if self.session.prompt_slots.planning:
                    valid_plan = (
                        not parsed.parse_failures
                        and not parsed.text.strip()
                        and len(parsed.commands) == 1
                        and parsed.commands[0].type == "tool_call"
                        and parsed.commands[0].name == "plan"
                    )
                    if not valid_plan:
                        planning_failure_count += 1
                        error = (
                            "当前是规划轮，只能调用一次 plan；不得直接回答、调用业务工具或技能。"
                        )
                        self.session.prompt_slots.fail(
                            category="planning_policy", error=error, raw=full_response
                        )
                        await emit("policy", {"message": error, "phase": "planning"})
                        result.text = ""
                        if planning_failure_count >= 2:
                            result.text = "模型未能生成有效计划，本次请求未执行。"
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            break
                        continue

                    plan_cmd = parsed.commands[0]
                    await emit("tool_call", {
                        "iteration": turn + 1,
                        "command_type": plan_cmd.type,
                        "name": plan_cmd.name,
                        "params": plan_cmd.params,
                        "confidence": plan_cmd.confidence,
                    })
                    plan_context = ToolContext(
                        session=self.session,
                        ros_bridge=self.mcp.ros_bridge,
                        image=None,
                    )
                    plan_result = await self.call_router.execute_command(
                        plan_cmd, context=plan_context
                    )
                    result.commands.append((plan_cmd, plan_result))
                    await emit("tool_result", {
                        "iteration": turn + 1,
                        "command_type": plan_cmd.type,
                        "name": plan_cmd.name,
                        "success": plan_result.success,
                        "data": plan_result.data,
                        "error": plan_result.error,
                    })
                    if not plan_result.success:
                        planning_failure_count += 1
                        self.session.prompt_slots.fail(
                            category="planning_error",
                            error=plan_result.error + (
                                ("；" + plan_result.recovery_hint)
                                if plan_result.recovery_hint else ""
                            ),
                            raw=plan_cmd.raw,
                            name="plan",
                            params=plan_cmd.params,
                        )
                        result.text = ""
                        if planning_failure_count >= 2:
                            result.text = "模型生成的计划连续无效，本次请求未执行。"
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            break
                        continue
                    normalized_plan = json.loads(plan_result.data)
                    task_plan = TaskPlan.from_payload(normalized_plan)
                    capabilities = {
                        tool.name: tool.get_harness_metadata()
                        for tool in self.mcp.get_tools_for_mode(self.name)
                        if tool.name not in {"plan", "act"}
                    }
                    capabilities.update({
                        name: ToolMetadata(
                            effect="READ", idempotent=True,
                            produces=(f"skill.{name}.instructions",),
                            max_attempts=1,
                        )
                        for name in getattr(
                            getattr(self.skills, "loader", None), "skills", {}
                        ).keys()
                    })
                    try:
                        task_plan.validate_capabilities(capabilities)
                    except ValueError as exc:
                        planning_failure_count += 1
                        message = f"计划引用了不可执行的工具或事实: {exc}"
                        self.session.prompt_slots.fail(
                            category="plan_incomplete", error=message,
                            raw=plan_cmd.raw, name="plan", params=plan_cmd.params,
                        )
                        await emit("policy", {
                            "message": message, "phase": "planning_validation",
                        })
                        result.text = ""
                        if planning_failure_count >= 2:
                            result.text = "模型连续生成不可执行的计划，本次请求未执行。"
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            break
                        continue
                    self.session.prompt_slots.set_plan(normalized_plan)
                    self.session.execution_state = ExecutionState.create(task_plan)
                    self.session.prompt_slots.append_execution_event(
                        self.session.execution_state.append_event("PLAN_ATTACHED")
                    )
                    await emit("harness_state", {
                        "event": "PLAN_ATTACHED",
                        "state": self.session.execution_state.projection(),
                    })
                    result.text = ""
                    continue

                if parsed.parse_failures:
                    execution_state = self.session.execution_state
                    repaired_terminal = (
                        self.adapt_compact_truncated_terminal_output(
                            full_response, execution_state
                        ) if execution_state is not None else None
                    )
                    if repaired_terminal is not None:
                        parsed = ParsedResponse(text="", commands=[repaired_terminal])
                        await emit("agent_action_repaired", {
                            "reason": "truncated_terminal_to_speak",
                            "name": repaired_terminal.name,
                            "params": repaired_terminal.params,
                        })
                if parsed.parse_failures:
                    for invalid in parsed.parse_failures:
                        if self.session.execution_state is not None:
                            self.session.execution_state.reject_action(
                                "MALFORMED_ACTION", "SYNTAX_ERROR",
                                "工具或技能调用格式无法解析",
                            )
                            self.session.prompt_slots.append_execution_event(
                                self.session.execution_state.append_event("ACTION_REJECTED")
                            )
                        self.session.prompt_slots.fail(
                            category="syntax_error",
                            error="工具或技能调用格式无法解析，请按固定格式修正",
                            raw=invalid,
                            execution_state=(
                                self.session.execution_state.projection()
                                if self.session.execution_state is not None else None
                            ),
                        )
                        await emit("policy", {
                            "message": "工具或技能调用格式无法解析",
                            "source": "parser",
                            "error_code": "MALFORMED_TOOL_CALL",
                        })
                    total_failed_calls += len(parsed.parse_failures)
                    if not parsed.commands:
                        if total_failed_calls >= 4:
                            result.text = "工具调用格式多次错误，已停止继续尝试。"
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            break
                        result.text = ""
                        continue

                if not parsed.commands:
                    logger.info(f"[{self.name}] No commands, plain text: {parsed.text[:80]}...")
                    execution_state = self.session.execution_state
                    if execution_state is not None:
                        message = (
                            "plan 之后禁止裸文本决策；请使用 act(CALL_TOOL/CALL_SKILL/FINISH)。"
                        )
                        execution_state.reject_action(
                            "PLAIN_TEXT", "ACT_WRAPPER_REQUIRED", message
                        )
                        self.session.prompt_slots.append_execution_event(
                            execution_state.append_event("ACT_REJECTED")
                        )
                        self.session.prompt_slots.fail(
                            category="act_wrapper_required", error=message,
                            raw=full_response,
                            execution_state=execution_state.projection(),
                        )
                        await emit("policy", {
                            "message": message, "source": "act_controller",
                            "error_code": "ACT_WRAPPER_REQUIRED",
                        })
                        result.text = ""
                        continue
                    policy_cpu_started = time.thread_time()
                    rejection = self.get_plain_response_rejection(
                        user_input,
                        parsed.text,
                        successful_tools,
                        attempted_tools,
                    )
                    result.metrics["policy_cpu_ms"] += (
                        time.thread_time() - policy_cpu_started
                    ) * 1000
                    if rejection:
                        rejection_code = (
                            rejection.split(":", 1)[0]
                            if re.match(r"^[A-Z][A-Z0-9_]+:", rejection)
                            else "EXECUTION_POLICY"
                        )
                        policy_rejection_count += 1
                        logger.warning(
                            f"[{self.name}] Rejecting ungrounded plain response: {rejection}"
                        )
                        await emit("policy", {"message": rejection})
                        if execution_state is not None:
                            execution_state.reject_action(
                                "FINISH", "EXECUTION_POLICY", rejection
                            )
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("ACTION_REJECTED")
                            )
                        fallback_command = self.get_policy_fallback_command(
                            user_input,
                            successful_tools,
                            attempted_tools,
                        )
                        self.session.add_message(
                            "tool_result",
                            "状态: 失败\n" + rejection,
                            metadata={"source": "execution_policy"},
                        )
                        self.session.prompt_slots.fail(
                            category="execution_policy",
                            error=rejection,
                            raw=full_response,
                            execution_state=(
                                execution_state.projection()
                                if execution_state is not None else None
                            ),
                        )
                        if fallback_command is not None:
                            logger.warning(
                                f"[{self.name}] Applying deterministic policy fallback: "
                                f"{fallback_command.name} params={fallback_command.params}"
                            )
                            await emit("policy", {
                                "message": "模型未调用必需工具，执行确定性导航兜底。",
                                "fallback_command": fallback_command.name,
                                "params": fallback_command.params,
                            })
                            parsed.commands = [fallback_command]
                            result.text = ""
                        elif policy_rejection_count >= 3:
                            result.text = "未能生成可靠的工具调用，本次请求未执行。"
                            await emit("final_output", {
                                "text": result.text,
                                "source": "execution_policy",
                            })
                            final_emitted = True
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            break
                        else:
                            continue
                    else:
                        if execution_state is not None:
                            finish_validation = execution_state.mark_plain_finish()
                            if not finish_validation.allowed:
                                execution_state.reject_action(
                                    "FINISH", finish_validation.error_code,
                                    finish_validation.message,
                                )
                                self.session.prompt_slots.append_execution_event(
                                    execution_state.append_event("FINISH_REJECTED")
                                )
                                self.session.prompt_slots.fail(
                                    category=finish_validation.error_code.lower(),
                                    error=finish_validation.message,
                                    raw=full_response,
                                    execution_state=execution_state.projection(),
                                )
                                await emit("policy", {
                                    "message": finish_validation.message,
                                    "source": "action_validator",
                                })
                                result.text = ""
                                continue
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("FINAL_ACCEPTED")
                            )
                        self.session.compact_current_turn(parsed.text)
                        if execution_state is not None and execution_state.status == "FAILED":
                            trace_status = "error"
                        await emit("final_output", {
                            "text": parsed.text,
                            "source": "plain_text",
                        })
                        final_emitted = True
                        break

                logger.info(
                    f"[{self.name}] Parsed {len(parsed.commands)} commands: "
                    + str([f"{c.type}:{c.name}" for c in parsed.commands])
                )
                if len(parsed.commands) != 1:
                    message = "Harness 每轮只允许一个动作；请只输出一个工具或 Skill 调用。"
                    if self.session.execution_state is not None:
                        self.session.execution_state.reject_action(
                            "MULTIPLE_ACTIONS", "MULTIPLE_ACTIONS", message
                        )
                        self.session.prompt_slots.append_execution_event(
                            self.session.execution_state.append_event("ACTION_REJECTED")
                        )
                    self.session.prompt_slots.fail(
                        category="multiple_actions", error=message, raw=full_response,
                        execution_state=(
                            self.session.execution_state.projection()
                            if self.session.execution_state is not None else None
                        ),
                    )
                    await emit("policy", {"message": message, "source": "action_validator"})
                    result.text = ""
                    continue
                execution_state = self.session.execution_state
                if execution_state is not None:
                    act_command = parsed.commands[0]
                    truncated_terminal = self.adapt_compact_truncated_terminal_output(
                        full_response, execution_state
                    )
                    if truncated_terminal is not None:
                        parsed.commands = [truncated_terminal]
                        await emit("agent_action_repaired", {
                            "reason": "truncated_terminal_to_speak",
                            "name": truncated_terminal.name,
                            "params": truncated_terminal.params,
                        })
                        unwrapped = None
                        act_error = None
                    elif act_command.type == "tool_call" and act_command.name == "act":
                        # Read old trajectories/clients during migration, while
                        # all newly generated prompts use direct business tools.
                        unwrapped, act_error = self._unwrap_act_command(
                            act_command, execution_state
                        )
                    else:
                        unwrapped, act_error = act_command, None
                    if act_error is not None:
                        error_code, message = act_error
                        execution_state.reject_action("ACT", error_code, message)
                        self.session.prompt_slots.append_execution_event(
                            execution_state.append_event("ACT_REJECTED")
                        )
                        self.session.prompt_slots.fail(
                            category=error_code.lower(), error=message,
                            raw=act_command.raw, name=act_command.name,
                            params=act_command.params,
                            execution_state=execution_state.projection(),
                        )
                        await emit("policy", {
                            "message": message,
                            "source": "act_controller",
                            "error_code": error_code,
                        })
                        result.text = ""
                        continue
                    if (
                        truncated_terminal is None
                        and act_command.type == "tool_call"
                        and act_command.name == "act"
                    ):
                        await emit("act_decision", {
                            "step_id": execution_state.active_step_id,
                            "action_type": unwrapped.type,
                            "name": unwrapped.name,
                            "arguments": unwrapped.params,
                        })
                    elif truncated_terminal is None:
                        await emit("direct_tool_decision", {
                            "step_id": execution_state.active_step_id,
                            "command_type": unwrapped.type,
                            "name": unwrapped.name,
                            "arguments": unwrapped.params,
                        })
                    else:
                        # It is already an executable speak command; do not
                        # attempt to unwrap a partial model act envelope.
                        unwrapped = truncated_terminal
                    adapted_terminal = self.adapt_compact_terminal_action(
                        act_command, execution_state
                    ) if (
                        truncated_terminal is None
                        and act_command.type == "tool_call"
                        and act_command.name == "act"
                    ) else None
                    if adapted_terminal is not None:
                        parsed.commands = [adapted_terminal]
                        await emit("agent_action_repaired", {
                            "reason": "terminal_finish_to_speak",
                            "name": adapted_terminal.name,
                            "params": adapted_terminal.params,
                        })
                    elif unwrapped.type == "judge" and unwrapped.name == "finish":
                        if self.name in {"Voice", "Benchmark"}:
                            message = (
                                "SESSION_NOT_TERMINATED: Voice/Benchmark 会话必须调用 speak "
                                "结束；不得使用 FINISH 代替播报。"
                            )
                            execution_state.reject_action(
                                "FINISH", "SESSION_NOT_TERMINATED", message
                            )
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("FINISH_REJECTED")
                            )
                            self.session.prompt_slots.fail(
                                category="session_not_terminated", error=message,
                                raw=act_command.raw, name="act",
                                params=act_command.params,
                                execution_state=execution_state.projection(),
                            )
                            await emit("policy", {
                                "message": message, "source": "act_controller",
                                "error_code": "SESSION_NOT_TERMINATED",
                            })
                            result.text = ""
                            continue
                        finish_validation = execution_state.mark_plain_finish()
                        if not finish_validation.allowed:
                            execution_state.reject_action(
                                "FINISH", finish_validation.error_code,
                                finish_validation.message,
                            )
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("FINISH_REJECTED")
                            )
                            self.session.prompt_slots.fail(
                                category=finish_validation.error_code.lower(),
                                error=finish_validation.message,
                                raw=act_command.raw, name="act",
                                params=act_command.params,
                                execution_state=execution_state.projection(),
                            )
                            result.text = ""
                            continue
                        final_response = str(unwrapped.params["response"])
                        self.session.prompt_slots.append_execution_event(
                            execution_state.append_event("FINAL_ACCEPTED")
                        )
                        self.session.compact_current_turn(final_response)
                        if execution_state.status == "FAILED":
                            trace_status = "error"
                        result.text = final_response
                        result.turn_end_reason = "finish"
                        await emit("final_output", {
                            "text": final_response, "source": "act_finish",
                        })
                        final_emitted = True
                        break
                    else:
                        parsed.commands = [unwrapped]
                if final_emitted:
                    break
                speak_done = False
                tool_context = ToolContext(
                    session=self.session,
                    ros_bridge=self.mcp.ros_bridge,
                    image=image,
                    extra={
                        "voice_mode": self if self.name == "Voice" else None,
                        **(tool_context_extra or {}),
                    },
                )
                for cmd in parsed.commands:
                    grounded = self.adapt_grounded_speak_command(
                        cmd, self.session.execution_state
                    )
                    if grounded is not None:
                        cmd = grounded
                        await emit("agent_action_repaired", {
                            "reason": "unsupported_department_removed",
                            "name": cmd.name,
                            "params": cmd.params,
                        })
                    await emit("tool_call", {
                        "iteration": turn + 1,
                        "command_type": cmd.type,
                        "name": cmd.name,
                        "params": cmd.params,
                        "confidence": cmd.confidence,
                    })
                    policy_cpu_started = time.thread_time()
                    rejection = self.get_command_rejection(
                        user_input,
                        cmd,
                        successful_tools,
                        attempted_tools,
                    )
                    result.metrics["policy_cpu_ms"] += (
                        time.thread_time() - policy_cpu_started
                    ) * 1000
                    if rejection:
                        rejection_code = (
                            rejection.split(":", 1)[0]
                            if re.match(r"^[A-Z][A-Z0-9_]+:", rejection)
                            else "EXECUTION_POLICY"
                        )
                        policy_rejection_count += 1
                        logger.warning(
                            f"[{self.name}] Rejecting command {cmd.name}: {rejection}"
                        )
                        await emit("policy", {
                            "message": rejection,
                            "command": cmd.name,
                            "error_code": rejection_code,
                        })
                        if self.session.execution_state is not None:
                            self.session.execution_state.reject_action(
                                cmd.name, rejection_code, rejection
                            )
                            self.session.prompt_slots.append_execution_event(
                                self.session.execution_state.append_event("ACTION_REJECTED")
                            )
                        fallback_command = self.get_policy_fallback_command(
                            user_input,
                            successful_tools,
                            attempted_tools,
                        )
                        self.session.add_message(
                            "tool_result",
                            "状态: 失败\n" + rejection,
                            metadata={"source": "execution_policy"},
                        )
                        self.session.prompt_slots.fail(
                            category="execution_policy",
                            error=rejection,
                            raw=cmd.raw,
                            name=cmd.name,
                            params=cmd.params,
                        )
                        if fallback_command is not None:
                            logger.warning(
                                f"[{self.name}] Replacing rejected {cmd.name} with "
                                f"policy fallback {fallback_command.name}"
                            )
                            cmd = fallback_command
                            await emit("tool_call", {
                                "iteration": turn + 1,
                                "command_type": cmd.type,
                                "name": cmd.name,
                                "params": cmd.params,
                                "confidence": cmd.confidence,
                                "source": "execution_policy_fallback",
                            })
                        elif policy_rejection_count >= 3:
                            result.text = "未能生成可靠的工具调用，本次请求未执行。"
                            await emit("final_output", {
                                "text": result.text,
                                "source": "execution_policy",
                            })
                            final_emitted = True
                            trace_status = "error"
                            self.session.compact_current_turn(result.text)
                            repeated_failure = True
                        else:
                            break
                        if repeated_failure:
                            break

                    signature = (
                        cmd.type,
                        cmd.name,
                        json.dumps(cmd.params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    )
                    execution_state = self.session.execution_state
                    tool_meta = ToolMetadata()
                    if execution_state is not None and cmd.type in {"tool_call", "skill_call"}:
                        tool = self.mcp.tools.get(cmd.name) if cmd.type == "tool_call" else None
                        if tool is not None:
                            tool_meta = tool.get_harness_metadata()
                        elif cmd.type == "skill_call":
                            tool_meta = ToolMetadata(
                                effect="READ",
                                idempotent=True,
                                produces=(f"skill.{cmd.name}.instructions",),
                                max_attempts=1,
                            )
                        available_actions = (
                            [
                                item.name for item in self.mcp.get_tools_for_mode(self.name)
                                if item.name not in {"plan", "act"}
                            ]
                            if cmd.type == "tool_call"
                            else list(getattr(
                                getattr(self.skills, "loader", None), "skills", {}
                            ).keys())
                        )
                        validation = self.action_validator.validate_tool(
                            state=execution_state,
                            tool=cmd.name,
                            arguments=cmd.params,
                            metadata=tool_meta,
                            available_tools=available_actions,
                        )
                        if not validation.allowed:
                            hint = (
                                "请缩短关键词、替换同义词或改变查询句式。"
                                if cmd.name in {"med_query", "medical_consult"}
                                else "请遵循 CURRENT STEP 和最新事实。"
                            )
                            message = validation.message + hint
                            execution_state.reject_action(
                                cmd.name, validation.error_code, message
                            )
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("ACTION_REJECTED")
                            )
                            self.session.prompt_slots.fail(
                                category=validation.error_code.lower(), error=message,
                                raw=cmd.raw, name=cmd.name, params=cmd.params,
                                execution_state=execution_state.projection(),
                            )
                            self.session.add_message(
                                "tool_result", "状态: 失败\n" + message,
                                metadata={"source": "action_validator"},
                            )
                            await emit("policy", {
                                "message": message, "command": cmd.name,
                                "error_code": validation.error_code,
                            })
                            if validation.error_code == "DUPLICATE_NO_WORLD_CHANGE":
                                blocked_duplicate_counts[signature] = (
                                    blocked_duplicate_counts.get(signature, 0) + 1
                                )
                                if blocked_duplicate_counts[signature] >= 2:
                                    result.text = "工具连续重复且世界状态未变化，已停止执行。"
                                    trace_status = "error"
                                    self.session.compact_current_turn(result.text)
                                    repeated_failure = True
                            if execution_state.status == "BUDGET_EXHAUSTED":
                                result.text = "任务动作预算已耗尽，已停止执行。"
                                trace_status = "error"
                                self.session.compact_current_turn(result.text)
                                repeated_failure = True
                            break

                    call_result = await self.call_router.execute_command(
                        cmd, context=tool_context
                    )
                    result.metrics["tool_wall_ms"] += call_result.duration_ms
                    result.metrics["tool_cpu_ms"] += call_result.cpu_ms
                    if cmd.type == "tool_call":
                        attempted_tools.add(cmd.name)
                    if execution_state is not None and cmd.type in {"tool_call", "skill_call"}:
                        error_type = call_result.error_type
                        if not error_type and "超时" in (call_result.error or ""):
                            error_type = "TIMEOUT"
                        observation_facts = dict(call_result.facts)
                        if cmd.type == "skill_call" and call_result.success:
                            observation_facts.setdefault(
                                f"skill.{cmd.name}.instructions", call_result.data
                            )
                        if (
                            execution_state.status in {"GOAL_SATISFIED", "BLOCKED", "FAILED"}
                            and execution_state.active_step is None
                            and tool_meta.session_terminal
                        ):
                            execution_state.apply_post_goal_terminal_result(
                                tool=cmd.name,
                                arguments=cmd.params,
                                metadata=tool_meta,
                                success=call_result.success,
                                facts=observation_facts,
                                error_type=error_type,
                                retryable=call_result.retryable,
                                observation=call_result.data or call_result.error,
                            )
                        else:
                            execution_state.apply_tool_result(
                                tool=cmd.name,
                                arguments=cmd.params,
                                metadata=tool_meta,
                                success=call_result.success,
                                facts=observation_facts,
                                error_type=error_type,
                                retryable=call_result.retryable,
                                observation=call_result.data or call_result.error,
                            )
                        harness_event_type = (
                            "ACTION_APPLIED" if call_result.success else "ACTION_FAILED"
                        )
                        self.session.prompt_slots.append_execution_event(
                            execution_state.append_event(harness_event_type)
                        )
                        await emit("harness_state", {
                            "event": harness_event_type,
                            "state": execution_state.projection(),
                        })
                    await emit("tool_result", {
                        "iteration": turn + 1,
                        "command_type": cmd.type,
                        "name": cmd.name,
                        "success": call_result.success,
                        "data": call_result.data,
                        "error": call_result.error,
                        "error_type": call_result.error_type,
                        "empty": call_result.empty,
                        "retryable": call_result.retryable,
                        "recovery_hint": call_result.recovery_hint,
                        "diagnostics": call_result.diagnostics,
                    })
                    result.commands.append((cmd, call_result))
                    result_text = (
                        self.mcp.format_result(cmd.name, call_result)
                        if cmd.type == "tool_call"
                        else call_result.data
                    )
                    if call_result.success and cmd.type == "tool_call":
                        successful_tools.add(cmd.name)
                    commit_cpu_started = time.thread_time()
                    if call_result.success:
                        self.session.prompt_slots.commit(
                            command_type=cmd.type,
                            name=cmd.name,
                            params=cmd.params,
                            model_output=cmd.raw,
                            result=result_text,
                        )
                    if not call_result.success:
                        total_failed_calls += 1
                        if self._record_failed_tool_attempt(cmd, call_result):
                            self.session.prompt_slots.fail(
                                category="tool_failure",
                                error=(call_result.error or call_result.data or "工具调用失败")
                                + (("；" + call_result.recovery_hint) if call_result.recovery_hint else ""),
                                raw=cmd.raw,
                                name=cmd.name,
                                params=cmd.params,
                                execution_state=(
                                    execution_state.projection()
                                    if execution_state is not None else None
                                ),
                            )
                    result.metrics["session_commit_cpu_ms"] += (
                        time.thread_time() - commit_cpu_started
                    ) * 1000

                    # query ends this external turn and hands its recording to
                    # the Voice server; speak ends both the turn and session.
                    if (
                        cmd.type == "tool_call"
                        and cmd.name == "query"
                        and call_result.success
                    ):
                        result.text = str(cmd.params.get("question", ""))
                        result.turn_end_reason = "query"
                        result.session_ended = False
                        result.continuation_pcm = bytes(
                            call_result.transient.get("recording_pcm", b"")
                        )
                        result.continuation_audio = dict(
                            call_result.transient.get("audio", {})
                        )
                        followup_text = str(
                            call_result.transient.get("followup_text", "")
                        ).strip()
                        if followup_text:
                            result.continuation_audio["followup_text"] = followup_text
                        if execution_state is not None:
                            consultation = execution_state.facts.get(
                                "medical.consultation"
                            )
                            resume_input = actual_user_input
                            if consultation is not None and consultation.valid:
                                medical_value = consultation.value
                                if isinstance(medical_value, dict):
                                    retrieval = medical_value.get("retrieval")
                                    if isinstance(retrieval, dict):
                                        resume_input = str(
                                            retrieval.get("medical_query")
                                            or medical_value.get("query")
                                            or resume_input
                                        )
                            self.session.set_pending_dialogue(
                                root_user_input=(
                                    str(continuation.get("root_user_input", ""))
                                    if continuation is not None
                                    else actual_user_input
                                ),
                                question=result.text,
                                source_execution_id=execution_state.execution_id,
                                resume_user_input=resume_input,
                                completed_tools=sorted(successful_tools),
                                prior=continuation,
                            )
                            execution_state.status = "FINISHED"
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("TURN_AWAITING_INPUT")
                            )
                        self.session.compact_current_turn(
                            result.text, status="awaiting_input", end_reason="query"
                        )
                        await emit("final_output", {
                            "text": result.text,
                            "source": "query",
                            "session_ended": False,
                        })
                        final_emitted = True
                        speak_done = True
                        trace_status = "awaiting_input"
                        break

                    if (
                        cmd.type == "tool_call"
                        and cmd.name == "navigate"
                        and call_result.success
                        and execution_state is not None
                        and self.is_compact_navigation_announcement_terminal(
                            cmd, call_result, execution_state
                        )
                    ):
                        result.text = str(cmd.params.get("announcement", "")).strip()
                        result.turn_end_reason = "speak"
                        result.session_ended = True
                        if execution_state.status == "GOAL_SATISFIED":
                            execution_state.status = "FINISHED"
                        self.session.prompt_slots.append_execution_event(
                            execution_state.append_event("FINAL_ACCEPTED")
                        )
                        self.session.compact_current_turn(result.text, end_reason="speak")
                        await emit("final_output", {
                            "text": result.text,
                            "source": "navigate_announcement",
                        })
                        final_emitted = True
                        speak_done = True
                        break

                    if (
                        cmd.type == "tool_call"
                        and cmd.name == "speak"
                        and call_result.success
                        and (
                            execution_state is None
                            or execution_state.status in {
                                "GOAL_SATISFIED", "BLOCKED", "FAILED"
                            }
                        )
                    ):
                        logger.info(f"[{self.name}] speak completed, stopping loop")
                        result.text = str(cmd.params.get("text", ""))
                        result.turn_end_reason = "speak"
                        result.session_ended = True
                        if execution_state is not None:
                            if execution_state.status in {
                                "GOAL_SATISFIED", "BLOCKED", "FAILED"
                            }:
                                execution_state.status = "FINISHED"
                            self.session.prompt_slots.append_execution_event(
                                execution_state.append_event("FINAL_ACCEPTED")
                            )
                        self.session.compact_current_turn(
                            result.text, end_reason="speak"
                        )
                        await emit("final_output", {
                            "text": str(cmd.params.get("text", "")),
                            "source": "speak",
                        })
                        final_emitted = True
                        speak_done = True
                        break

                    role = "tool_result" if cmd.type == "tool_call" else "skill_result"
                    self.session.add_message(
                        role,
                        result_text,
                        metadata={"source": cmd.name},
                    )
                    if total_failed_calls >= 4:
                        logger.error(
                            f"[{self.name}] Too many failed tool calls; stopping loop"
                        )
                        result.text = "工具调用多次失败，已停止继续尝试。"
                        await emit("final_output", {
                            "text": result.text,
                            "source": "tool_failure_limit",
                        })
                        final_emitted = True
                        trace_status = "error"
                        self.session.compact_current_turn(result.text)
                        repeated_failure = True
                        break

                if (
                    speak_done
                    or repeated_failure
                ):
                    break
        except Exception as exc:
            self.session.compact_current_turn("本轮请求处理失败。")
            trace_status = "error"
            await emit("error", {
                "message": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            if not result.turn_end_reason:
                trace_status = "error"
                await emit("state_error", {
                    "error_code": "TURN_NOT_TERMINATED",
                    "message": "本轮未以 query 或 speak 结束。",
                })
            if not final_emitted and not result.text and trace_status == "completed":
                trace_status = "error"
            if not self.session._active_turn_finalized:
                fallback = result.text or (
                    "本轮请求处理失败。" if trace_status == "error" else "本轮未生成最终答复。"
                )
                self.session.finish_external_turn(
                    fallback, status=trace_status,
                    end_reason=result.turn_end_reason or "not_terminated",
                )
            self.skills.deactivate_all()
            if not final_emitted and result.text:
                await emit("final_output", {
                    "text": result.text,
                    "source": "agent_result",
                })
            await emit("turn_end", {
                "status": trace_status,
                "duration_ms": round((time.monotonic() - trace_started) * 1000),
                "tool_count": len(result.commands),
            })
            result.metrics["agent_e2e_ms"] = (
                time.monotonic() - loop_wall_started
            ) * 1000
            result.metrics["agent_main_thread_cpu_ms"] = (
                time.thread_time() - loop_cpu_started
            ) * 1000
            result.metrics["agent_process_cpu_ms_raw"] = (
                time.process_time() - loop_process_cpu_started
            ) * 1000
            slots = self.session.prompt_slots
            result.metrics["execution_event_count"] = len(slots.execution_events)
            result.metrics["visible_execution_event_count"] = len(
                slots.visible_execution_events
            )
            result.metrics["execution_compaction_count"] = (
                slots.execution_compaction_count
            )
            result.metrics["visible_execution_event_chars"] = len(json.dumps(
                slots.visible_execution_events,
                ensure_ascii=False,
                separators=(",", ":"),
            ))
            result.metrics["attempt_chars"] = len(json.dumps(
                slots.attempt, ensure_ascii=False, separators=(",", ":")
            ))
            result.metrics["prompt_preflight"] = dict(
                self.session.last_prompt_preflight
            )
            final_text = result.text
            if self.session.conversation_turns:
                latest = self.session.conversation_turns[-1]
                if latest.task_id == self.session.prompt_slots.task_id:
                    final_text = latest.assistant
            writer = getattr(self.api, "write_trajectory", None)
            if writer is not None:
                try:
                    await asyncio.to_thread(
                        writer,
                        session=self.session,
                        trace_id=trace_id,
                        mode=self.name,
                        status=trace_status,
                        final_text=final_text,
                        metrics=result.metrics,
                        events=trajectory_events,
                    )
                except Exception as exc:
                    logger.warning("Trajectory write failed for %s: %s", trace_id, exc)
        return result
