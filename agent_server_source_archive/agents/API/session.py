from __future__ import annotations

import json
import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    image: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_api_dict(self) -> dict:
        if self.role == "tool_result" or self.role == "skill_result":
            label = "工具" if self.role == "tool_result" else "技能"
            source = self.metadata.get("source", "")
            return {
                "role": "user",
                "content": f"[{label}结果: {source}]\n{self.content}",
            }

        if self.role == "system":
            return {"role": "system", "content": self.content}

        if self.image is not None:
            image_url = (
                self.image
                if self.image.startswith("data:")
                else f"data:image/jpeg;base64,{self.image}"
            )
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                ],
            }

        if self.role == "assistant":
            return {"role": "assistant", "content": self.content}

        return {"role": "user", "content": self.content}


@dataclass
class ConversationTurn:
    index: int
    task_id: str
    user: str
    assistant: str
    status: str = "completed"
    started_at: float = 0.0
    completed_at: float = 0.0
    end_reason: str = ""
    input_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "task_id": self.task_id,
            "user": self.user,
            "assistant": self.assistant,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "end_reason": self.end_reason,
            "input_metadata": self.input_metadata,
        }


@dataclass
class PendingDialogue:
    """Small, explicit continuation contract between external turns."""

    continuation_id: str
    source_task_id: str
    source_execution_id: str
    root_user_input: str
    resume_user_input: str
    question: str
    answers: List[str] = field(default_factory=list)
    completed_tools: List[str] = field(default_factory=list)
    resume_tool: str = "medical_consult"
    depth: int = 1
    status: str = "WAITING_INPUT"
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def to_prompt_dict(self, current_answer: str = "") -> Dict[str, Any]:
        answers = list(self.answers)
        if current_answer.strip():
            answers.append(current_answer.strip())
        return {
            "schema_version": "pending-dialogue.v1",
            "continuation_id": self.continuation_id,
            "kind": "medical_followup",
            "source_task_id": self.source_task_id,
            "root_user_input": self.root_user_input,
            "resume_user_input": self.resume_user_input,
            "question": self.question,
            "answers": answers,
            "resume_tool": self.resume_tool,
            "depth": self.depth,
            "completed_tools": list(self.completed_tools),
            "instruction": (
                "本轮输入是上轮追问的回答；将原始问题、追问和补充回答合并后继续医疗查询，"
                "不要把补充回答当成独立问题。"
            ),
        }


@dataclass
class HistoryLedger:
    """Archive every committed tool call while exposing only a bounded tail."""

    max_visible: int = 8
    entries: List[Dict[str, Any]] = field(default_factory=list)
    _next_index: int = 1

    def set_max_visible(self, value: int) -> None:
        if value < 1:
            raise ValueError("history max_visible must be at least 1")
        self.max_visible = value

    def clear(self) -> None:
        self.entries.clear()
        self._next_index = 1

    def append(
        self,
        *,
        task_id: str,
        command_type: str,
        name: str,
        params: Dict[str, Any],
        model_output: str,
        result: str,
    ) -> None:
        self.entries.append({
            "index": self._next_index,
            "task_id": task_id,
            "type": command_type,
            "name": name,
            "params": dict(sorted(params.items())),
            "model_output": model_output,
            "result": (
                result if len(result) <= 2000 else result[:2000] + "…[truncated]"
            ),
        })
        self._next_index += 1

    def visible_entries(
        self, exclude_task_id: str = "", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        entries = (
            [item for item in self.entries if item.get("task_id") != exclude_task_id]
            if exclude_task_id else self.entries
        )
        visible_limit = self.max_visible if limit is None else max(0, int(limit))
        selected = entries[-visible_limit:] if visible_limit else []
        rendered = copy.deepcopy(selected)
        for item in rendered:
            if item.get("name") != "medical_consult":
                continue
            raw = str(item.get("result", "") or "")
            payload: Dict[str, Any] = {}
            start = raw.find("{")
            if start >= 0:
                try:
                    parsed = json.loads(raw[start:])
                    if isinstance(parsed, dict):
                        payload = parsed
                except json.JSONDecodeError:
                    payload = {}
            item["result"] = json.dumps({
                "status": payload.get("status", "recorded"),
                "intent": payload.get("intent", ""),
                "evidence_count": len(
                    payload.get("evidence")
                    if isinstance(payload.get("evidence"), list) else []
                ),
                "followup_question_count": len(
                    payload.get("questions")
                    if isinstance(payload.get("questions"), list) else []
                ),
                "note": "完整医疗事实未在跨任务历史重复；续问必须使用 pending_dialogue 重新查询。",
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return rendered

    @property
    def hidden_count(self) -> int:
        return max(0, len(self.entries) - len(self.visible_entries()))


@dataclass
class PromptSlots:
    """Deterministic prompt-cache segments backed by a session ledger."""

    system: str = ""
    conversation: str = ""
    user: str = ""
    user_image: Optional[str] = None
    history: HistoryLedger = field(default_factory=HistoryLedger)
    attempt: List[Dict[str, Any]] = field(default_factory=list)
    task_id: str = ""
    external_turn: int = 0
    planning: bool = False
    plan: Optional[Dict[str, Any]] = None
    execution_events: List[Dict[str, Any]] = field(default_factory=list)
    visible_execution_events: List[Dict[str, Any]] = field(default_factory=list)
    execution_history_max_chars: int = 6500
    execution_compaction_count: int = 0
    request_history_max_visible: Optional[int] = None
    preflight_compaction_count: int = 0
    _base_user: str = ""

    PLANNING_DIRECTIVE = (
        '<system kind="planning">这是当前任务的第一次模型调用。你必须先调用 plan 工具，'
        '本轮不得回答用户、调用业务工具或执行技能。输出格式：'
        '<tool>{"tool_call":"plan","param":{"goal":"最终目标",'
        '"success_conditions":[],"steps":[{"step_id":"s1","goal":"步骤目的",'
        '"preferred_tool":null,"depends_on":[],"condition":null,'
        '"verification":false}],"done_when":"完成标准"}}</tool>'
        '计划只描述语义目标和依赖，不预先固定未来工具参数。success_conditions '
        '只能引用工具说明中的 Harness事实；WRITE 后如需确认结果，增加 verification 步骤。'
        '普通直接回答可以只设一个无 preferred_tool 的步骤。</system>'
    )

    @staticmethod
    def _dump(records: List[Dict[str, Any]]) -> str:
        if not records:
            return "[]"
        return json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def start_task(
        self,
        *,
        system: str,
        user_input: str,
        image: Optional[str] = None,
        conversation_context: str = "",
        runtime_context: str = "",
        sensor_data: str = "",
        task_policy: str = "",
        pending_dialogue: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.system and self.system != system:
            raise RuntimeError("system prompt slot changed inside an existing session")
        self.system = system
        self.task_id = uuid.uuid4().hex[:16]
        self.external_turn += 1
        task_parts = [f"task_id: {self.task_id}", "用户输入:", user_input]
        self.conversation = conversation_context
        if runtime_context:
            task_parts.extend(["外部上下文:", runtime_context])
        if sensor_data:
            task_parts.extend(["本轮传感器信息:", sensor_data])
        if task_policy:
            task_parts.extend(["本轮执行约束:", task_policy])
        if pending_dialogue:
            task_parts.extend([
                "待续对话状态:",
                "<pending_dialogue>"
                + json.dumps(
                    pending_dialogue,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "</pending_dialogue>",
            ])
        self.user = "\n".join(task_parts)
        self._base_user = self.user
        self.user_image = image
        # Successful work is a session-level archive.  A new user task changes
        # the user slot, but must not erase committed work from earlier tasks.
        self.attempt.clear()
        # Direct-action protocol: the model chooses one real business tool per
        # turn.  Deterministic workflows may still attach an internal plan for
        # validation, but no model-only PLAN round is required.
        self.planning = False
        self.plan = None
        self.execution_events.clear()
        self.visible_execution_events.clear()
        self.execution_compaction_count = 0
        self.request_history_max_visible = None
        self.preflight_compaction_count = 0

    def set_plan(self, plan: Dict[str, Any]) -> None:
        """Freeze a normalized, visible task plan into the user slot."""
        self.plan = plan
        self.planning = False
        plan_json = json.dumps(
            plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.user = (
            self._base_user
            + "\n<plan>" + plan_json + "</plan>"
            + "\n<instruction>按照当前步骤每轮直接调用一个业务工具。"
              "工具名和参数必须符合可用工具说明，最终答复调用 speak。"
              "不得调用 plan 或 act，不得输出裸文本，也不得自行修改或声明执行状态。"
              "Harness 会在执行前验证步骤、参数、事实和安全策略。</instruction>"
        )
        self.attempt.clear()

    def append_execution_event(self, event: Dict[str, Any]) -> None:
        """Keep full diagnostics and a separately bounded model-visible stream."""
        diagnostic = copy.deepcopy(event)
        visible = copy.deepcopy(event)
        self.execution_events.append(diagnostic)
        self.visible_execution_events.append(visible)
        rendered = "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in self.visible_execution_events
        )
        if len(rendered) <= self.execution_history_max_chars:
            return

        # Compact only when crossing the threshold. The newest full projection
        # is authoritative; subsequent events append to this new stable prefix.
        latest = copy.deepcopy(self.visible_execution_events[-1])
        latest["compacted_prior_events"] = max(
            0, len(self.visible_execution_events) - 1
        )
        state = latest.get("state")
        if isinstance(state, dict):
            facts = state.get("known_facts")
            if isinstance(facts, dict):
                state["known_facts"] = {
                    key: (
                        value if not isinstance(value, str) or len(value) <= 500
                        else value[:500] + "…[truncated]"
                    )
                    for key, value in facts.items()
                }
        self.visible_execution_events[:] = [latest]
        self.execution_compaction_count += 1

    def commit(
        self,
        *,
        command_type: str,
        name: str,
        params: Dict[str, Any],
        model_output: str,
        result: str,
    ) -> None:
        self.history.append(
            task_id=self.task_id,
            command_type=command_type,
            name=name,
            params=params,
            model_output=model_output,
            result=result,
        )
        # Attempts describe only failures since the latest committed success.
        self.attempt.clear()

    def fail(
        self,
        *,
        category: str,
        error: str,
        raw: str = "",
        name: str = "",
        params: Optional[Dict[str, Any]] = None,
        execution_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "index": len(self.attempt) + 1,
            "category": category,
            "error": error,
        }
        if raw:
            record["raw"] = raw if len(raw) <= 1200 else raw[:1200] + "…[truncated]"
        if name:
            record["name"] = name
        if params:
            record["params"] = dict(sorted(params.items()))
        if execution_state is not None:
            # The full authoritative state already exists in the append-only
            # execution event. Attempts need only enough context to repair the
            # rejected action; duplicating medical facts here rapidly exhausts 8K.
            record["execution_state"] = {
                key: copy.deepcopy(execution_state.get(key))
                for key in (
                    "execution_id", "status", "current_step_id",
                    "current_step_detail", "last_action", "budgets",
                ) if key in execution_state
            }
        self.attempt.append(record)
        # Only the newest repair context is useful to the model. Full failures
        # remain available in trajectory/model-call telemetry; keeping an
        # unbounded prompt copy caused repeated invalid actions to exceed 8K.
        del self.attempt[:-2]

    def to_request_dict(self) -> Dict[str, Any]:
        if not self.system or not self.user:
            raise RuntimeError("prompt slots are not initialized")

        # Keep the current task's base byte-stable: its successful observations
        # are already represented by append-only execution events.  The full
        # raw success record remains in the ledger and becomes prior-task
        # history after the next external user turn changes the user slot.
        history_text = self._dump(
            self.history.visible_entries(
                exclude_task_id=self.task_id if self.visible_execution_events else "",
                limit=self.request_history_max_visible,
            )
        )
        if self.visible_execution_events:
            history_text += "\n" + "\n".join(
                "<execution_state_event>"
                + json.dumps(
                    event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "</execution_state_event>"
                for event in self.visible_execution_events
            )
        if self.planning:
            history_text += "\n" + self.PLANNING_DIRECTIVE
        result: Dict[str, Any] = {
            "version": "suha.v3",
            "system": self.system,
            "conversation": self.conversation,
            "user": self.user,
            "history": history_text,
            "attempt": self._dump(self.attempt),
        }
        if self.user_image is not None:
            result["image"] = (
                self.user_image
                if self.user_image.startswith("data:")
                else f"data:image/jpeg;base64,{self.user_image}"
            )
        return result

    @staticmethod
    def _keep_text_tail(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        marker = "…[preflight_compacted]\n"
        return marker + value[-max(0, max_chars - len(marker)):]

    def compact_for_token_budget(self, token_counter, token_budget: int) -> Dict[str, Any]:
        """Apply deterministic late-slot compaction until a prompt fits.

        System and current user/plan slots are never rewritten.  Compaction
        starts with prior-task history and only then touches repair attempts or
        older conversation text.  The latest authoritative execution event is
        retained throughout.
        """
        budget = max(1, int(token_budget))

        def count() -> int:
            return int(token_counter(self.to_request_dict()))

        initial = count()
        current = initial
        actions: List[str] = []
        if current <= budget:
            return {
                "initial_prompt_tokens": initial,
                "final_prompt_tokens": current,
                "token_budget": budget,
                "compacted": False,
                "actions": actions,
                "fits": True,
            }

        # Previous tasks are diagnostic context, while the current execution
        # event contains the facts needed for the next action.
        configured = self.history.max_visible
        for limit in (min(configured, 4), 2, 0):
            if self.request_history_max_visible == limit:
                continue
            self.request_history_max_visible = limit
            actions.append(f"history_limit:{limit}")
            current = count()
            if current <= budget:
                break

        if current > budget and len(self.visible_execution_events) > 1:
            latest = copy.deepcopy(self.visible_execution_events[-1])
            latest["compacted_prior_events"] = max(
                int(latest.get("compacted_prior_events", 0) or 0),
                len(self.visible_execution_events) - 1,
            )
            self.visible_execution_events[:] = [latest]
            actions.append("execution_events:latest")
            current = count()

        if current > budget and len(self.attempt) > 1:
            self.attempt[:] = self.attempt[-1:]
            actions.append("attempt:latest")
            current = count()
        if current > budget and self.attempt:
            self.attempt.clear()
            actions.append("attempt:drop")
            current = count()

        for chars in (400, 250, 120):
            if current <= budget or len(self.conversation) <= chars:
                continue
            self.conversation = self._keep_text_tail(self.conversation, chars)
            actions.append(f"conversation_tail:{chars}")
            current = count()

        if actions:
            self.preflight_compaction_count += 1
        return {
            "initial_prompt_tokens": initial,
            "final_prompt_tokens": current,
            "token_budget": budget,
            "compacted": bool(actions),
            "actions": actions,
            "fits": current <= budget,
        }


@dataclass
class Session:
    mode: str = "Voice"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    history: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    need_clear: bool = False
    task_tree: Optional[dict] = None
    prompt_slots: PromptSlots = field(default_factory=PromptSlots)
    last_usage: Dict[str, int] = field(default_factory=dict)
    last_prompt_preflight: Dict[str, Any] = field(default_factory=dict)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    conversation_max_turns: int = 3
    conversation_max_chars: int = 600
    _active_user: str = ""
    _active_turn_finalized: bool = True
    _active_turn_started_at: float = 0.0
    _active_input_metadata: Dict[str, Any] = field(default_factory=dict)
    model_call_records: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_context: Dict[str, Any] = field(default_factory=dict)
    execution_state: Optional[Any] = None
    pending_dialogue: Optional[PendingDialogue] = None
    # 增量合并后的非 system 消息（用于 API 构建时避免全量遍历）
    _merged: List[Message] = field(default_factory=list)
    # 摘要缓存
    _summary_cache: Optional[str] = None
    _summary_cache_valid: bool = False

    def add_message(self, role: str, content: str, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.history.append(msg)
        self.last_active = time.time()
        # 让缓存失效
        self._summary_cache_valid = False
        # 非 system 消息追加到增量合并列表
        if role != "system":
            self._append_merged(msg)
        return msg

    def _append_merged(self, msg: Message):
        """将一条非 system 消息增量合并到 _merged 列表末尾。"""
        if not self._merged:
            self._merged.append(msg)
            return

        # Preserve enough context for multi-step tool recovery.  Ten messages
        # only cover roughly four tool attempts and can discard the original
        # question while the model is still resolving an entity name.
        MAX_MERGED = 20
        last = self._merged[-1]
        if (
            last.role == msg.role
            and last.metadata == msg.metadata
            and msg.image is None
            and last.image is None
        ):
            self._merged[-1] = Message(
                role=last.role,
                content=last.content + "\n" + msg.content,
                metadata=last.metadata,
            )
        else:
            self._merged.append(msg)
        # Trim history to prevent unbounded growth
        while len(self._merged) > MAX_MERGED:
            self._merged.pop(0)
        # tool_result/skill_result are serialized as API ``user`` messages.
        # Treat them as valid turn boundaries; otherwise a long tool loop can
        # trim the original user message and then accidentally delete the whole
        # remaining chain, leaving the request with only a system prompt.
        api_user_roles = {"user", "tool_result", "skill_result"}
        while self._merged and self._merged[0].role not in api_user_roles:
            self._merged.pop(0)

    def touch(self):
        self.last_active = time.time()

    def begin_external_turn(
        self, user_input: str, input_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._active_user and not self._active_turn_finalized:
            self.finish_external_turn("本轮未正常结束。", status="abandoned")
        self._active_user = user_input
        self._active_turn_finalized = False
        self._active_turn_started_at = time.time()
        self._active_input_metadata = dict(input_metadata or {})

    def consume_pending_dialogue(
        self, user_input: str
    ) -> Optional[Dict[str, Any]]:
        """Resume one continuation without discarding it before terminal success."""
        pending = self.pending_dialogue
        if pending is None:
            return None
        now = time.time()
        if pending.status not in {"WAITING_INPUT", "RESUMING"} or (
            pending.expires_at and pending.expires_at <= now
        ):
            self.pending_dialogue = None
            return None
        if pending.status == "WAITING_INPUT" and user_input.strip():
            pending.answers.append(user_input.strip())
            pending.status = "RESUMING"
        return pending.to_prompt_dict()

    def set_pending_dialogue(
        self,
        *,
        root_user_input: str,
        question: str,
        source_execution_id: str = "",
        resume_user_input: str = "",
        completed_tools: Optional[List[str]] = None,
        prior: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 300.0,
    ) -> None:
        prior = prior or {}
        answers = [
            str(item).strip() for item in prior.get("answers", [])
            if str(item).strip()
        ]
        self.pending_dialogue = PendingDialogue(
            continuation_id=str(
                prior.get("continuation_id") or uuid.uuid4().hex[:16]
            ),
            source_task_id=self.prompt_slots.task_id,
            source_execution_id=source_execution_id,
            root_user_input=str(root_user_input or "").strip()[:1500],
            resume_user_input=str(
                resume_user_input or root_user_input or ""
            ).strip()[:1500],
            question=str(question or "").strip()[:500],
            answers=answers,
            completed_tools=sorted(
                {
                    str(item) for item in (
                        list(prior.get("completed_tools", []))
                        + list(completed_tools or [])
                    ) if str(item)
                }
            ),
            resume_tool="medical_consult",
            depth=max(1, int(prior.get("depth", 0) or 0) + 1),
            status="WAITING_INPUT",
            expires_at=time.time() + max(1.0, float(ttl_seconds)),
        )

    def render_conversation_context(self) -> str:
        selected: List[str] = []
        total = 0
        recent = self.conversation_turns[-self.conversation_max_turns:]
        for reverse_index, turn in enumerate(reversed(recent)):
            block = f"第{turn.index}轮\n用户: {turn.user}\n助手: {turn.assistant}"
            extra = len(block) + (1 if selected else 0)
            if total + extra > self.conversation_max_chars:
                # Never discard the newest turn wholesale: for a query
                # continuation it contains both the original answer context
                # and the question the user is now answering.
                if reverse_index == 0 and not selected:
                    header = f"第{turn.index}轮\n用户: "
                    middle = "\n助手: "
                    marker = "…[truncated]"
                    available = max(
                        0,
                        self.conversation_max_chars
                        - len(header) - len(middle) - 2 * len(marker),
                    )
                    user_budget = available // 2
                    assistant_budget = available - user_budget
                    user = (
                        turn.user if len(turn.user) <= user_budget
                        else turn.user[:user_budget] + marker
                    )
                    assistant = (
                        turn.assistant if len(turn.assistant) <= assistant_budget
                        else turn.assistant[:assistant_budget] + marker
                    )
                    selected.append(header + user + middle + assistant)
                    total = len(selected[0])
                continue
            selected.insert(0, block)
            total += extra
        return "\n".join(selected)

    def finish_external_turn(
        self, final_answer: str, status: str = "completed", end_reason: str = ""
    ) -> None:
        if self._active_turn_finalized or not self._active_user:
            return
        self.conversation_turns.append(ConversationTurn(
            index=len(self.conversation_turns) + 1,
            task_id=self.prompt_slots.task_id,
            user=self._active_user,
            assistant=final_answer.strip(),
            status=status,
            started_at=self._active_turn_started_at,
            completed_at=time.time(),
            end_reason=end_reason,
            input_metadata=dict(self._active_input_metadata),
        ))
        self._active_turn_finalized = True
        if end_reason and end_reason != "query":
            self.pending_dialogue = None

    def compact_current_turn(
        self, final_answer: str = "", status: str = "completed", end_reason: str = ""
    ):
        """Keep only the external user request and its final answer.

        Tool calls and tool results are useful while one request is running,
        but carrying all of them into the next spoken request distracts small
        models and consumes most of the context window.  ``history`` remains
        untouched for diagnostics; only the API-facing bounded view is compacted.
        """
        user_index = None
        for index in range(len(self._merged) - 1, -1, -1):
            if self._merged[index].role == "user":
                user_index = index
                break
        if user_index is None:
            return

        compacted = self._merged[:user_index + 1]
        answer = final_answer.strip()
        if answer:
            compacted.append(Message(role="assistant", content=answer))
        self._merged = compacted
        self.finish_external_turn(final_answer, status=status, end_reason=end_reason)

    def clear(self):
        self.history.clear()
        self._merged.clear()
        self.need_clear = True
        self._summary_cache_valid = False
        self._summary_cache = None
        self.prompt_slots = PromptSlots()
        self.last_usage.clear()
        self.last_prompt_preflight.clear()
        self.conversation_turns.clear()
        self.model_call_records.clear()
        self.benchmark_context.clear()
        self.execution_state = None
        self.pending_dialogue = None
        self._active_user = ""
        self._active_turn_finalized = True
        self._active_turn_started_at = 0.0
        self._active_input_metadata.clear()

    def new_session(self) -> "Session":
        new = Session(need_clear=True)
        summary = self.summarize_for_context()
        if summary:
            new.add_message(
                "system",
                f"[上下文延续]\n上一会话摘要:\n{summary}",
            )
        return new

    def summarize_for_context(self, max_chars: int = 600) -> str:
        if self._summary_cache_valid and self._summary_cache is not None:
            return self._summary_cache

        if not self.history:
            return ""

        lines = []
        total = 0
        # 从最新消息开始回溯，最多检查 20 条
        for m in reversed(self.history[-20:]):
            if m.role == "user":
                text = f"用户: {m.content[:100]}"
            elif m.role == "assistant":
                text = f"助手: {m.content[:100]}"
            elif m.role in ("tool_result", "skill_result"):
                label = "工具" if m.role == "tool_result" else "技能"
                source = m.metadata.get("source", "")
                text = f"[{label}: {source}] {m.content[:80]}"
            else:
                continue
            if total + len(text) > max_chars:
                break
            lines.insert(0, text)
            total += len(text)

        result = "\n".join(lines)
        self._summary_cache = result
        self._summary_cache_valid = True
        return result

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        return self.history[-n:]

    def estimate_tokens(self) -> int:
        total = 0
        for msg in self.history:
            content = msg.content
            # 中文约 2 token/字，ASCII 约 0.25 token/字符
            chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
            ascii_chars = len(content) - chinese_chars
            total += chinese_chars * 2 + ascii_chars // 4
            if msg.image:
                total += 576
        return total
