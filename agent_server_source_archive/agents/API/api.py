from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import httpx

from .session import Message, Session
from .trajectory import TrajectoryWriter

logger = logging.getLogger(__name__)


def _visible_message_content(message: dict) -> str:
    """Preserve native OpenAI/Qwen function calls behind the XML adapter."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        function_call = message.get("function_call")
        calls = ([{"function": function_call}] if isinstance(function_call, dict) else [])
    rendered = []
    for item in calls:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "") or "").strip()
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not name or not isinstance(arguments, dict):
            continue
        rendered.append(
            "<tool>"
            + json.dumps(
                {"tool_call": name, "param": arguments},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            + "</tool>"
        )
    return "\n".join(rendered)


class API:
    BASE_PATH = "/v1/chat/completions"

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        default_model: str = "qwen3.5-4b-history",
        benchmark: bool = False,
        max_tokens: int = 128,
        emit_events: bool = True,
        trajectory_enabled: bool = True,
        trajectory_dir: str = "/data/structure/trajectories",
        online_max_concurrency: int = 3,
        context_window_tokens: int = 8192,
        online_cache_user_ids: Optional[tuple[str, ...]] = None,
        tokenizer_path: str = "",
        prompt_tokenizer=None,
    ):
        self.server_url = server_url.rstrip("/")
        self.default_model = default_model
        self.benchmark = benchmark
        self.max_tokens = max_tokens
        self.emit_events = emit_events
        self.trajectory_writer = TrajectoryWriter(
            trajectory_dir, enabled=trajectory_enabled
        )
        self.record_callback = None
        self._client: Optional[httpx.AsyncClient] = None
        # qwen3_5_server exposes one physical pipeline/KV cache.  Serialize
        # requests even across different Agent modes/clients so a clear-and-
        # generate sequence cannot interleave with another request.
        self._request_lock = asyncio.Lock()
        self._online_semaphore = asyncio.Semaphore(max(1, int(online_max_concurrency)))
        self.context_window_tokens = max(1, int(context_window_tokens))
        self.tokenizer_path = str(tokenizer_path or "").strip()
        self._prompt_tokenizer = prompt_tokenizer
        self._tokenizer_load_attempted = prompt_tokenizer is not None
        configured_lanes = online_cache_user_ids or tuple(
            value.strip() for value in os.environ.get(
                "DEEPSEEK_CACHE_USER_IDS",
                "teacher-prompt-lane-0,teacher-prompt-lane-1,"
                "teacher-prompt-lane-2,teacher-prompt-lane-3",
            ).split(",") if value.strip()
        )
        self.online_cache_user_ids = tuple(configured_lanes) or (
            "teacher-prompt-lane-0",
        )
        self._trace_warning_logged = False

    def _get_prompt_tokenizer(self):
        if self._prompt_tokenizer is not None:
            return self._prompt_tokenizer
        if self._tokenizer_load_attempted or not self.tokenizer_path:
            return None
        self._tokenizer_load_attempted = True
        try:
            from transformers import AutoTokenizer

            path = Path(self.tokenizer_path).expanduser()
            self._prompt_tokenizer = AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=True
            )
        except Exception as exc:  # pragma: no cover - deployment diagnostics
            logger.warning("Prompt tokenizer preflight disabled: %s", exc)
        return self._prompt_tokenizer

    def _count_prompt_tokens(self, prompt_slots: dict) -> int:
        tokenizer = self._get_prompt_tokenizer()
        if tokenizer is None:
            raise RuntimeError("prompt tokenizer is not configured")
        from server.prompt_protocol import messages_from_prompt_slots

        text_slots = dict(prompt_slots)
        text_slots.pop("image", None)
        messages = messages_from_prompt_slots(text_slots)
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    def _preflight_prompt(self, session: Session) -> dict:
        started = time.perf_counter()
        tokenizer = self._get_prompt_tokenizer()
        if tokenizer is None:
            stats = {
                "enabled": False,
                "reason": "tokenizer_unavailable",
                "context_window_tokens": self.context_window_tokens,
                "tokenize_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            session.last_prompt_preflight = stats
            return session.prompt_slots.to_request_dict()
        # Match the history server's exact input boundary (len(tokens) must be
        # below seqlen - 1).  ``max_tokens`` is a generation cap, not a prompt
        # reservation; teacher runs may legitimately set it to the full
        # window, so subtracting it here would reject every request.
        token_budget = max(1, self.context_window_tokens - 2)
        stats = session.prompt_slots.compact_for_token_budget(
            self._count_prompt_tokens, token_budget
        )
        stats.update({
            "enabled": True,
            "method": "exact",
            "context_window_tokens": self.context_window_tokens,
            "reserved_completion_tokens": self.max_tokens,
            "safe_input_limit_tokens": token_budget,
            "margin_tokens": max(0, token_budget - stats["final_prompt_tokens"]),
            "tokenize_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        actions = list(stats.get("actions", []))
        changed_slots = []
        if any(item.startswith("conversation_") for item in actions):
            changed_slots.append("C")
        if any(
            item.startswith("history_") or item.startswith("execution_")
            for item in actions
        ):
            changed_slots.append("H")
        if any(item.startswith("attempt:") for item in actions):
            changed_slots.append("A")
        stats["changed_slots"] = changed_slots
        stats["status"] = (
            "blocked" if not stats.get("fits", True)
            else "compacted" if stats.get("compacted") else "pass"
        )
        stats["cache_preservation_level"] = (
            "S" if "C" in changed_slots else
            "U" if "H" in changed_slots else
            "H" if "A" in changed_slots else "A"
        )
        stats["overflow_prevented"] = bool(
            stats["initial_prompt_tokens"] > token_budget
            and stats.get("fits", False)
        )
        session.last_prompt_preflight = stats
        if not stats.get("fits", True):
            record = {
                **session.benchmark_context,
                "session_id": session.id,
                "task_id": session.prompt_slots.task_id,
                "external_turn": session.prompt_slots.external_turn,
                "agent_iteration": 1 + sum(
                    1 for row in session.model_call_records
                    if row.get("task_id") == session.prompt_slots.task_id
                ),
                "status": "preflight_blocked",
                "error_type": "PROMPT_CONTEXT_LIMIT",
                "error": (
                    f"prompt={stats['final_prompt_tokens']} budget={token_budget}"
                ),
                "usage": {},
                "context_stats": {
                    **self._context_stats({}),
                    "prompt_tokens": stats["final_prompt_tokens"],
                    "prompt_context_overflow": True,
                    "context_overflow": True,
                    "overflow_tokens": max(
                        0, stats["final_prompt_tokens"] - token_budget
                    ),
                    "preflight_blocked": True,
                },
                "prompt_preflight": dict(stats),
                "benchmark": {},
                "output": "",
                "model": self.default_model,
                "provider": (
                    "deepseek" if self.default_model.startswith("deepseek-") else "local"
                ),
                "prompt_slots": session.prompt_slots.to_request_dict(),
                "has_image": session.prompt_slots.user_image is not None,
                "http_request_sent": False,
            }
            session.model_call_records.append(record)
            if self.record_callback is not None:
                self.record_callback(record)
            raise RuntimeError(
                "PROMPT_CONTEXT_OVERFLOW_PRE_SEND: "
                f"prompt={stats['final_prompt_tokens']} budget={token_budget} "
                f"actions={stats['actions']}"
            )
        return session.prompt_slots.to_request_dict()

    def _provider_cache_user_id(self, session: Session, model: str) -> str | None:
        """Choose a stable DeepSeek ``user_id`` without sharing Agent history.

        DeepSeek uses this field for KV-cache/scheduling isolation.  It is a
        provider identity lane, not the local ``x-session-id``; the latter
        remains unique per task so prompt-slot histories cannot interleave.
        """
        if not model.startswith("deepseek-"):
            return None
        digest = hashlib.sha256(str(session.id).encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(self.online_cache_user_ids)
        return self.online_cache_user_ids[index]

    @staticmethod
    def _estimate_visible_tokens(text: str) -> int:
        """Conservative tokenizer-free estimate for stored visible output.

        Provider completion usage may include hidden reasoning, which is
        intentionally removed from SFT. Keep it as cost telemetry, but assess
        the 8K training sequence against prompt + visible PLAN/ACT output.
        """
        value = str(text or "")
        cjk = len(re.findall(r"[\u3400-\u9fff]", value))
        ascii_runs = re.findall(r"[A-Za-z0-9_]+", value)
        ascii_tokens = sum((len(run) + 2) // 3 for run in ascii_runs)
        punctuation = len(re.findall(r"[^\sA-Za-z0-9_\u3400-\u9fff]", value))
        return cjk + ascii_tokens + punctuation

    def _context_stats(self, usage: dict, visible_output: str = "") -> dict:
        prompt = max(0, int(usage.get("prompt_tokens", 0) or 0))
        completion = max(0, int(usage.get("completion_tokens", 0) or 0))
        total = max(prompt + completion, int(usage.get("total_tokens", 0) or 0))
        limit = self.context_window_tokens
        visible = self._estimate_visible_tokens(visible_output)
        training_sequence = prompt + visible
        return {
            "context_window_tokens": limit,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "remaining_after_prompt_tokens": max(0, limit - prompt),
            "provider_total_tokens": total,
            "provider_total_overflow": total > limit,
            "visible_completion_tokens_estimate": visible,
            "training_sequence_tokens_estimate": training_sequence,
            "remaining_after_completion_tokens": max(0, limit - training_sequence),
            "context_utilization": round(training_sequence / limit, 6),
            "near_context_limit": training_sequence >= int(limit * 0.9),
            "prompt_context_overflow": prompt > limit,
            "context_overflow": training_sequence > limit,
            "overflow_tokens": max(0, training_sequence - limit),
        }

    def _request_gate(self, model: str):
        """Select the transport gate without weakening local KV-cache isolation.

        Online DeepSeek requests do not share the local TPU history backend and
        may therefore overlap.  Local model requests remain serialized because
        their physical prefix-cache state is process-global.
        """
        return (
            self._online_semaphore
            if model.startswith("deepseek-") else self._request_lock
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=600.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def release_history_session_cache(self, session_id: str) -> None:
        """Best-effort release of local Qwen History KV snapshots after a turn.

        This intentionally talks to the history backend directly: the Router
        need not be restarted solely to expose an internal lifecycle endpoint.
        """
        session_id = str(session_id or "").strip()
        if not session_id or self.default_model.startswith("deepseek-"):
            return
        base_url = os.environ.get(
            "QWEN_HISTORY_CACHE_URL", "http://127.0.0.1:8007"
        ).rstrip("/")
        try:
            client = await self._get_client()
            response = await client.delete(
                f"{base_url}/v1/cache/sessions/{session_id}", timeout=5.0
            )
            response.raise_for_status()
            logger.info("Released local history KV cache for session=%s", session_id)
        except Exception as exc:
            # Session disposal must never make the browser/voice flow fail.
            logger.warning("Unable to release local history KV cache for session=%s: %s", session_id, exc)

    async def emit_agent_event(
        self,
        trace_id: str,
        event_type: str,
        *,
        session_id: str = "",
        mode: str = "",
        payload: Optional[dict] = None,
    ) -> None:
        """Best-effort delivery of one observable Agent event to the router.

        Observability must never make a user request fail, so transport errors
        are logged once and otherwise ignored.
        """
        event = {
            "trace_id": trace_id,
            "type": event_type,
            "timestamp": time.time(),
            "session_id": session_id,
            "mode": mode,
            "payload": payload or {},
        }
        if not self.emit_events:
            return
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.server_url}/v1/agent/events",
                json=event,
                timeout=2.0,
            )
            response.raise_for_status()
            self._trace_warning_logged = False
        except Exception as exc:
            if not self._trace_warning_logged:
                logger.warning(f"Agent trace delivery unavailable: {exc}")
                self._trace_warning_logged = True

    async def chat_complete(
        self,
        session: Session,
        messages: Optional[List[Message]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Run one non-streaming completion and retain its token usage."""
        del messages
        client = await self._get_client()
        model = model or self.default_model
        # The history backend owns isolated physical snapshots per session.
        # A clear is sent only for an explicit/new local session reset.
        headers = {
            "Content-Type": "application/json",
            "x-clear-history": "true" if session.need_clear else "false",
            "x-session-id": session.id,
            "x-prompt-slots": "suha.v3",
        }

        prompt_slots = self._preflight_prompt(session)
        trajectory_slots = dict(prompt_slots)
        had_image = trajectory_slots.pop("image", None) is not None
        payload = {
            "model": model,
            "prompt_slots": prompt_slots,
            "stream": False,
        }
        provider_cache_user_id = self._provider_cache_user_id(session, model)
        if provider_cache_user_id is not None:
            payload["user_id"] = provider_cache_user_id
        if self.benchmark:
            payload["benchmark"] = True
            payload["max_tokens"] = self.max_tokens

        url = f"{self.server_url}{self.BASE_PATH}"

        # Log full request
        logger.info(
            "LLM_REQ: model=%s slots=system,conversation,user,history,attempt payload_len=%d",
            model,
            len(json.dumps(payload)),
        )
        for slot_name in ("system", "conversation", "user", "history", "attempt"):
            slot_value = str(prompt_slots.get(slot_name, ""))
            logger.info(
                "LLM_REQ_SLOT[%s]: len=%d content=%s",
                slot_name,
                len(slot_value),
                slot_value[:300],
            )

        request_gate = self._request_gate(model)
        async with request_gate:
            request_started_at = time.time()
            wall_started = time.perf_counter()
            cpu_started = time.thread_time()
            process_cpu_started = time.process_time()
            try:
                response = await client.post(url, json=payload, headers=headers)
                client_ms = (time.perf_counter() - wall_started) * 1000
                client_cpu_ms = (time.thread_time() - cpu_started) * 1000
                client_process_cpu_ms = (
                    time.process_time() - process_cpu_started
                ) * 1000
                if response.status_code != 200:
                    message = response.text[:500]
                    raise RuntimeError(
                        f"LLM API HTTP {response.status_code}: {message}"
                    )
                data = response.json()
                if "error" in data:
                    raise RuntimeError(f"LLM error: {data['error']}")

                usage = data.get("usage")
                if isinstance(usage, dict):
                    session.last_usage = {
                        key: int(usage.get(key, 0) or 0)
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "prompt_cache_hit_tokens",
                            "prompt_cache_miss_tokens",
                        )
                    }
                    logger.info("LLM_USAGE: %s", session.last_usage)

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("LLM response contains no choices")
                message = choices[0].get("message", {})
                content = _visible_message_content(message)
                diagnostic = data.get("benchmark", {}) if self.benchmark else {}
                record = {
                    **session.benchmark_context,
                    "session_id": session.id,
                    "task_id": session.prompt_slots.task_id,
                    "external_turn": session.prompt_slots.external_turn,
                    "agent_iteration": 1 + sum(
                        1 for row in session.model_call_records
                        if row.get("task_id") == session.prompt_slots.task_id
                    ),
                    "status": "ok",
                    "request_started_at": request_started_at,
                    "response_completed_at": time.time(),
                    "client_e2e_ms": client_ms,
                    "client_cpu_ms": client_cpu_ms,
                    "client_process_cpu_ms_raw": client_process_cpu_ms,
                    "usage": dict(session.last_usage),
                    "context_stats": self._context_stats(session.last_usage, content or ""),
                    "prompt_preflight": dict(session.last_prompt_preflight),
                    "benchmark": diagnostic,
                    "output": content or "",
                    "model": model,
                    "provider": "deepseek" if model.startswith("deepseek-") else "local",
                    "provider_cache_user_id": provider_cache_user_id,
                    "prompt_slots": trajectory_slots,
                    "has_image": had_image,
                    "history_entries": len(session.prompt_slots.history.entries),
                    "visible_history_entries": len(session.prompt_slots.history.visible_entries()),
                    "attempt_entries": len(session.prompt_slots.attempt),
                    "conversation_context_chars": len(session.render_conversation_context()),
                    "visible_history_task_ids": [
                        item.get("task_id", "")
                        for item in session.prompt_slots.history.visible_entries()
                    ],
                    "attempt_categories": [
                        item.get("category", "") for item in session.prompt_slots.attempt
                    ],
                }
                session.model_call_records.append(record)
                if self.record_callback is not None:
                    self.record_callback(record)
                session.need_clear = False
                return content or ""
            except httpx.ConnectError as exc:
                logger.error("LLM server unreachable: %s", exc)
                self._record_failed_call(
                    session, exc, wall_started, cpu_started, process_cpu_started
                )
                raise RuntimeError(f"LLM server unreachable: {exc}") from exc
            except Exception as exc:
                logger.error("LLM request error: %s", exc)
                self._record_failed_call(
                    session, exc, wall_started, cpu_started, process_cpu_started
                )
                raise

    def _record_failed_call(self, session: Session, exc: Exception,
                            wall_started: float, cpu_started: float,
                            process_cpu_started: float) -> None:
        failed_slots = session.prompt_slots.to_request_dict()
        failed_had_image = failed_slots.pop("image", None) is not None
        record = {
            **session.benchmark_context,
            "session_id": session.id,
            "task_id": session.prompt_slots.task_id,
            "external_turn": session.prompt_slots.external_turn,
            "agent_iteration": 1 + sum(
                1 for row in session.model_call_records
                if row.get("task_id") == session.prompt_slots.task_id
            ),
            "status": "error",
            "request_started_at": time.time() - (time.perf_counter() - wall_started),
            "response_completed_at": time.time(),
            "client_e2e_ms": (time.perf_counter() - wall_started) * 1000,
            "client_cpu_ms": (time.thread_time() - cpu_started) * 1000,
            "client_process_cpu_ms_raw": (
                time.process_time() - process_cpu_started
            ) * 1000,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "usage": {},
            "context_stats": {
                **self._context_stats({}),
                "overflow_detected_from_error": bool(re.search(
                    r"context.{0,20}(?:length|window|overflow)|sequence.{0,20}(?:length|overflow)|"
                    r"maximum context|too many tokens",
                    str(exc), re.I,
                )),
            },
            "prompt_preflight": dict(session.last_prompt_preflight),
            "benchmark": {},
            "output": "",
            "model": self.default_model,
            "provider": "deepseek" if self.default_model.startswith("deepseek-") else "local",
            "prompt_slots": failed_slots,
            "has_image": failed_had_image,
        }
        session.model_call_records.append(record)
        if self.record_callback is not None:
            self.record_callback(record)

    def write_trajectory(
        self, *, session: Session, trace_id: str, mode: str, status: str,
        final_text: str, metrics: dict, events: list,
    ):
        return self.trajectory_writer.write_turn(
            session=session,
            trace_id=trace_id,
            mode=mode,
            status=status,
            final_text=final_text,
            metrics=metrics,
            events=events,
            model_config={
                "server_url": self.server_url,
                "model": self.default_model,
                "provider": (
                    "deepseek" if self.default_model.startswith("deepseek-") else "local"
                ),
                "benchmark": self.benchmark,
                "prompt_protocol": "suha.v3",
                "context_window_tokens": self.context_window_tokens,
            },
        )
