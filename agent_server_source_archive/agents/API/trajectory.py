"""Atomic, redacted per-turn Agent trajectory export."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:128]
    return cleaned or "unknown"


def _redact(value: Any) -> Any:
    """Defense-in-depth removal of secrets, images and hidden reasoning."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lower = str(key).lower()
            if lower in {"authorization", "api_key", "reasoning_content", "image", "image_url"}:
                continue
            if "secret" in lower or "token_value" in lower:
                continue
            result[str(key)] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image omitted]"
    return value


def summarize_token_usage(calls: Iterable[dict], context_limit: int = 8192) -> dict:
    rows = list(calls)
    usages = [row.get("usage") or {} for row in rows]
    context_rows = [row.get("context_stats") or {} for row in rows]
    return {
        "context_window_tokens": int(context_limit),
        "model_call_count": len(rows),
        "calls_with_usage": sum(bool(usage) for usage in usages),
        "prompt_tokens_sum": sum(int(u.get("prompt_tokens", 0) or 0) for u in usages),
        "completion_tokens_sum": sum(
            int(u.get("completion_tokens", 0) or 0) for u in usages
        ),
        "total_tokens_sum": sum(int(u.get("total_tokens", 0) or 0) for u in usages),
        "peak_prompt_tokens": max(
            (int(u.get("prompt_tokens", 0) or 0) for u in usages), default=0
        ),
        "peak_total_tokens": max(
            (int(u.get("total_tokens", 0) or 0) for u in usages), default=0
        ),
        "near_context_limit_calls": sum(
            bool(row.get("near_context_limit")) for row in context_rows
        ),
        "context_overflow_calls": sum(
            bool(row.get("context_overflow") or row.get("overflow_detected_from_error"))
            for row in context_rows
        ),
        "provider_total_overflow_calls": sum(
            bool(row.get("provider_total_overflow")) for row in context_rows
        ),
        "context_overflow": any(
            bool(row.get("context_overflow") or row.get("overflow_detected_from_error"))
            for row in context_rows
        ),
    }


class TrajectoryWriter:
    def __init__(self, directory: str, enabled: bool = True):
        self.directory = Path(directory).expanduser().resolve()
        self.enabled = bool(enabled)

    def write_turn(
        self, *, session, trace_id: str, mode: str, status: str,
        final_text: str, metrics: Dict[str, Any], events: Iterable[dict],
        model_config: Dict[str, Any],
    ) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        session_dir = self.directory / _safe_component(session.id)
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(session_dir, 0o700)
        except OSError:
            pass
        turn = int(session.prompt_slots.external_turn)
        task_id = _safe_component(session.prompt_slots.task_id)
        target = session_dir / f"{turn:06d}_{task_id}.json"
        calls = [
            row for row in session.model_call_records
            if row.get("task_id") == session.prompt_slots.task_id
        ]
        context_limit = int(model_config.get("context_window_tokens", 8192) or 8192)
        token_usage = summarize_token_usage(calls, context_limit)
        event_rows = list(events)
        latest_turn = (
            session.conversation_turns[-1].to_dict()
            if session.conversation_turns else None
        )
        errors = []
        for event in event_rows:
            event_type = str(event.get("type", ""))
            detail = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            if event_type == "policy":
                errors.append({
                    "code": str(detail.get("error_code") or "STATE_VALIDATION_ERROR"),
                    "category": "state",
                    "message": str(detail.get("message", "")),
                    "timestamp": event.get("timestamp"),
                })
            elif event_type == "error":
                errors.append({
                    "code": "MODEL_OR_AGENT_ERROR",
                    "category": "runtime",
                    "message": str(detail.get("message", "")),
                    "timestamp": event.get("timestamp"),
                })
            elif event_type == "state_error":
                errors.append({
                    "code": str(detail.get("error_code") or "STATE_VALIDATION_ERROR"),
                    "category": "state",
                    "message": str(detail.get("message", "")),
                    "timestamp": event.get("timestamp"),
                })
            elif event_type == "tool_result" and detail.get("success") is False:
                errors.append({
                    "code": str(detail.get("error_type") or "TOOL_EXECUTION_ERROR"),
                    "category": "tool",
                    "message": str(detail.get("error", "")),
                    "timestamp": event.get("timestamp"),
                })
        if latest_turn and latest_turn.get("end_reason") == "not_terminated":
            errors.append({
                "code": "TURN_NOT_TERMINATED", "category": "state",
                "message": "本轮未以 query 或 speak 结束。",
                "timestamp": latest_turn.get("completed_at"),
            })
        if token_usage["context_overflow"]:
            errors.append({
                "code": "CONTEXT_OVERFLOW", "category": "context",
                "message": (
                    f"模型调用超过目标上下文窗口 {context_limit} tokens；"
                    f"峰值 total={token_usage['peak_total_tokens']}"
                ),
                "timestamp": time.time(),
            })
        payload = _redact({
            "schema_version": "trajectory.v3",
            "saved_at": time.time(),
            "session_id": session.id,
            "task_id": session.prompt_slots.task_id,
            "external_turn": turn,
            "trace_id": trace_id,
            "mode": mode,
            "status": status,
            "model_config": model_config,
            "has_image": bool(session.prompt_slots.user_image),
            "plan": session.prompt_slots.plan,
            "execution_state": (
                session.execution_state.to_dict()
                if session.execution_state is not None else None
            ),
            "execution_state_events": session.prompt_slots.execution_events,
            "model_calls": calls,
            "token_usage": token_usage,
            "conversation_turn": latest_turn,
            "errors": errors,
            "events": event_rows,
            "final": {"text": final_text, "metrics": metrics},
        })
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=session_dir
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)

            turns = [item.to_dict() for item in session.conversation_turns]
            ended = bool(turns and turns[-1].get("end_reason") == "speak")
            prior_errors = []
            prior_session_path = session_dir / "session.json"
            if prior_session_path.is_file():
                try:
                    prior_payload = json.loads(prior_session_path.read_text("utf-8"))
                    if isinstance(prior_payload.get("errors"), list):
                        prior_errors = prior_payload["errors"]
                except (OSError, ValueError, TypeError):
                    prior_errors = []
            session_payload = _redact({
                "schema_version": "trajectory_session.v1",
                "session_id": session.id,
                "created_at": session.created_at,
                "updated_at": time.time(),
                "completed_at": turns[-1].get("completed_at") if ended else None,
                "status": "completed" if ended else (
                    "error" if status == "error" else "active"
                ),
                "end_reason": "speak" if ended else (
                    "SESSION_NOT_TERMINATED" if status == "error" else ""
                ),
                "max_turns": session.conversation_max_turns,
                "turn_count": len(turns),
                "turns": turns,
                "trajectory_files": sorted(
                    item.name for item in session_dir.glob("[0-9]*_*.json")
                ),
                "model_calls": session.model_call_records,
                "token_usage": summarize_token_usage(
                    session.model_call_records, context_limit
                ),
                "errors": prior_errors + errors,
            })
            session_target = session_dir / "session.json"
            session_fd, session_tmp = tempfile.mkstemp(
                prefix=".session.json.", suffix=".tmp", dir=session_dir
            )
            try:
                os.fchmod(session_fd, 0o600)
                with os.fdopen(session_fd, "w", encoding="utf-8") as session_handle:
                    json.dump(
                        session_payload, session_handle, ensure_ascii=False,
                        indent=2, sort_keys=True,
                    )
                    session_handle.write("\n")
                    session_handle.flush()
                    os.fsync(session_handle.fileno())
                os.replace(session_tmp, session_target)
            except Exception:
                try:
                    os.close(session_fd)
                except OSError:
                    pass
                try:
                    os.unlink(session_tmp)
                except OSError:
                    pass
                raise
            return target
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
