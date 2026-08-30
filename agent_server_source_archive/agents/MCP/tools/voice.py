"""Voice-related MCP tools."""
from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any, Dict

from ..base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SpeakTool(Tool):
    """Send TTS text to the browser for speech synthesis."""

    name = "speak"
    description = "通过语音播报文字内容给用户"
    param_schema = {
        "text": "要播报的文字内容",
    }
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "WRITE", "idempotent": False,
        "terminal": True,
        "turn_terminal": True,
        "session_terminal": True,
        "produces": ["speech.last_text"], "invalidates": [],
        "retry": {"max_attempts": 1},
    }

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        text = params.get("text", "")
        if not text:
            return ToolResult(success=False, error="text参数为空",
                              data="(无播报内容)")

        remote_audio = context.extra.get("remote_audio") if context.extra else None
        if remote_audio:
            try:
                operation_id = await remote_audio.speak(text)
                logger.info("speak completed: operation_id=%s", operation_id)
                return ToolResult(
                    success=True,
                    data=text,
                    facts={
                        "speech.last_text": text,
                        "speech.operation_id": operation_id,
                    },
                )
            except Exception as exc:
                logger.warning("Remote speak failed: %s", exc)
                return ToolResult(
                    success=False,
                    error=str(exc),
                    error_type=getattr(exc, "code", "REMOTE_AUDIO_ERROR"),
                    retryable=bool(getattr(exc, "retryable", False)),
                    recovery_hint="检查远端设备连接、播放能力与 operation_id 回执",
                )

        # Legacy in-process VoiceMode path (non-headless use only).
        voice_mode = context.extra.get("voice_mode") if context.extra else None
        if voice_mode:
            try:
                await voice_mode.broadcast_response(text)
            except Exception as e:
                logger.warning(f"Voice broadcast failed: {e}")

        logger.info(f"speak: {text[:80]}...")
        return ToolResult(success=True, data=text, facts={"speech.last_text": text})


class QueryTool(Tool):
    """Ask one follow-up question, then capture the user's next utterance."""

    name = "query"
    description = (
        "仅当已有业务工具明确返回需要用户补充信息时，播报一个澄清问题并在播放完成后录音。"
        "question 只写问题本身，不要加入录音提示语。"
    )
    param_schema = {"question": "需要用户补充回答的问题"}
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "WRITE",
        "idempotent": False,
        "terminal": True,
        "turn_terminal": True,
        "session_terminal": False,
        "produces": ["dialogue.question"],
        "invalidates": ["dialogue.followup_required"],
        "retry": {"max_attempts": 1},
    }

    @staticmethod
    def playback_text(question: str) -> str:
        value = question.strip().rstrip("。！？!?；;，,")
        return value + "。在滴的一声后开始回答"

    async def call(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        question = str(params.get("question", "") or "").strip()
        if not question:
            return ToolResult(success=False, error="question参数为空", data="(无追问内容)")

        rendered = self.playback_text(question)
        remote_audio = context.extra.get("remote_audio") if context.extra else None
        if remote_audio is not None:
            try:
                recording = await remote_audio.speak_and_record(rendered)
                pcm = bytes(recording.pcm)
                audio = {
                    "operation_id": recording.operation_id,
                    "duration_seconds": recording.duration_seconds,
                    "bytes": len(pcm),
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "stop_reason": recording.stop_reason,
                    "sample_rate": recording.audio_format.sample_rate,
                    "channels": recording.audio_format.channels,
                }
                return ToolResult(
                    success=True,
                    data=question,
                    facts={
                        "dialogue.question": question,
                        "audio.playback_text": rendered,
                        "audio.operation_id": recording.operation_id,
                        "audio.duration_seconds": recording.duration_seconds,
                        "audio.bytes": len(pcm),
                        "audio.sha256": audio["sha256"],
                        "audio.stop_reason": recording.stop_reason,
                    },
                    transient={"recording_pcm": pcm, "audio": audio},
                )
            except Exception as exc:
                logger.warning("Remote query failed: %s", exc)
                return ToolResult(
                    success=False,
                    error=str(exc),
                    error_type=getattr(exc, "code", "REMOTE_AUDIO_ERROR"),
                    retryable=bool(getattr(exc, "retryable", False)),
                    recovery_hint="检查远端播放、录音和 operation_id 回执",
                )

        provider = context.extra.get("query_followup_provider") if context.extra else None
        if provider is not None:
            value = provider(question)
            if inspect.isawaitable(value):
                value = await value
            followup = str(value or "").strip()
            if not followup:
                return ToolResult(success=False, error="mock追问没有后续用户输入")
            return ToolResult(
                success=True,
                data=question,
                facts={"dialogue.question": question, "audio.playback_text": rendered},
                transient={"followup_text": followup, "audio": {"source": "benchmark_mock"}},
            )

        return ToolResult(
            success=False,
            error="query 需要远程音频控制器或 Benchmark follow-up provider",
            error_type="QUERY_INPUT_UNAVAILABLE",
        )
