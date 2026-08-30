"""Authoritative controller for remote speak/record operations."""
from __future__ import annotations

import asyncio
import base64
import io
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .protocol import (
    ProtocolError,
    decode_audio,
    message,
    new_operation_id,
    require_operation_id,
    validate_version,
)
from .tts_stream import split_text
from .vad import WebRtcVadStopDetector

SendCallable = Callable[[dict], Awaitable[None]]
SynthesizeCallable = Callable[[str], Awaitable[str | bytes]]


def _wav_metadata(wav_bytes: bytes) -> tuple[int, int]:
    """Read channel/rate from PCM *or* IEEE-float WAV.

    Python's :mod:`wave` in the deployed Python version rejects format 3
    (IEEE float), although browsers and VITS both handle it as a valid WAV.
    Playback transports opaque WAV bytes, so validating the RIFF ``fmt``
    chunk is sufficient here; the receiving device performs actual decoding.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            return source.getframerate(), source.getnchannels()
    except wave.Error:
        pass
    if len(wav_bytes) < 20 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        start, end = offset + 8, offset + 8 + chunk_size
        if end > len(wav_bytes):
            break
        if chunk_id == b"fmt " and chunk_size >= 16:
            audio_format, channels, sample_rate = struct.unpack_from("<HHI", wav_bytes, start)
            if audio_format in {1, 3, 0xFFFE} and channels > 0 and sample_rate > 0:
                return sample_rate, channels
            break
        offset = end + (chunk_size & 1)
    raise ValueError("missing or unsupported WAV fmt chunk")


class RemoteAudioError(RuntimeError):
    def __init__(self, code: str, detail: str, operation_id: str = "", retryable: bool = False):
        super().__init__(detail)
        self.code = code
        self.operation_id = operation_id
        self.retryable = retryable


@dataclass(frozen=True)
class AudioFormat:
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1

    def as_dict(self) -> dict:
        return {
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class RecordingResult:
    operation_id: str
    pcm: bytes
    audio_format: AudioFormat
    stop_reason: str
    started_at: float
    completed_at: float

    @property
    def duration_seconds(self) -> float:
        denominator = self.audio_format.sample_rate * self.audio_format.channels * 2
        return len(self.pcm) / denominator if denominator else 0.0

    def to_wav(self) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(self.audio_format.channels)
            target.setsampwidth(2)
            target.setframerate(self.audio_format.sample_rate)
            target.writeframes(self.pcm)
        return output.getvalue()


@dataclass
class _ActiveOperation:
    operation_id: str
    name: str
    accepted: asyncio.Future
    playback_completed: asyncio.Future
    record_started: asyncio.Future
    record_completed: asyncio.Future
    failed: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)
    record_started_at: float = 0.0
    expected_record_seq: int = 0
    record_pcm: bytearray = field(default_factory=bytearray)
    record_last_seen: bool = False
    stop_sent: bool = False
    stop_reason: str = ""
    vad: Optional[WebRtcVadStopDetector] = None
    playback_last_sent: bool = False


class RemoteAudioOperations:
    """One-device operation controller.

    The WebSocket receive loop must forward operation events to
    :meth:`handle_message` while an operation coroutine is awaiting them.
    """

    def __init__(
        self,
        send: SendCallable,
        synthesize: SynthesizeCallable,
        *,
        accept_timeout: float = 3.0,
        playback_timeout: float = 120.0,
        record_start_timeout: float = 3.0,
        record_max_seconds: float = 10.0,
        record_drain_timeout: float = 2.0,
    ):
        self._send = send
        self._synthesize = synthesize
        self.accept_timeout = accept_timeout
        self.playback_timeout = playback_timeout
        self.record_start_timeout = record_start_timeout
        self.record_max_seconds = record_max_seconds
        self.record_drain_timeout = record_drain_timeout
        self._lock = asyncio.Lock()
        self._active: Optional[_ActiveOperation] = None

    @property
    def active_operation_id(self) -> str:
        return self._active.operation_id if self._active else ""

    @property
    def active_operation(self) -> str:
        return self._active.name if self._active else ""

    def _new_active(self, name: str) -> _ActiveOperation:
        loop = asyncio.get_running_loop()
        return _ActiveOperation(
            operation_id=new_operation_id(),
            name=name,
            accepted=loop.create_future(),
            playback_completed=loop.create_future(),
            record_started=loop.create_future(),
            record_completed=loop.create_future(),
            failed=loop.create_future(),
        )

    async def _wait(self, event: asyncio.Future, timeout: float, label: str) -> Any:
        active = self._active
        if active is None:
            raise RemoteAudioError("NO_ACTIVE_OPERATION", "当前没有活动操作")
        done, _ = await asyncio.wait(
            {event, active.failed}, timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if active.failed in done:
            raise active.failed.result()
        if event not in done:
            raise RemoteAudioError(
                "OPERATION_TIMEOUT", f"等待 {label} 超时", active.operation_id, True
            )
        return event.result()

    async def _start(self, active: _ActiveOperation) -> None:
        self._active = active
        payload: dict[str, Any] = {"operation": active.name}
        if active.name in {"record", "speak_and_record"}:
            payload["record"] = {
                "audio": AudioFormat().as_dict(),
                "min_seconds": 2.0,
                "silence_seconds": 1.5,
                "max_seconds": self.record_max_seconds,
                "stop_authority": "server_webrtc_vad",
            }
        await self._send(message(
            "operation.start", operation_id=active.operation_id, **payload
        ))
        await self._wait(active.accepted, self.accept_timeout, "operation.accepted")

    async def _send_playback(self, active: _ActiveOperation, text: str) -> None:
        parts = split_text(text)
        if not parts:
            raise RemoteAudioError("EMPTY_TEXT", "播报文本为空", active.operation_id)
        for seq, part in enumerate(parts):
            generated = await self._synthesize(part)
            if isinstance(generated, str):
                try:
                    wav_bytes = base64.b64decode(generated, validate=True)
                except Exception as exc:
                    raise RemoteAudioError(
                        "TTS_INVALID_AUDIO", "TTS 返回了无效 Base64", active.operation_id
                    ) from exc
            else:
                wav_bytes = bytes(generated or b"")
            if not wav_bytes:
                raise RemoteAudioError("TTS_EMPTY_AUDIO", f"第 {seq} 段 TTS 为空", active.operation_id)
            try:
                sample_rate, channels = _wav_metadata(wav_bytes)
            except (ValueError, struct.error) as exc:
                raise RemoteAudioError(
                    "TTS_INVALID_WAV", f"第 {seq} 段不是有效 WAV", active.operation_id
                ) from exc
            active.playback_last_sent = seq == len(parts) - 1
            await self._send(message(
                "playback.chunk",
                operation_id=active.operation_id,
                seq=seq,
                chunk_count=len(parts),
                is_first=seq == 0,
                is_last=seq == len(parts) - 1,
                text=part,
                audio={
                    "encoding": "wav",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "data": base64.b64encode(wav_bytes).decode("ascii"),
                },
            ))

    async def speak(self, text: str) -> str:
        async with self._lock:
            active = self._new_active("speak")
            try:
                await self._start(active)
                await self._send_playback(active, text)
                await self._wait(active.playback_completed, self.playback_timeout, "playback.completed")
                await self._send(message(
                    "operation.completed", operation_id=active.operation_id,
                    operation="speak",
                ))
                return active.operation_id
            except Exception as exc:
                await self._cancel_on_error(active, exc)
                raise
            finally:
                self._active = None

    async def record(self) -> RecordingResult:
        async with self._lock:
            active = self._new_active("record")
            active.vad = WebRtcVadStopDetector(max_seconds=self.record_max_seconds)
            try:
                await self._start(active)
                await self._wait(active.record_started, self.record_start_timeout, "record.started")
                result = await self._finish_record(active)
                await self._send(message(
                    "operation.completed", operation_id=active.operation_id,
                    operation="record", duration_seconds=result.duration_seconds,
                    bytes=len(result.pcm), stop_reason=result.stop_reason,
                ))
                return result
            except Exception as exc:
                await self._cancel_on_error(active, exc)
                raise
            finally:
                self._active = None

    async def speak_and_record(self, text: str) -> RecordingResult:
        async with self._lock:
            active = self._new_active("speak_and_record")
            active.vad = WebRtcVadStopDetector(max_seconds=self.record_max_seconds)
            try:
                await self._start(active)
                await self._send_playback(active, text)
                await self._wait(active.playback_completed, self.playback_timeout, "playback.completed")
                await self._wait(active.record_started, self.record_start_timeout, "record.started")
                result = await self._finish_record(active)
                await self._send(message(
                    "operation.completed", operation_id=active.operation_id,
                    operation="speak_and_record", duration_seconds=result.duration_seconds,
                    bytes=len(result.pcm), stop_reason=result.stop_reason,
                ))
                return result
            except Exception as exc:
                await self._cancel_on_error(active, exc)
                raise
            finally:
                self._active = None

    async def _finish_record(self, active: _ActiveOperation) -> RecordingResult:
        try:
            await self._wait(
                active.record_completed,
                self.record_max_seconds + self.record_drain_timeout,
                "record.completed",
            )
        except RemoteAudioError as exc:
            if exc.code != "OPERATION_TIMEOUT":
                raise
            await self._request_record_stop(active, "max_duration")
            await self._wait(active.record_completed, self.record_drain_timeout, "record.completed")
        return RecordingResult(
            operation_id=active.operation_id,
            pcm=bytes(active.record_pcm),
            audio_format=AudioFormat(),
            stop_reason=active.stop_reason or "device_completed",
            started_at=active.record_started_at,
            completed_at=time.monotonic(),
        )

    async def _request_record_stop(self, active: _ActiveOperation, reason: str) -> None:
        if active.stop_sent:
            return
        active.stop_sent = True
        active.stop_reason = reason
        await self._send(message(
            "record.stop", operation_id=active.operation_id, reason=reason
        ))

    async def _cancel_on_error(self, active: _ActiveOperation, exc: Exception) -> None:
        try:
            await self._send(message(
                "operation.cancel", operation_id=active.operation_id,
                reason=getattr(exc, "code", "INTERNAL_ERROR"),
                message=str(exc),
            ))
        except Exception:
            pass

    async def handle_message(self, value: dict) -> None:
        validate_version(value)
        operation_id = require_operation_id(value)
        active = self._active
        if active is None or operation_id != active.operation_id:
            raise ProtocolError(
                "UNKNOWN_OPERATION", "消息不属于当前活动操作", operation_id
            )
        kind = str(value.get("type", ""))
        if kind == "operation.accepted":
            if not active.accepted.done():
                active.accepted.set_result(True)
            return
        if kind == "playback.completed":
            if active.name not in {"speak", "speak_and_record"}:
                raise ProtocolError("INVALID_STATE", "record 操作没有播放阶段", operation_id)
            if not active.playback_last_sent:
                raise ProtocolError("PLAYBACK_COMPLETED_EARLY", "最后一片发送前不能完成播放", operation_id)
            if not active.playback_completed.done():
                active.playback_completed.set_result(time.monotonic())
            return
        if kind == "record.started":
            if active.name == "speak_and_record" and not active.playback_completed.done():
                raise ProtocolError("RECORD_STARTED_EARLY", "最后一片播放结束前不得录音", operation_id)
            if not active.record_started.done():
                active.record_started_at = time.monotonic()
                active.record_started.set_result(True)
            return
        if kind == "record.chunk":
            await self._handle_record_chunk(active, value)
            return
        if kind == "record.completed":
            if not active.record_started.done():
                raise ProtocolError("INVALID_STATE", "record.started 之前不能完成录音", operation_id)
            if not active.record_last_seen:
                raise ProtocolError("MISSING_FINAL_CHUNK", "record.completed 前必须发送 is_last 音频片", operation_id)
            if not active.stop_sent:
                raise ProtocolError("RECORD_COMPLETED_EARLY", "必须等待服务端 record.stop", operation_id)
            if not active.record_completed.done():
                active.record_completed.set_result(True)
            return
        if kind in {"operation.error", "operation.cancelled"}:
            error = value.get("error") if isinstance(value.get("error"), dict) else {}
            exc = RemoteAudioError(
                str(error.get("code") or kind.upper().replace(".", "_")),
                str(error.get("message") or value.get("message") or "远程设备操作失败"),
                operation_id,
                bool(error.get("retryable", False)),
            )
            if not active.failed.done():
                active.failed.set_result(exc)
            return
        raise ProtocolError("UNKNOWN_MESSAGE_TYPE", f"不支持的操作消息: {kind}", operation_id)

    async def _handle_record_chunk(self, active: _ActiveOperation, value: dict) -> None:
        if not active.record_started.done():
            raise ProtocolError("INVALID_STATE", "record.started 之前不能发送音频", active.operation_id)
        seq = value.get("seq")
        if not isinstance(seq, int) or seq < 0:
            raise ProtocolError("INVALID_SEQUENCE", "record.chunk seq 必须是非负整数", active.operation_id)
        if seq < active.expected_record_seq:
            return  # A retransmitted packet is ignored, never appended twice.
        if seq != active.expected_record_seq:
            exc = RemoteAudioError(
                "AUDIO_SEQUENCE_GAP",
                f"录音序号不连续，期望 {active.expected_record_seq}，收到 {seq}",
                active.operation_id,
            )
            if not active.failed.done():
                active.failed.set_result(exc)
            return
        audio = value.get("audio")
        if not isinstance(audio, dict):
            raise ProtocolError("INVALID_AUDIO", "record.chunk 缺少 audio object", active.operation_id)
        expected = AudioFormat().as_dict()
        for key, wanted in expected.items():
            if audio.get(key) != wanted:
                raise ProtocolError(
                    "UNSUPPORTED_AUDIO_FORMAT",
                    f"record.audio.{key} 必须为 {wanted!r}",
                    active.operation_id,
                )
        pcm = decode_audio(audio.get("data"), active.operation_id)
        if len(pcm) % 2:
            raise ProtocolError("INVALID_AUDIO", "PCM 字节数必须为偶数", active.operation_id)
        active.record_pcm.extend(pcm)
        active.expected_record_seq += 1
        active.record_last_seen = bool(value.get("is_last", False))
        if active.vad and not active.stop_sent:
            decision = active.vad.feed(pcm)
            if decision.should_stop:
                await self._request_record_stop(active, decision.reason)

    async def close(self, detail: str = "设备连接已关闭") -> None:
        active = self._active
        if active and not active.failed.done():
            active.failed.set_result(RemoteAudioError(
                "CONNECTION_CLOSED", detail, active.operation_id, True
            ))

    def fail_protocol(self, exc: ProtocolError) -> None:
        active = self._active
        if active and exc.operation_id == active.operation_id and not active.failed.done():
            active.failed.set_result(RemoteAudioError(
                exc.code, str(exc), exc.operation_id, False
            ))

    @staticmethod
    def hello_ack(device_id: str) -> dict:
        return message(
            "hello.ack", device_id=device_id,
            operations=["speak", "record", "speak_and_record"],
            record_audio=AudioFormat().as_dict(),
        )

    @staticmethod
    def validate_hello(value: dict) -> str:
        validate_version(value)
        if value.get("type") != "hello":
            raise ProtocolError("HELLO_REQUIRED", "首条消息必须是 hello")
        device_id = str(value.get("device_id", "")).strip()
        if not device_id:
            raise ProtocolError("INVALID_DEVICE_ID", "hello 缺少 device_id")
        capabilities = value.get("capabilities")
        operations = capabilities.get("operations", []) if isinstance(capabilities, dict) else []
        required = {"speak", "record", "speak_and_record"}
        if not required.issubset(set(operations)):
            raise ProtocolError("MISSING_CAPABILITY", "设备必须支持三个原子操作")
        playback_formats = set(capabilities.get("playback_formats", []))
        record_format = capabilities.get("record_format", {})
        if "wav" not in playback_formats:
            raise ProtocolError("MISSING_CAPABILITY", "设备必须支持 WAV 播放")
        if not isinstance(record_format, dict) or any(
            record_format.get(key) != wanted
            for key, wanted in AudioFormat().as_dict().items()
        ):
            raise ProtocolError("MISSING_CAPABILITY", "设备必须支持 16kHz 单声道 PCM S16LE 录音")
        return device_id
