"""Server-authoritative WebRTC VAD recording boundary detector."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VadDecision:
    should_stop: bool = False
    reason: str = ""


class WebRtcVadStopDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        min_seconds: float = 2.0,
        silence_seconds: float = 1.5,
        max_seconds: float = 10.0,
    ):
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("record 操作需要安装 webrtcvad") from exc
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("WebRTC VAD 不支持该采样率")
        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD 帧长必须是 10/20/30ms")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = sample_rate * frame_ms // 1000 * 2
        self.min_frames = int(min_seconds * 1000 / frame_ms)
        self.silence_frames = int(silence_seconds * 1000 / frame_ms)
        self.max_frames = int(max_seconds * 1000 / frame_ms)
        self._vad = webrtcvad.Vad(aggressiveness)
        self._pending = bytearray()
        self.total_frames = 0
        self.silent_run = 0
        self.speech_seen = False

    def feed(self, pcm: bytes) -> VadDecision:
        self._pending.extend(pcm)
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[: self.frame_bytes])
            del self._pending[: self.frame_bytes]
            self.total_frames += 1
            voiced = self._vad.is_speech(frame, self.sample_rate)
            if voiced:
                self.speech_seen = True
                self.silent_run = 0
            elif self.speech_seen:
                self.silent_run += 1
            if self.total_frames >= self.max_frames:
                return VadDecision(True, "max_duration")
            if (
                self.total_frames >= self.min_frames
                and self.speech_seen
                and self.silent_run >= self.silence_frames
            ):
                return VadDecision(True, "silence")
        return VadDecision()
