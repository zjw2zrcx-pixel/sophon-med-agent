"""WebSocket server that receives ASR text from browser and
manages voice interaction state machine."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VoiceState(Enum):
    LISTENING = "listening"
    HOTWORD_DETECTED = "hotword_detected"
    CAPTURING = "capturing"
    PROCESSING = "processing"


@dataclass
class BrowserMessage:
    """Message received from browser."""
    type: str
    text: str = ""
    segment: int = 0


@dataclass
class AgentMessage:
    """Message sent to browser."""
    type: str
    text: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "text": self.text, **self.data},
                          ensure_ascii=False)


class VoiceServer:
    """Manages WebSocket connections from browser and voice state machine.

    This runs as a lightweight WebSocket server within the VoiceMode.
    It receives ASR text segments forwarded by the browser, detects
    hotwords, captures utterances, and coordinates with the Agent.
    """

    def __init__(
        self,
        hotword_detector,
        capture_timeout: float = 5.0,
        silence_timeout: float = 2.0,
    ):
        self._detector = hotword_detector
        self.capture_timeout = capture_timeout
        self.silence_timeout = silence_timeout

        self.state = VoiceState.LISTENING
        self._capture_buffer: list[str] = []
        self._capture_start: float = 0.0
        self._last_segment_time: float = 0.0

        # Connected browser clients
        self._clients: list = []

        # Callback: called when a complete utterance is captured
        self._on_utterance: Optional[Callable] = None

    @property
    def detector(self):
        return self._detector

    def set_on_utterance(self, callback: Callable):
        """Set async callback for when an utterance is captured."""
        self._on_utterance = callback

    # ── client management ──

    def add_client(self, ws):
        self._clients.append(ws)
        logger.info(f"Voice client connected ({len(self._clients)} total)")

    def remove_client(self, ws):
        if ws in self._clients:
            self._clients.remove(ws)
            logger.info(f"Voice client disconnected ({len(self._clients)} remaining)")

    async def broadcast(self, msg: AgentMessage):
        """Send a message to all connected browser clients."""
        dead = []
        payload = msg.to_json()
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_client(ws)

    # ── message handling ──

    async def handle_message(self, ws, raw: str) -> Optional[str]:
        """Process a JSON message from browser. Returns response text if any."""
        try:
            data = json.loads(raw)
            msg = BrowserMessage(
                type=data.get("type", ""),
                text=data.get("text", ""),
                segment=data.get("segment", 0),
            )
        except json.JSONDecodeError:
            return None

        if msg.type == "segment":
            return await self._handle_segment(msg.text)
        elif msg.type == "reset":
            self._reset_state()
            return None
        return None

    async def _handle_segment(self, text: str) -> Optional[str]:
        """Process an ASR segment. Returns the utterance if ready for LLM."""
        text = text.strip()
        if not text:
            return None

        now = time.time()

        if self.state == VoiceState.LISTENING:
            # Feed to hotword detector
            hotword = self._detector.feed(text)
            if hotword:
                logger.info(f"Hotword '{hotword}' detected, acknowledging")
                await self.broadcast(AgentMessage(type="ack"))
                self._transition_to(VoiceState.HOTWORD_DETECTED)

                # Check if there's content after the hotword in this same segment
                after = self._detector.extract_after_hotword(text, hotword)
                if after:
                    self._capture_buffer.append(after)
                    self._capture_start = now
                    self._last_segment_time = now
                    self._transition_to(VoiceState.CAPTURING)
                else:
                    self._capture_start = now

        elif self.state == VoiceState.HOTWORD_DETECTED:
            self._capture_buffer.append(text)
            self._capture_start = self._capture_start or now
            self._last_segment_time = now
            self._transition_to(VoiceState.CAPTURING)

        elif self.state == VoiceState.CAPTURING:
            self._capture_buffer.append(text)
            self._last_segment_time = now

        elif self.state == VoiceState.PROCESSING:
            # Ignore segments while processing previous utterance
            return None

        return None

    async def check_timeout(self) -> Optional[str]:
        """Check for capture timeout. Returns captured utterance or None."""
        if self.state not in (VoiceState.HOTWORD_DETECTED, VoiceState.CAPTURING):
            return None

        now = time.time()

        # Hotword detected but no speech within capture_timeout
        if self.state == VoiceState.HOTWORD_DETECTED:
            if now - self._capture_start > self.capture_timeout:
                logger.info("Capture timeout (no speech after hotword)")
                await self.broadcast(AgentMessage(
                    type="speak", text="请说"
                ))
                self._reset_state()
                return None

        # Speech captured, waiting for silence
        if self.state == VoiceState.CAPTURING and self._capture_buffer:
            elapsed = now - self._last_segment_time
            if elapsed > self.silence_timeout:
                utterance = "".join(self._capture_buffer).strip()
                self._reset_state()
                if utterance:
                    logger.info(f"Utterance captured: {utterance[:80]}...")
                    if self._on_utterance:
                        await self._on_utterance(utterance)
                    return utterance
                return None

        # Capture timeout (e.g. user spoke too long or no more segments)
        if self.state == VoiceState.CAPTURING:
            if now - self._capture_start > self.capture_timeout:
                utterance = "".join(self._capture_buffer).strip()
                self._reset_state()
                if utterance:
                    logger.info(f"Utterance timeout, using: {utterance[:80]}...")
                    if self._on_utterance:
                        await self._on_utterance(utterance)
                    return utterance
                return None

        return None

    # ── state management ──

    def _transition_to(self, new_state: VoiceState):
        old = self.state
        self.state = new_state
        if old != new_state:
            logger.debug(f"Voice state: {old.value} -> {new_state.value}")

    def _reset_state(self):
        self.state = VoiceState.LISTENING
        self._capture_buffer.clear()
        self._capture_start = 0.0
        self._last_segment_time = 0.0

    def is_active(self) -> bool:
        """True if currently capturing or processing (not idle listening)."""
        return self.state != VoiceState.LISTENING
