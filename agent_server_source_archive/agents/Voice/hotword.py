"""Hotword detection with sliding-window buffer for streaming ASR text."""
from __future__ import annotations

import re
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class HotwordDetector:
    """Detect wake word in streaming ASR text with sliding window."""

    def __init__(self, hotwords=None, window_seconds=5.0):
        self.hotwords = hotwords or ["朋友"]
        self.window_seconds = window_seconds
        self._buffer = deque()

    @property
    def hotword_patterns(self):
        return [re.compile(re.escape(hw)) for hw in self.hotwords]

    def feed(self, text):
        if not text.strip():
            return None
        now = time.time()
        self._buffer.append((now, text))
        self._prune(now)
        return self._match(self._combined_text())

    def reset(self):
        self._buffer.clear()

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def _combined_text(self):
        if not self._buffer:
            return ""
        result = self._buffer[0][1]
        for i in range(1, len(self._buffer)):
            prev, curr = result, self._buffer[i][1]
            if curr in prev:
                continue
            if prev in curr:
                result = curr
            else:
                result += curr
        return result

    def _match(self, text):
        for pat in self.hotword_patterns:
            if pat.search(text):
                logger.info(f"Hotword: {pat.pattern}")
                return pat.pattern
        return None

    def extract_after_hotword(self, text, hotword):
        idx = text.rfind(hotword)
        if idx == -1:
            return text
        after = text[idx + len(hotword):]
        return re.sub(r"^[\s,，。.!！？?~～、]+", "", after).strip()
