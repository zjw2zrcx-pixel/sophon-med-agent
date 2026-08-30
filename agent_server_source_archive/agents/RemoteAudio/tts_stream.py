"""Sentence-oriented TTS streaming helpers."""
from __future__ import annotations

import re


def split_text(text: str, max_chars: int = 80) -> list[str]:
    """Split at natural Chinese/English boundaries and hard-limit long spans."""
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if not text:
        return []
    natural = re.split(r"(?<=[。！？!?；;.!])|\n+", text)
    chunks: list[str] = []
    for part in natural:
        part = part.strip()
        while len(part) > max_chars:
            cut = max(part.rfind(mark, 0, max_chars + 1) for mark in ("，", ",", "、", " "))
            if cut < max_chars // 2:
                cut = max_chars
            else:
                cut += 1
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks
