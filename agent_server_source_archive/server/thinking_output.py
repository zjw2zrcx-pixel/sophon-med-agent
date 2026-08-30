"""Separate Qwen thinking text from the command-bearing visible response."""

from __future__ import annotations

import hashlib


def split_thinking_output(text: str, enabled: bool) -> tuple[str, str, bool]:
    """Return ``(thinking, visible, closed)`` without feeding think to Agent."""
    value = str(text or "")
    if not enabled:
        return "", value, False
    before, marker, after = value.partition("</think>")
    if not marker:
        return before.removeprefix("<think>").strip(), "", False
    thinking = before.removeprefix("<think>").strip()
    return thinking, after.lstrip(), True


def thinking_metadata(tokenizer, thinking: str, closed: bool) -> dict:
    value = str(thinking or "")
    tokens = tokenizer.encode(value, add_special_tokens=False) if value else []
    return {
        "thinking_closed": bool(closed),
        "thinking_chars": len(value),
        "thinking_tokens": len(tokens),
        "thinking_sha256": hashlib.sha256(value.encode()).hexdigest(),
    }
