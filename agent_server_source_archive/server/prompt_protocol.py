"""Rendering for the stable SUHA prompt-slot wire protocols."""
from __future__ import annotations

from typing import Dict, Tuple


SLOT_NAMES = ("system", "user", "history", "attempt")
SUPPORTED_VERSIONS = {"suha.v1", "suha.v2", "suha.v3"}


def slot_names_for_version(version: str) -> Tuple[str, ...]:
    """Return the semantic slot order used on the wire.

    v3 separates prior natural-language turns from successful tool history.
    This keeps ``user`` limited to the current request plus its frozen plan and
    lets the cache reuse an append-only conversation prefix independently.
    """
    return (
        ("system", "conversation", "user", "history", "attempt")
        if version == "suha.v3" else SLOT_NAMES
    )


def validate_prompt_slots(prompt_slots: dict) -> Tuple[str, Dict[str, str]]:
    if not isinstance(prompt_slots, dict):
        raise ValueError("prompt_slots must be an object")
    version = prompt_slots.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported prompt_slots version: {version!r}; "
            "expected suha.v1, suha.v2 or suha.v3"
        )
    values: Dict[str, str] = {}
    for name in slot_names_for_version(version):
        value = prompt_slots.get(name)
        if not isinstance(value, str):
            raise ValueError(f"prompt_slots.{name} must be a string")
        values[name] = value
    if not values["system"].strip() or not values["user"].strip():
        raise ValueError("prompt_slots.system and prompt_slots.user cannot be empty")
    return version, values


def render_slot_text(prompt_slots: dict) -> Tuple[str, str, Dict[str, str]]:
    """Return system text, user text and boundary marker strings."""
    version, values = validate_prompt_slots(prompt_slots)
    if version == "suha.v1":
        markers = {
            "system": "[SLOT:system]\n",
            "user": "[SLOT:user]\n",
            "history": "\n[SLOT:history]\n",
            "attempt": "\n[SLOT:attempt]\n",
        }
        system_text = markers["system"] + values["system"]
        user_text = (
            markers["user"] + values["user"]
            + markers["history"] + values["history"]
            + markers["attempt"] + values["attempt"]
        )
    elif version == "suha.v2":
        markers = {
            "system": "<system>\n",
            "user": "<user>\n",
            "history": "\n</user>\n<history>\n",
            "attempt": "\n</history>\n<attempt>\n",
        }
        system_text = markers["system"] + values["system"] + "\n</system>"
        user_text = (
            markers["user"] + values["user"]
            + markers["history"] + values["history"]
            + markers["attempt"] + values["attempt"] + "\n</attempt>"
        )
    else:
        markers = {
            "system": "<system>\n",
            "conversation": "<conversation>\n",
            "user": "\n</conversation>\n<user>\n",
            "history": "\n</user>\n<history>\n",
            "attempt": "\n</history>\n<attempt>\n",
        }
        system_text = markers["system"] + values["system"] + "\n</system>"
        user_text = (
            markers["conversation"] + values["conversation"]
            + markers["user"] + values["user"]
            + markers["history"] + values["history"]
            + markers["attempt"] + values["attempt"] + "\n</attempt>"
        )
    return system_text, user_text, markers


def messages_from_prompt_slots(prompt_slots: dict) -> list:
    system_text, user_text, _ = render_slot_text(prompt_slots)
    image = prompt_slots.get("image")
    if image is None:
        user_content = user_text
    else:
        if not isinstance(image, str) or not image:
            raise ValueError("prompt_slots.image must be a non-empty string")
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]
