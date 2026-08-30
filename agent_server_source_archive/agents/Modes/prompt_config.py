from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PROMPT_CONFIG_PATH = Path(__file__).with_name("prompts.json")


@lru_cache(maxsize=1)
def load_prompt_config() -> dict:
    data = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0":
        raise ValueError("unsupported prompt configuration schema")
    fragments = data.get("fragments")
    modes = data.get("modes")
    if not isinstance(fragments, dict) or not isinstance(modes, dict):
        raise ValueError("prompt configuration requires fragments and modes")
    for mode, names in modes.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f"prompt mode {mode!r} must contain fragments")
        missing = [name for name in names if name not in fragments]
        if missing:
            raise ValueError(f"prompt mode {mode!r} references missing fragments: {missing}")
    return data


def get_mode_prompt(mode: str) -> str:
    data = load_prompt_config()
    try:
        names = data["modes"][mode]
    except KeyError as exc:
        raise KeyError(f"unknown prompt mode: {mode}") from exc
    return "\n\n".join(str(data["fragments"][name]).strip() for name in names)


VOICE_SYSTEM_PROMPT = get_mode_prompt("Voice")
BENCHMARK_SYSTEM_PROMPT = get_mode_prompt("Benchmark")
