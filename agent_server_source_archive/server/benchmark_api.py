"""Small, dependency-free helpers shared by the Qwen3.5 HTTP frontends."""

from contextlib import contextmanager
import hashlib
import json
import time


def prompt_slot_messages(prompt_slots: dict):
    """Render SUHA identically for cached and uncached requests."""
    from prompt_protocol import messages_from_prompt_slots
    return messages_from_prompt_slots(prompt_slots)


def request_max_tokens(body: dict, default: int) -> int:
    """Validate the two OpenAI-compatible completion-limit spellings."""
    first = body.get("max_completion_tokens")
    second = body.get("max_tokens")
    if first is not None and second is not None and first != second:
        raise ValueError("max_tokens and max_completion_tokens must match when both are set")
    value = first if first is not None else second
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return value


@contextmanager
def completion_limit(pipeline, value: int):
    previous = int(getattr(pipeline, "max_new_tokens", value))
    pipeline.max_new_tokens = int(value)
    try:
        yield
    finally:
        pipeline.max_new_tokens = previous


def finish_reason(pipeline, completion_tokens: int, maximum: int) -> str:
    if completion_tokens >= maximum:
        return "length"
    model = getattr(pipeline, "model", None)
    if model is not None and int(getattr(model, "history_length", 0)) >= int(
            getattr(model, "SEQLEN", 2 ** 31)):
        return "length"
    return "stop"


def output_token_hash(pipeline) -> str:
    payload = json.dumps(list(getattr(pipeline, "last_output_token_ids", [])),
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def no_cache_diagnostics(prepared, pipeline, elapsed_ms: float) -> dict:
    boundary = prepared.boundaries
    logical = len(prepared.token_ids)
    return {
        "cache_event": "cold",
        "cache_hit": False,
        "matched_checkpoint": None,
        "logical_prompt_tokens": logical,
        "reused_prefix_tokens": 0,
        "physical_prefill_tokens": logical,
        "slot_tokens": {
            "system": boundary["S"],
            "user": boundary["U"] - boundary["S"],
            "history": boundary["H"] - boundary["U"],
            "attempt": boundary["A"] - boundary["H"],
        },
        "boundaries": dict(boundary),
        "history_length": int(getattr(pipeline.model, "history_length", 0)),
        "stage_ms": {**dict(getattr(pipeline, "last_stage_ms", {})),
                     "request": elapsed_ms},
        "request_ms": elapsed_ms,
        "snapshot_count": 0,
        "snapshot_bytes": 0,
    }


def timed_collect(generator):
    started = time.perf_counter()
    text = "".join(generator)
    return text, (time.perf_counter() - started) * 1000
