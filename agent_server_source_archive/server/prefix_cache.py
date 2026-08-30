"""Exact-token-prefix snapshot manager for the qwen3.5 history backend.

Checkpoints sit at semantic slot boundaries. A slot may be restored either as
an exact match or as a shorter token prefix when its content grew append-only;
no semantic-equivalence assumption is made.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Iterator, List, Optional, Tuple

from prompt_protocol import (
    render_slot_text,
    slot_names_for_version,
    validate_prompt_slots as validate_protocol_slots,
)


CHECKPOINT_BY_SLOT = {
    "system": "S", "conversation": "C", "user": "U",
    "history": "H", "attempt": "A",
}


@dataclass
class PreparedPrompt:
    token_ids: List[int]
    boundaries: Dict[str, int]
    checkpoint_names: Tuple[str, ...]


@dataclass
class Checkpoint:
    handle: int
    token_ids: Tuple[int, ...]
    pending_token: int
    bytes: int


@dataclass
class SessionCache:
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)
    last_inference: float = field(default_factory=time.monotonic)


def validate_prompt_slots(prompt_slots: dict) -> dict:
    _, values = validate_protocol_slots(prompt_slots)
    if prompt_slots.get("image") is not None:
        raise ValueError(
            "SUHA prefix cache is text-only; put an image description in "
            "history or attempt")
    return values


def _longest_common_prefix(left: List[int], right: List[int]) -> int:
    length = min(len(left), len(right))
    index = 0
    while index < length and left[index] == right[index]:
        index += 1
    return index


def prepare_prompt(tokenizer, prompt_slots: dict,
                   enable_thinking: bool = False) -> PreparedPrompt:
    """Render the normal chat template and locate safe token boundaries.

    A slot boundary is independently tokenized and then retreated to its exact
    longest common prefix with the full prompt.  This remains correct when a
    tokenizer merges characters across a textual slot boundary.
    """
    values = validate_prompt_slots(prompt_slots)
    system_text, user_text, markers = render_slot_text(prompt_slots)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(enable_thinking))
    full_tokens = list(tokenizer.encode(rendered, add_special_tokens=False))

    version = str(prompt_slots["version"])
    slot_names = slot_names_for_version(version)
    checkpoint_names = tuple(CHECKPOINT_BY_SLOT[name] for name in slot_names)
    char_boundaries: Dict[str, int] = {}
    cursor = 0
    for slot in slot_names:
        marker = markers[slot]
        marker_start = rendered.find(marker, cursor)
        if marker_start < 0:
            raise RuntimeError(f"chat template did not preserve the {slot} slot")
        value_start = marker_start + len(marker)
        value = values[slot]
        if rendered[value_start:value_start + len(value)] != value:
            raise RuntimeError(f"chat template changed the {slot} slot content")
        char_boundaries[CHECKPOINT_BY_SLOT[slot]] = value_start + len(value)
        cursor = value_start + len(value)

    # The final checkpoint also owns template closing markers and the assistant
    # generation prefix. Earlier checkpoints deliberately end at slot content,
    # allowing a later request to append within that same slot safely.
    char_boundaries[checkpoint_names[-1]] = len(rendered)
    boundaries = {}
    previous = 0
    for name in checkpoint_names:
        if name == checkpoint_names[-1]:
            token_end = len(full_tokens)
        else:
            prefix_tokens = list(tokenizer.encode(
                rendered[:char_boundaries[name]], add_special_tokens=False))
            token_end = _longest_common_prefix(prefix_tokens, full_tokens)
        if token_end <= previous:
            raise RuntimeError(f"empty or overlapping {name} token checkpoint")
        boundaries[name] = token_end
        previous = token_end
    return PreparedPrompt(full_tokens, boundaries, checkpoint_names)


class ExactPrefixCacheManager:
    def __init__(self, pipeline, max_sessions: int, max_snapshot_bytes: int,
                 logger=None, enable_thinking: bool = False):
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be positive")
        self.pipeline = pipeline
        self.model = pipeline.model
        self.max_sessions = int(max_sessions)
        self.max_snapshot_bytes = int(max_snapshot_bytes)
        self.sessions: Dict[str, SessionCache] = {}
        self.logger = logger
        self.enable_thinking = bool(enable_thinking)
        self.hits = {name: 0 for name in CHECKPOINT_BY_SLOT.values()}
        self.cold_misses = 0
        self.last_diagnostics: Dict[str, object] = {}

    @property
    def snapshot_bytes(self) -> int:
        return int(self.model.total_snapshot_bytes())

    def _log(self, message, *args):
        if self.logger is not None:
            self.logger.info(message, *args)

    def _release_checkpoint(self, checkpoint: Checkpoint):
        self.model.release_snapshot(checkpoint.handle)

    def evict(self, session_id: str, reason: str = "explicit") -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        for checkpoint in session.checkpoints.values():
            self._release_checkpoint(checkpoint)
        self._log("Prefix cache evicted session=%s reason=%s", session_id, reason)
        return True

    def clear(self):
        for session_id in list(self.sessions):
            self.evict(session_id, reason="shutdown")

    def _evict_lru(self, exclude: Optional[str], reason: str) -> bool:
        candidates = [
            (session.last_inference, sid) for sid, session in self.sessions.items()
            if sid != exclude
        ]
        if not candidates:
            return False
        _, session_id = min(candidates)
        return self.evict(session_id, reason=reason)

    def _get_or_create_session(self, session_id: str) -> SessionCache:
        session = self.sessions.get(session_id)
        if session is not None:
            return session
        while len(self.sessions) >= self.max_sessions:
            if not self._evict_lru(exclude=None, reason="session_limit"):
                raise RuntimeError("no evictable prefix-cache session")
        session = SessionCache()
        self.sessions[session_id] = session
        return session

    def _ensure_budget(self, required_bytes: int, session_id: str) -> bool:
        if required_bytes > self.max_snapshot_bytes:
            return False
        while self.snapshot_bytes + required_bytes > self.max_snapshot_bytes:
            if not self._evict_lru(exclude=session_id, reason="snapshot_budget"):
                return False
        return True

    def _drop_from(self, session: SessionCache, checkpoint_name: str,
                   checkpoint_names: Tuple[str, ...]):
        start = checkpoint_names.index(checkpoint_name)
        for name in checkpoint_names[start:]:
            checkpoint = session.checkpoints.pop(name, None)
            if checkpoint is not None:
                self._release_checkpoint(checkpoint)

    @staticmethod
    def _matches(checkpoint: Checkpoint, tokens: List[int]) -> bool:
        length = len(checkpoint.token_ids)
        return length <= len(tokens) and tuple(tokens[:length]) == checkpoint.token_ids

    def _deepest_match(self, session: SessionCache,
                       prompt: PreparedPrompt) -> Optional[str]:
        for name in reversed(prompt.checkpoint_names):
            checkpoint = session.checkpoints.get(name)
            expected_end = prompt.boundaries[name]
            if (checkpoint is not None
                    and len(checkpoint.token_ids) <= expected_end
                    and self._matches(checkpoint, prompt.token_ids)):
                return name
        return None

    def _save(self, session_id: str, session: SessionCache, name: str,
              tokens: List[int], pending_token: int):
        previous = session.checkpoints.pop(name, None)
        if previous is not None:
            self._release_checkpoint(previous)
        required = int(self.model.estimate_snapshot_bytes())
        if not self._ensure_budget(required, session_id):
            self._log(
                "Prefix snapshot skipped session=%s checkpoint=%s bytes=%d budget=%d",
                session_id, name, required, self.max_snapshot_bytes)
            return
        handle = int(self.model.save_snapshot())
        actual_bytes = int(self.model.snapshot_bytes(handle))
        session.checkpoints[name] = Checkpoint(
            handle=handle,
            token_ids=tuple(tokens),
            pending_token=int(pending_token),
            bytes=actual_bytes,
        )

    def generate(self, session_id: str, prompt_slots: dict,
                 clear: bool = False) -> Iterator[str]:
        if not session_id:
            raise ValueError("x-session-id is required for SUHA prefix caching")
        if clear:
            self.evict(session_id, reason="client_clear")
        request_started = time.perf_counter()
        stage_ms = {"tokenizer": 0.0, "snapshot_restore": 0.0,
                    "prefill": 0.0, "snapshot_save": 0.0, "decode": 0.0}
        first_token_ms = None
        thinking_end_ms = None
        final_output_ttft_ms = None
        generated_text = ""
        started = time.perf_counter()
        prompt = prepare_prompt(
            self.pipeline.tokenizer, prompt_slots, self.enable_thinking)
        stage_ms["tokenizer"] = (time.perf_counter() - started) * 1000
        seqlen = int(getattr(self.model, "SEQLEN", 0) or 0)
        # Native history prefill retains one pending token and asserts
        # history_length + token_length < SEQLEN. Reject at the Python boundary
        # before any cache mutation so an oversized case cannot abort the whole
        # model server process.
        if seqlen and len(prompt.token_ids) >= seqlen - 1:
            raise ValueError(
                f"Input length {len(prompt.token_ids)} exceeds safe maximum "
                f"{seqlen - 2} for history prefill"
            )
        session = self._get_or_create_session(session_id)
        match = self._deepest_match(session, prompt)
        restored_tokens = 0

        if match is None:
            self.cold_misses += 1
            self._drop_from(session, prompt.checkpoint_names[0],
                            prompt.checkpoint_names)
            # Release checkpoints left by a different protocol version.
            for stale in list(session.checkpoints):
                checkpoint = session.checkpoints.pop(stale)
                self._release_checkpoint(checkpoint)
            self.model.clear_history()
            start = 0
            pending_token = None
        else:
            self.hits[match] += 1
            checkpoint = session.checkpoints[match]
            started = time.perf_counter()
            pending_token = int(self.model.restore_snapshot(checkpoint.handle))
            stage_ms["snapshot_restore"] += (time.perf_counter() - started) * 1000
            start = len(checkpoint.token_ids)
            restored_tokens = start
            match_index = prompt.checkpoint_names.index(match)
            if match_index + 1 < len(prompt.checkpoint_names):
                self._drop_from(session, prompt.checkpoint_names[match_index + 1],
                                prompt.checkpoint_names)
            self._log(
                "Prefix cache hit session=%s checkpoint=%s tokens=%d",
                session_id, match, start)

        try:
            if match is None:
                first_to_build = 0
            elif start < prompt.boundaries[match]:
                # The slot grew append-only. Restore its older content and
                # prefill only the newly appended suffix before refreshing the
                # checkpoint at the new boundary.
                first_to_build = prompt.checkpoint_names.index(match)
            else:
                first_to_build = prompt.checkpoint_names.index(match) + 1
            for name in prompt.checkpoint_names[first_to_build:]:
                end = prompt.boundaries[name]
                self._log(
                    "Prefix prefill session=%s checkpoint=%s range=%d:%d cache_history=%d",
                    session_id, name, start, end, int(self.model.history_length))
                started = time.perf_counter()
                pending_token = self.pipeline.prefill_token_ids(
                    prompt.token_ids[start:end], start)
                stage_ms["prefill"] += (time.perf_counter() - started) * 1000
                self._log(
                    "Prefix prefill complete session=%s checkpoint=%s cache_history=%d",
                    session_id, name, int(self.model.history_length))
                started = time.perf_counter()
                self._save(
                    session_id, session, name, prompt.token_ids[:end], pending_token)
                stage_ms["snapshot_save"] += (time.perf_counter() - started) * 1000
                start = end

            if pending_token is None:
                raise RuntimeError("prefix cache produced no pending token")
            started = time.perf_counter()
            for chunk in self.pipeline.generate_from_pending(
                    pending_token, len(prompt.token_ids)):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - request_started) * 1000
                generated_text += chunk
                if self.enable_thinking:
                    if thinking_end_ms is None and "</think>" in generated_text:
                        thinking_end_ms = (
                            time.perf_counter() - request_started) * 1000
                        if generated_text.partition("</think>")[2].strip():
                            final_output_ttft_ms = thinking_end_ms
                    elif (thinking_end_ms is not None
                          and final_output_ttft_ms is None and chunk.strip()):
                        final_output_ttft_ms = (
                            time.perf_counter() - request_started) * 1000
                yield chunk
            stage_ms["decode"] = (time.perf_counter() - started) * 1000
        finally:
            session.last_inference = time.monotonic()
            reused = restored_tokens
            boundary = prompt.boundaries
            slot_names = slot_names_for_version(prompt_slots["version"])
            self.last_diagnostics = {
                "prefix_cache_enabled": True,
                "thinking_enabled": self.enable_thinking,
                "thinking_end_internal_ms": thinking_end_ms,
                "final_output_ttft_internal_ms": final_output_ttft_ms,
                "cache_event": "cold" if match is None else match,
                "cache_hit": match is not None,
                "matched_checkpoint": match,
                "logical_prompt_tokens": len(prompt.token_ids),
                "reused_prefix_tokens": reused,
                "physical_prefill_tokens": len(prompt.token_ids) - reused,
                "cache_match_mode": (
                    "cold" if match is None else
                    ("partial" if reused < boundary[match] else "exact")
                ),
                "slot_tokens": {
                    slot: boundary[checkpoint] - (
                        boundary[CHECKPOINT_BY_SLOT[slot_names[index - 1]]]
                        if index else 0
                    )
                    for index, slot in enumerate(slot_names)
                    for checkpoint in (CHECKPOINT_BY_SLOT[slot],)
                },
                "boundaries": dict(boundary),
                "history_length": int(self.model.history_length),
                "stage_ms": stage_ms,
                "request_ms": (time.perf_counter() - request_started) * 1000,
                "ttft_internal_ms": first_token_ms,
                "snapshot_count": sum(
                    len(item.checkpoints) for item in self.sessions.values()),
                "snapshot_bytes": self.snapshot_bytes,
            }

    def stats(self) -> dict:
        return {
            "prefix_cache_enabled": True,
            "max_sessions": self.max_sessions,
            "sessions": len(self.sessions),
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "snapshot_bytes": self.snapshot_bytes,
            "snapshot_count": sum(
                len(session.checkpoints) for session in self.sessions.values()),
            "hits": dict(self.hits),
            "cold_misses": self.cold_misses,
        }


class FullPrefillManager:
    """Prompt-slot execution with snapshots and prefix reuse disabled.

    The full rendered prompt is tokenized and physically prefetched from
    position zero for every model invocation.  This keeps the prompt protocol
    and history-capable runtime identical to the cached path while providing a
    clean no-cache benchmark control.
    """

    def __init__(self, pipeline, logger=None, enable_thinking: bool = False):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.logger = logger
        self.enable_thinking = bool(enable_thinking)
        self.last_diagnostics: Dict[str, object] = {}
        self.requests = 0

    def clear(self):
        self.model.clear_history()

    def generate(self, session_id: str, prompt_slots: dict,
                 clear: bool = False) -> Iterator[str]:
        del clear
        if not session_id:
            raise ValueError("x-session-id is required for benchmark tracing")
        request_started = time.perf_counter()
        stage_ms = {"tokenizer": 0.0, "snapshot_restore": 0.0,
                    "prefill": 0.0, "snapshot_save": 0.0, "decode": 0.0}
        first_token_ms = None
        thinking_end_ms = None
        final_output_ttft_ms = None
        generated_text = ""
        started = time.perf_counter()
        prompt = prepare_prompt(
            self.pipeline.tokenizer, prompt_slots, self.enable_thinking)
        stage_ms["tokenizer"] = (time.perf_counter() - started) * 1000
        seqlen = int(getattr(self.model, "SEQLEN", 0) or 0)
        if seqlen and len(prompt.token_ids) >= seqlen - 1:
            raise ValueError(
                f"Input length {len(prompt.token_ids)} exceeds safe maximum "
                f"{seqlen - 2} for history prefill"
            )

        pending_token = None
        try:
            self.model.clear_history()
            started = time.perf_counter()
            pending_token = self.pipeline.prefill_token_ids(
                prompt.token_ids, 0)
            stage_ms["prefill"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            for chunk in self.pipeline.generate_from_pending(
                    pending_token, len(prompt.token_ids)):
                if first_token_ms is None:
                    first_token_ms = (
                        time.perf_counter() - request_started) * 1000
                generated_text += chunk
                if self.enable_thinking:
                    if thinking_end_ms is None and "</think>" in generated_text:
                        thinking_end_ms = (
                            time.perf_counter() - request_started) * 1000
                        if generated_text.partition("</think>")[2].strip():
                            final_output_ttft_ms = thinking_end_ms
                    elif (thinking_end_ms is not None
                          and final_output_ttft_ms is None and chunk.strip()):
                        final_output_ttft_ms = (
                            time.perf_counter() - request_started) * 1000
                yield chunk
            stage_ms["decode"] = (time.perf_counter() - started) * 1000
        finally:
            self.requests += 1
            boundary = prompt.boundaries
            slot_names = slot_names_for_version(prompt_slots["version"])
            self.last_diagnostics = {
                "prefix_cache_enabled": False,
                "thinking_enabled": self.enable_thinking,
                "thinking_end_internal_ms": thinking_end_ms,
                "final_output_ttft_internal_ms": final_output_ttft_ms,
                "cache_event": "disabled_full_prefill",
                "cache_hit": False,
                "matched_checkpoint": None,
                "logical_prompt_tokens": len(prompt.token_ids),
                "reused_prefix_tokens": 0,
                "physical_prefill_tokens": len(prompt.token_ids),
                "cache_match_mode": "disabled",
                "slot_tokens": {
                    slot: boundary[checkpoint] - (
                        boundary[CHECKPOINT_BY_SLOT[slot_names[index - 1]]]
                        if index else 0
                    )
                    for index, slot in enumerate(slot_names)
                    for checkpoint in (CHECKPOINT_BY_SLOT[slot],)
                },
                "boundaries": dict(boundary),
                "history_length": int(self.model.history_length),
                "stage_ms": stage_ms,
                "request_ms": (time.perf_counter() - request_started) * 1000,
                "ttft_internal_ms": first_token_ms,
                "snapshot_count": 0,
                "snapshot_bytes": 0,
            }

    def stats(self) -> dict:
        return {
            "prefix_cache_enabled": False,
            "mode": "full_prefill",
            "requests": self.requests,
            "sessions": 0,
            "snapshot_bytes": 0,
            "snapshot_count": 0,
        }
