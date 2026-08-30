from __future__ import annotations

import asyncio
import time
import uuid
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agent import Agent
from ..API.session import Session, Message
from ..Modes.base import LoopResult

logger = logging.getLogger(__name__)


@dataclass
class HeadlessSession:
    """A headless session wrapping an isolated conversation."""
    id: str = ""
    session_obj: Optional[Session] = None
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        ledger = self.session_obj.prompt_slots.history if self.session_obj else None
        execution = self.session_obj.execution_state if self.session_obj else None
        return {
            "id": self.id,
            "mode": "Voice",
            "tags": list(self.tags),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": len(self.session_obj.history) if self.session_obj else 0,
            "token_estimate": self.session_obj.estimate_tokens() if self.session_obj else 0,
            "history_stored_entries": len(ledger.entries) if ledger else 0,
            "history_visible_entries": len(ledger.visible_entries()) if ledger else 0,
            "history_hidden_entries": ledger.hidden_count if ledger else 0,
            "last_usage": dict(self.session_obj.last_usage) if self.session_obj else {},
            "execution_state": execution.projection() if execution else None,
        }


class HeadlessManager:
    """Manages multiple isolated headless sessions on a single Agent."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.sessions: Dict[str, HeadlessSession] = {}

    # Session CRUD

    def create_session(
        self,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        if session_id and session_id in self.sessions:
            raise ValueError(f"Session {session_id!r} already exists")

        sid = session_id or uuid.uuid4().hex[:8]
        if sid in self.sessions:
            sid = uuid.uuid4().hex[:8]

        inner = Session()
        if self.agent.voice_mode:
            full_prompt = self.agent.voice_mode.build_full_prompt(inner)
            inner.add_message("system", full_prompt)

        hs = HeadlessSession(
            id=sid,
            session_obj=inner,
            tags=tags or [],
        )
        self.sessions[sid] = hs
        logger.info(f"Headless Voice session created: {sid}")
        return sid

    def get_session(self, session_id: str) -> Optional[HeadlessSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[HeadlessSession]:
        return list(self.sessions.values())

    def delete_session(self, session_id: str) -> bool:
        hs = self.sessions.pop(session_id, None)
        if hs is not None:
            inner_session = hs.session_obj
            if inner_session is not None:
                try:
                    asyncio.get_running_loop().create_task(
                        self.agent.api.release_history_session_cache(inner_session.id)
                    )
                except RuntimeError:
                    logger.warning(
                        "Headless session %s removed outside an event loop; local KV cache will expire by TTL",
                        session_id,
                    )
            logger.info(f"Headless session deleted: {session_id}")
            return True
        return False

    def _init_session_obj(self, hs: HeadlessSession):
        if hs.session_obj is not None and hs.session_obj.history:
            return
        hs.session_obj = Session()
        if self.agent.voice_mode:
            full_prompt = self.agent.voice_mode.build_full_prompt(hs.session_obj)
            hs.session_obj.add_message("system", full_prompt)

    # Prompt submission

    async def submit(
        self,
        session_id: str,
        text: str,
        image: Optional[str] = None,
        tool_context_extra: Optional[dict] = None,
    ) -> LoopResult:
        hs = self.sessions.get(session_id)
        if hs is None:
            raise KeyError(f"Session not found: {session_id}")

        self._init_session_obj(hs)

        result = await self.agent.handle_input_in_session(
            user_input=text,
            image=image,
            session=hs.session_obj,
            tool_context_extra=tool_context_extra,
        )

        hs.last_active = time.time()
        return result

    # History / Export

    def get_history(
        self,
        session_id: str,
        tail: Optional[int] = None,
    ) -> List[Message]:
        hs = self.sessions.get(session_id)
        if hs is None:
            raise KeyError(f"Session not found: {session_id}")
        if hs.session_obj is None:
            return []
        msgs = hs.session_obj.history
        if tail is not None and tail > 0:
            msgs = msgs[-tail:]
        return msgs

    def export_session(self, session_id: str, fmt: str = "json") -> str:
        hs = self.sessions.get(session_id)
        if hs is None:
            raise KeyError(f"Session not found: {session_id}")

        if fmt == "json":
            return json.dumps(hs.to_dict(), indent=2, ensure_ascii=False)
        elif fmt == "md":
            lines = [
                f"# Session: {hs.id}",
                "- Mode: Voice",
                f"- Tags: {', '.join(hs.tags) if hs.tags else '(none)'}",
                f"- Messages: {len(hs.session_obj.history) if hs.session_obj else 0}",
                "",
                "## Conversation",
                "",
            ]
            if hs.session_obj:
                for msg in hs.session_obj.history:
                    if msg.role == "system":
                        continue
                    label = msg.role.replace("_", " ").title()
                    lines.append(f"**{label}:** {msg.content[:200]}")
                    lines.append("")
            return "\n".join(lines)
        else:
            lines = [
                f"Session: {hs.id}  Mode: Voice  Tags: {', '.join(hs.tags) if hs.tags else '-'}",
            ]
            if hs.session_obj:
                for msg in hs.session_obj.history:
                    if msg.role == "system":
                        continue
                    label = msg.role.replace("_", " ").title()
                    content = msg.content[:300]
                    lines.append(f"  [{label}] {content}")
            return "\n".join(lines)
