"""Bounded in-memory storage for observable Agent execution events.

The trace deliberately contains only events emitted by the application
(model output, parsed commands and tool results).  It is not a transport for
hidden model reasoning.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List


MAX_TRACES = 100
MAX_EVENTS_PER_TRACE = 200
MAX_STRING_LENGTH = 12_000
MAX_CONTAINER_ITEMS = 100


def _bounded(value: Any, depth: int = 0) -> Any:
    """Make an event JSON-safe and prevent accidental unbounded retention."""
    if depth >= 6:
        return "[内容层级过深，已省略]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return value[:MAX_STRING_LENGTH] + "\n…[内容已截断]"
    if isinstance(value, dict):
        return {
            str(k)[:200]: _bounded(v, depth + 1)
            for k, v in list(value.items())[:MAX_CONTAINER_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(v, depth + 1) for v in value[:MAX_CONTAINER_ITEMS]]
    return _bounded(str(value), depth + 1)


class TraceStore:
    """Keep the latest traces in insertion order for reconnecting viewers."""

    def __init__(self, max_traces: int = MAX_TRACES):
        self.max_traces = max(1, int(max_traces))
        self._traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def add(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_event, dict):
            raise ValueError("event must be an object")

        trace_id = str(raw_event.get("trace_id", "")).strip()[:128]
        event_type = str(raw_event.get("type", "")).strip()[:64]
        if not trace_id:
            raise ValueError("trace_id is required")
        if not event_type:
            raise ValueError("type is required")

        event = _bounded(dict(raw_event))
        event["trace_id"] = trace_id
        event["type"] = event_type
        event.setdefault("timestamp", time.time())

        trace = self._traces.get(trace_id)
        if trace is None:
            trace = {
                "trace_id": trace_id,
                "session_id": str(event.get("session_id", ""))[:128],
                "mode": str(event.get("mode", ""))[:64],
                "created_at": event["timestamp"],
                "updated_at": event["timestamp"],
                "status": "running",
                "events": [],
            }
            self._traces[trace_id] = trace

        if event.get("session_id"):
            trace["session_id"] = str(event["session_id"])[:128]
        if event.get("mode"):
            trace["mode"] = str(event["mode"])[:64]
        trace["updated_at"] = event["timestamp"]
        trace["events"].append(event)
        if len(trace["events"]) > MAX_EVENTS_PER_TRACE:
            del trace["events"][:-MAX_EVENTS_PER_TRACE]

        if event_type == "turn_end":
            payload = event.get("payload") or {}
            trace["status"] = str(payload.get("status", "completed"))[:32]
        elif event_type == "error":
            trace["status"] = "error"

        self._traces.move_to_end(trace_id)
        while len(self._traces) > self.max_traces:
            self._traces.popitem(last=False)
        return deepcopy(event)

    def snapshot(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), self.max_traces))
        values = list(self._traces.values())[-limit:]
        return deepcopy(values)

    def clear(self) -> None:
        self._traces.clear()

