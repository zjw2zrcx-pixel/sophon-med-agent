"""Startup-time chat model discovery, selection and local loading."""
from __future__ import annotations

import sys
import time
from typing import Optional

import httpx


def select_model(
    router_url: str,
    requested: Optional[str] = None,
    default: str = "qwen3.5-4b-history",
    interactive: bool = True,
    ready_timeout: float = 600.0,
) -> str:
    base = router_url.rstrip("/")
    response = httpx.get(base + "/v1/models", timeout=10.0)
    response.raise_for_status()
    models = [m for m in response.json().get("data", []) if m.get("type") == "chat"]
    if not models:
        raise RuntimeError("Router did not report any chat models")
    by_id = {str(model.get("id")): model for model in models}

    selected = requested
    if not selected and interactive and sys.stdin.isatty():
        print("Available chat models:")
        for index, model in enumerate(models, 1):
            print(f"  {index}. {model['id']} [{model.get('backend','?')}/{model.get('status','?')}]")
        raw = input(f"Select model [{default}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            selected = str(models[int(raw) - 1]["id"])
        else:
            selected = raw or default
    selected = selected or default
    if selected not in by_id:
        raise ValueError(f"Unknown chat model {selected!r}; available: {', '.join(by_id)}")

    model = by_id[selected]
    if model.get("backend") == "openai":
        if model.get("status") != "online":
            raise RuntimeError(f"Remote model {selected!r} is unavailable (check its API key)")
        return selected
    if model.get("status") == "online":
        return selected

    load = httpx.post(base + f"/v1/load/{selected}", timeout=15.0)
    if load.status_code not in {200, 202, 409}:
        raise RuntimeError(f"Unable to load {selected}: HTTP {load.status_code} {load.text[:300]}")
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        time.sleep(2.0)
        status_response = httpx.get(base + "/v1/models", timeout=10.0)
        status_response.raise_for_status()
        current = {
            str(item.get("id")): item
            for item in status_response.json().get("data", [])
        }.get(selected, {})
        if current.get("status") == "online":
            return selected
        if current.get("status") == "offline" and load.status_code == 409:
            raise RuntimeError(f"Local model {selected!r} could not become ready")
    raise TimeoutError(f"Timed out waiting for local model {selected!r}")
