"""Shared TOML configuration loader for model services."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python 3.10 deployment
    import tomli as tomllib  # type: ignore


CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data.get("router"), dict):
        raise ValueError("config.toml requires [router]")
    servers = data.get("servers")
    if not isinstance(servers, list):
        raise ValueError("config.toml requires one or more [[servers]] entries")
    names = set()
    displays = set()
    for server in servers:
        if not isinstance(server, dict):
            raise ValueError("each [[servers]] entry must be a table")
        name = str(server.get("name", "")).strip()
        display = str(server.get("display_name", name)).strip()
        if not name or not display:
            raise ValueError("each server requires name and display_name")
        if name in names or display in displays:
            raise ValueError(f"duplicate model/server identifier: {name}/{display}")
        names.add(name)
        displays.add(display)
        backend = server.get("backend", "local")
        if backend not in {"local", "openai"}:
            raise ValueError(f"unsupported backend {backend!r} for {name}")
        if backend == "openai":
            for field in ("base_url", "upstream_model", "api_key_env"):
                if not str(server.get(field, "")).strip():
                    raise ValueError(f"remote server {name} requires {field}")
    return data

