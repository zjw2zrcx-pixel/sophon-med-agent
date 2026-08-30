#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Optional

from ..agent import Agent, AgentConfig
from ..Headless.manager import HeadlessManager
from ..model_selection import select_model

logger = logging.getLogger(__name__)


_AGENT_INSTANCE: Optional[Agent] = None
_MANAGER_INSTANCE: Optional[HeadlessManager] = None
_REQUESTED_MODEL: Optional[str] = None
_TRAJECTORY_DIR = "/data/structure/trajectories"
_TRAJECTORY_ENABLED = True


def _get_manager() -> HeadlessManager:
    global _AGENT_INSTANCE, _MANAGER_INSTANCE
    if _MANAGER_INSTANCE is not None:
        return _MANAGER_INSTANCE
    model = select_model(
        AgentConfig.server_url,
        requested=_REQUESTED_MODEL,
        default=AgentConfig.model_name,
        interactive=True,
    )
    config = AgentConfig(
        model_name=model,
        trajectory_dir=_TRAJECTORY_DIR,
        trajectory_enabled=_TRAJECTORY_ENABLED,
    )
    _AGENT_INSTANCE = Agent(config)
    _AGENT_INSTANCE.initialize()
    _MANAGER_INSTANCE = HeadlessManager(_AGENT_INSTANCE)
    return _MANAGER_INSTANCE


def _shutdown():
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is not None:
        try:
            asyncio.run(_AGENT_INSTANCE.shutdown())
        except Exception:
            pass


def _print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    sep = "  ".join("-" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(row, col_widths)) for row in rows)
    return f"{hdr}\n{sep}\n{body}"


def _submit_async(mgr, session_id, text, image=None):
    return asyncio.run(mgr.submit(session_id, text, image))


def cmd_session_create(args):
    mgr = _get_manager()
    sid = mgr.create_session(tags=args.tag, session_id=args.id)
    print(sid)


def cmd_session_list(args):
    mgr = _get_manager()
    sessions = mgr.list_sessions()
    if not sessions:
        print("No sessions.")
        return
    if args.format == "json":
        _print_json([s.to_dict() for s in sessions])
    elif args.format == "table":
        import time as t
        headers = ["ID", "Tags", "Msgs", "Created"]
        rows = []
        for s in sessions:
            created = t.strftime("%m-%d %H:%M", t.localtime(s.created_at))
            tags = ", ".join(s.tags) if s.tags else "-"
            msgs = str(len(s.session_obj.history) if s.session_obj else 0)
            rows.append([s.id, tags, msgs, created])
        print(_format_table(headers, rows))
    else:
        for s in sessions:
            msgs = len(s.session_obj.history) if s.session_obj else 0
            print(f"  {s.id}  tags={s.tags}  msgs={msgs}")


def cmd_session_show(args):
    mgr = _get_manager()
    s = mgr.get_session(args.id)
    if s is None:
        print(f"Session not found: {args.id}")
        sys.exit(1)
    _print_json(s.to_dict())


def cmd_session_delete(args):
    mgr = _get_manager()
    ok = mgr.delete_session(args.id)
    print("deleted" if ok else "not found")


def cmd_prompt(args):
    mgr = _get_manager()
    text = ""
    if args.file:
        text = Path(args.file).read_text("utf-8").strip()
    if args.text:
        text = (args.text + " " + text).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print("Empty input.")
        sys.exit(1)
    result = _submit_async(mgr, args.id, text, args.image)
    if args.format == "json":
        _print_json({
            "session": args.id,
            "text": result.text,
            "commands": [(c[0].type, c[0].name, c[1].success) for c in result.commands],
        })
    else:
        if result.text:
            print(result.text)
        if result.commands:
            for cmd, cres in result.commands:
                status = "ok" if cres.success else "FAIL"
                print(f"  [{cres.type}] {cmd.name} -> {status}")
                if cres.data and len(cres.data) < 500:
                    print(f"    {cres.data}")


def cmd_history(args):
    mgr = _get_manager()
    try:
        msgs = mgr.get_history(args.id, tail=args.tail)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.format == "json":
        data = []
        for m in msgs:
            data.append({
                "role": m.role,
                "content": m.content[:500],
                "has_image": m.image is not None,
                "source": m.metadata.get("source", ""),
            })
        _print_json(data)
    elif args.format == "md":
        print(f"# Session {args.id} History")
        print()
        for m in msgs:
            if m.role == "system":
                continue
            label = m.role.replace("_", " ").title()
            content = m.content[:300]
            print(f"**{label}:** {content}")
            print()
    else:
        for m in msgs:
            if m.role == "system":
                continue
            label = m.role.replace("_", " ").upper()
            content = m.content[:300]
            print(f"[{label}] {content}")


def cmd_status(args):
    mgr = _get_manager()
    agent = mgr.agent
    sessions = mgr.list_sessions()
    print("Agent status:")
    print("  Runtime: Voice")
    print(f"  Model: {agent.config.model_name}")
    print(f"  Server: {agent.config.server_url}")
    print(f"  Sessions: {len(sessions)}")


def cmd_config(args):
    mgr = _get_manager()
    cfg = mgr.agent.config
    if args.action == "show":
        _print_json({
            "server_url": cfg.server_url,
            "model_name": cfg.model_name,
            "max_context_tokens": cfg.max_context_tokens,
            "auto_compact_threshold": cfg.auto_compact_threshold,
            "history_visible_entries": cfg.history_visible_entries,
            "skill_dir": cfg.skill_dir,
            "trajectory_enabled": cfg.trajectory_enabled,
            "trajectory_dir": cfg.trajectory_dir,
        })
    elif args.action == "get":
        val = getattr(cfg, args.key, None)
        if val is None:
            print(f"Unknown config key: {args.key}")
        else:
            print(val)
    elif args.action == "set":
        if hasattr(cfg, args.key):
            old = getattr(cfg, args.key)
            if isinstance(old, bool):
                normalized = str(args.value).strip().lower()
                if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                    raise ValueError("boolean value must be true/false")
                typed_val = normalized in {"true", "1", "yes", "on"}
            else:
                typed_val = type(old)(args.value)
            if args.key == "history_visible_entries" and typed_val < 1:
                raise ValueError("history_visible_entries must be at least 1")
            setattr(cfg, args.key, typed_val)
            if args.key == "model_name":
                mgr.agent.api.default_model = typed_val
            elif args.key == "server_url":
                mgr.agent.api.server_url = str(typed_val).rstrip("/")
            elif args.key == "history_visible_entries":
                if mgr.agent.voice_mode is not None:
                    mgr.agent.voice_mode.config.history_visible_entries = typed_val
            elif args.key == "trajectory_enabled":
                mgr.agent.api.trajectory_writer.enabled = typed_val
            elif args.key == "trajectory_dir":
                from pathlib import Path as _Path
                mgr.agent.api.trajectory_writer.directory = _Path(typed_val).expanduser().resolve()
            print(f"set: {args.key} = {typed_val}")
        else:
            print(f"Unknown config key: {args.key}")


def cmd_batch(args):
    mgr = _get_manager()
    p = Path(args.file)
    if not p.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)
    commands = p.read_text("utf-8").strip().splitlines()
    for line in commands:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        print(f">>> {line}")
        try:
            _execute_line(line)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error: {e}")


def _execute_line(cmdline):
    parser = _build_parser()
    argv = cmdline.split()
    if not argv:
        return
    if argv[0] in ("exit", "quit"):
        _shutdown()
        sys.exit(0)
    if argv[0] == "help":
        parser.print_help()
        return
    try:
        ns = parser.parse_args(argv)
        if hasattr(ns, "func"):
            ns.func(ns)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error: {e}")


def repl():
    _get_manager()
    print("Headless Agent REPL. Type help for commands, exit to quit.")
    print()
    try:
        import readline
        readline.set_completer(lambda t, s: None)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        _execute_line(line)
    _shutdown()


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="agent-headless",
        description="Headless CLI for the Agent. Manage sessions, submit prompts, export history.",
    )
    parser.add_argument("--model", default=None, help="Chat model from Router /v1/models")
    parser.add_argument("--trajectory-dir", default="/data/structure/trajectories")
    parser.add_argument("--no-trajectory", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sess = sub.add_parser("session", help="Manage sessions")
    ssub = p_sess.add_subparsers(dest="action", required=True)

    p = ssub.add_parser("create", help="Create a new session")
    p.add_argument("--tag", action="append", default=None)
    p.add_argument("--id", default=None)
    p.set_defaults(func=cmd_session_create)

    p = ssub.add_parser("list", help="List sessions")
    p.add_argument("--format", choices=["table", "json", "text"], default="table")
    p.set_defaults(func=cmd_session_list)

    p = ssub.add_parser("show", help="Show session details")
    p.add_argument("id")
    p.set_defaults(func=cmd_session_show)

    p = ssub.add_parser("delete", help="Delete a session")
    p.add_argument("id")
    p.set_defaults(func=cmd_session_delete)

    p = sub.add_parser("prompt", help="Send a prompt to a session")
    p.add_argument("id")
    p.add_argument("--text", default="")
    p.add_argument("--file", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("history", help="View session history")
    p.add_argument("id")
    p.add_argument("--tail", type=int, default=None)
    p.add_argument("--format", choices=["text", "json", "md"], default="text")
    p.set_defaults(func=cmd_history)

    sub.add_parser("status", help="Show agent status").set_defaults(func=cmd_status)

    p = sub.add_parser("config", help="View or modify configuration")
    p.add_argument("action", choices=["show", "get", "set"])
    p.add_argument("key", nargs="?", default=None)
    p.add_argument("value", nargs="?", default=None)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("batch", help="Execute commands from a file")
    p.add_argument("file")
    p.set_defaults(func=cmd_batch)

    sub.add_parser("repl", help="Interactive REPL mode").set_defaults(func=lambda a: repl())

    return parser


def main():
    global _REQUESTED_MODEL, _TRAJECTORY_DIR, _TRAJECTORY_ENABLED
    parser = _build_parser()
    if len(sys.argv) == 1:
        repl()
        return
    args = parser.parse_args()
    _REQUESTED_MODEL = args.model
    _TRAJECTORY_DIR = args.trajectory_dir
    _TRAJECTORY_ENABLED = not args.no_trajectory
    if hasattr(args, "func"):
        args.func(args)
    _shutdown()


if __name__ == "__main__":
    main()
