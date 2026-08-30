#!/usr/bin/env python3
import sys
import os
import time
import signal
import argparse
import subprocess
import httpx
import logging
from pathlib import Path

from logging_utils import setup_colored_logging
from config_loader import load_config

logger = setup_colored_logging("manage")


def _kill_server_processes(config):
    import subprocess as sp
    server_dir = Path(__file__).parent
    for srv in config.get("servers", []):
        if srv.get("backend", "local") != "local":
            continue
        script = srv.get("server_script", "")
        port = srv.get("port", 0)
        found = sp.run(
            ["pgrep", "-f", f"{script}.*--port.*{port}"],
            capture_output=True, text=True,
        )
        pids = found.stdout.strip().split("\n") if found.stdout.strip() else []
        for pid_str in pids:
            pid = int(pid_str.strip())
            if pid == os.getpid():
                continue
            logger.warning(f"Killing orphan PID {pid} ({script} port {port})")
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            except PermissionError:
                logger.error(f"  Permission denied for PID {pid}")
            except Exception as e:
                logger.error(f"  Failed to kill PID {pid}: {e}")

    found = sp.run(
        ["pgrep", "-f", "router.py"],
        capture_output=True, text=True,
    )
    pids = found.stdout.strip().split("\n") if found.stdout.strip() else []
    for pid_str in pids:
        pid = int(pid_str.strip())
        if pid == os.getpid():
            continue
        logger.warning(f"Killing orphan router PID {pid}")
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def get_router_url(config=None):
    if config is None:
        config = load_config()
    router_cfg = config.get("router", {})
    return f"http://{router_cfg.get('host', '0.0.0.0')}:{router_cfg.get('port', 8000)}"


def check_health(host, port, timeout=3):
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("status", "unknown")
    except Exception:
        pass
    return "offline"


def start_command(args):
    config = load_config()
    server_dir = Path(__file__).parent
    router_cfg = config.get("router", {})

    router_script = server_dir / "router.py"
    if not router_script.exists():
        logger.error(f"Router script not found: {router_script}")
        return

    cmd = [
        sys.executable, str(router_script),
        "--host", str(router_cfg["host"]),
        "--port", str(router_cfg["port"]),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    logger.info(f"Starting router on {router_cfg['host']}:{router_cfg['port']} ...")
    logger.info(f"Router will manage model servers based on config (startup field).")
    logger.info(f"Use 'manage.py load <name>' / 'manage.py unload <name>' to dynamically manage servers.")
    logger.info(f"Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(cmd, env=env, cwd=str(server_dir))

    logger.info(f"Router PID: {proc.pid}")

    def handle_signal(signum, frame):
        logger.info("\nStopping all services via router shutdown ...")
        try:
            router_url = get_router_url()
            httpx.post(f"{router_url}/shutdown", timeout=5)
        except Exception:
            pass
        time.sleep(3)
        try:
            proc.terminate()
        except Exception:
            pass
        logger.info("All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    proc.wait()
    logger.info("Router process exited.")


def stop_command(args):
    config = load_config()
    router_url = get_router_url(config)
    servers = config.get("servers", [])

    for srv in servers:
        if srv.get("backend", "local") != "local":
            continue
        host = srv["host"]
        port = srv["port"]
        name = srv["name"]
        display = srv.get("display_name", name)
        logger.info(f"[{display}] Sending /shutdown to {host}:{port} ...")
        try:
            httpx.post(f"http://{host}:{port}/shutdown", timeout=5)
            logger.info(f"[{display}] Shutdown signal sent.")
        except Exception:
            logger.warning(f"[{display}] Not reachable, may already be down.")

    time.sleep(2)

    logger.info("Sending /shutdown to router ...")
    try:
        httpx.post(f"{router_url}/shutdown", timeout=5)
    except Exception:
        logger.warning("Router may already be down.")

    time.sleep(3)

    remaining = []
    for srv in servers:
        if srv.get("backend", "local") != "local":
            continue
        name = srv["name"]
        proc = server_processes.get(name) if False else None
        host = srv["host"]
        port = srv["port"]
        try:
            resp = httpx.get(f"http://{host}:{port}/health", timeout=1)
            if resp.status_code == 200:
                remaining.append(f"{srv.get('display_name', name)} (port {port})")
        except Exception:
            pass

    try:
        status_resp = httpx.get(f"{router_url}/status", timeout=2)
    except Exception:
        status_resp = None

    if status_resp and status_resp.status_code == 200:
        data = status_resp.json()
        for name, info in data.get("servers", {}).items():
            if info.get("running"):
                remaining.append(f"{info.get('display_name', name)} (PID unknown)")
    else:
        remaining.append("Router (still running)")

    if remaining:
        logger.warning(f"Still alive: {', '.join(remaining)}")
        logger.info("Attempting to find and kill remaining processes...")
        _kill_server_processes(config)
    else:
        logger.info("All services stopped cleanly.")


def status_command(args):
    config = load_config()
    plain = getattr(args, "plain", False)
    target_name = getattr(args, "name", None)
    router_url = get_router_url(config)
    servers = config.get("servers", [])

    # --name mode: resolve display_name → config name, print single status word
    if target_name:
        # resolve: if user gave display_name (e.g. "qwen3.5"), map to config name ("qwen3-5")
        resolved = target_name
        for srv in servers:
            if srv.get("display_name") == target_name:
                resolved = srv["name"]
                break
        try:
            resp = httpx.get(f"{router_url}/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("servers", {}).get(resolved)
                if info:
                    state = info.get("status", "unknown")
                    if state in ("ready",):
                        print("ready")
                    elif state in ("loading", "starting"):
                        print("loading")
                    elif state in ("offline", "stopped"):
                        print("offline")
                    else:
                        print("offline")
                else:
                    print("offline")
            else:
                print("offline")
        except Exception:
            print("offline")
        return

    try:
        resp = httpx.get(f"{router_url}/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("=" * 60)
            logger.info("Router: ONLINE")
            logger.info("-" * 60)
            for name, info in data.get("servers", {}).items():
                state = info.get("status", "unknown")
                running = info.get("running", False)
                if plain:
                    print(f"{name} {state.upper()} {'YES' if running else 'NO'}")
                    continue
                if state == "ready":
                    status_str = "\033[92mREADY\033[0m"
                elif state == "loading":
                    status_str = "\033[93mLOADING\033[0m"
                elif state == "starting":
                    status_str = "\033[93mSTARTING\033[0m"
                elif state == "error":
                    status_str = "\033[91mERROR\033[0m"
                elif state in ("offline", "stopped"):
                    status_str = "\033[91mOFFLINE\033[0m"
                else:
                    status_str = f"\033[93m{state}\033[0m"
                running_str = "\033[92mYES\033[0m" if running else "\033[91mNO\033[0m"
                logger.info(f"  {info.get('display_name', name):15s} ({name:12s}) [{info.get('type','?'):5s}] {status_str}  process={running_str}")
                if info.get("backend") == "openai":
                    logger.info(f"    Provider: {info.get('provider', 'remote')}")
                else:
                    logger.info(f"    URL: http://{info['host']}:{info['port']}")
            logger.info(f"  Router endpoint: {router_url}/v1")
            logger.info("=" * 60)
        else:
            logger.error(f"Router returned status {resp.status_code}")
    except Exception:
        logger.error("Router is OFFLINE or unreachable")
        logger.info("  Individual server status:")
        for srv in servers:
            if srv.get("backend", "local") != "local":
                continue
            state = check_health(srv["host"], srv["port"])
            logger.info(f"  {srv['display_name']:15s} ({srv['name']:12s}) [{srv.get('type','?'):5s}] {state}")
            logger.info(f"    URL: http://{srv['host']}:{srv['port']}")


def load_command(args):
    config = load_config()
    router_url = get_router_url(config)
    name = args.name

    srv_found = None
    for srv in config.get("servers", []):
        if srv["name"] == name or srv["display_name"] == name:
            srv_found = srv
            break

    if srv_found is None:
        logger.error(f"Server '{name}' not found in config.toml")
        logger.info("Available servers:")
        for srv in config.get("servers", []):
            logger.info(f"  - {srv['name']} (display: {srv['display_name']})")
        return

    logger.info(f"Loading server '{srv_found['name']}' ...")
    try:
        resp = httpx.post(f"{router_url}/v1/load/{srv_found['name']}", timeout=10)
        data = resp.json()
        if resp.status_code == 200 or resp.status_code == 202:
            logger.info(f"  Status: {data.get('status')}, PID: {data.get('pid', '?')}")
            logger.info(f"  Use 'manage.py status' to monitor loading progress.")
        elif resp.status_code == 409:
            logger.warning(f"  Server '{name}' is already running.")
        else:
            logger.error(f"  Error: {data}")
    except Exception as e:
        logger.error(f"  Failed to connect to router: {e}")
        logger.error(f"  Is the router running? Use 'manage.py start' first.")


def unload_command(args):
    config = load_config()
    router_url = get_router_url(config)
    name = args.name

    srv_found = None
    for srv in config.get("servers", []):
        if srv["name"] == name or srv["display_name"] == name:
            srv_found = srv
            break

    if srv_found is None:
        logger.error(f"Server '{name}' not found in config.toml")
        logger.info("Available servers:")
        for srv in config.get("servers", []):
            logger.info(f"  - {srv['name']} (display: {srv['display_name']})")
        return

    logger.info(f"Unloading server '{srv_found['name']}' (calling deinit and terminating process) ...")
    try:
        resp = httpx.post(f"{router_url}/v1/unload/{srv_found['name']}", timeout=15)
        data = resp.json()
        logger.info(f"  Status: {data.get('status')}")
    except Exception as e:
        logger.error(f"  Failed to connect to router: {e}")


def model_list_help():
    """Return the configured model names for CLI help output."""
    try:
        servers = load_config().get("servers", [])
    except (OSError, ValueError) as e:
        return f"Configured models unavailable: {e}"

    if not servers:
        return "Configured models: (none)"

    lines = ["Configured models (name / display name):"]
    for server in servers:
        name = server.get("name", "<unnamed>")
        display_name = server.get("display_name", name)
        lines.append(f"  {name} / {display_name}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Manage OpenAI-compatible model servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s start\n"
            "  %(prog)s status --plain\n"
            "  %(prog)s status --name qwen3-5\n"
            "  %(prog)s load vits-melo-tts\n"
            "  %(prog)s unload qwen3.5\n\n"
            + model_list_help()
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("start", help="Start router (which auto-starts servers with startup=true)")
    subparsers.add_parser("stop", help="Stop all services gracefully")
    subparsers.add_parser("help", help="Show usage, command syntax, and configured models")
    status_parser = subparsers.add_parser("status", help="Check status of all services (including loading state)")
    status_parser.add_argument("--plain", action="store_true", help="Plain text output for scripting")
    status_parser.add_argument("--name", type=str, default=None, help="Check single model: prints 'ready'/'loading'/'offline'")

    load_parser = subparsers.add_parser("load", help="Load (start) a model server dynamically")
    load_parser.add_argument("name", help="Server name or display_name from config.toml")

    unload_parser = subparsers.add_parser("unload", help="Unload (stop) a model server dynamically")
    unload_parser.add_argument("name", help="Server name or display_name from config.toml")

    args = parser.parse_args()

    if args.command == "start":
        start_command(args)
    elif args.command == "stop":
        stop_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "load":
        load_command(args)
    elif args.command == "unload":
        unload_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
