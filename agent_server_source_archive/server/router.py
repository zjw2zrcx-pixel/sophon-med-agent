import sys
import os
import asyncio
import json
import argparse
import subprocess
import threading
import logging
from pathlib import Path

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uvicorn

from logging_utils import setup_colored_logging
from agent_trace import TraceStore
from config_loader import load_config
from prompt_protocol import messages_from_prompt_slots

logger = setup_colored_logging("router")


config = load_config()

server_processes = {}
server_states = {}
managed_pids = []

# Agent observability is intentionally independent from model routing.  The
# Agent process posts bounded execution events here and any number of read-only
# dashboards can subscribe over the same-origin WebSocket.
agent_traces = TraceStore(max_traces=100)
agent_trace_viewers = set()

# Persistent HTTP client with connection pooling — reused across all proxy calls.
_http_client: httpx.AsyncClient | None = None


def _remote_config_state(srv: dict) -> str:
    """Return only configuration failures; reachability is probed asynchronously."""
    if not srv.get("enabled", True):
        return "disabled"
    key_name = str(srv.get("api_key_env", ""))
    return "checking" if key_name and os.environ.get(key_name) else "missing_credentials"


for _srv in config.get("servers", []):
    if _srv.get("backend", "local") == "openai":
        server_states[_srv["name"]] = _remote_config_state(_srv)


async def _probe_remote_server(srv: dict) -> str:
    """Poll an OpenAI-compatible provider instead of inferring online from env.

    A successful authenticated ``GET /models`` proves DNS, TLS, routing and
    provider availability.  We intentionally do not send a completion merely
    to refresh the health badge.
    """
    configured = _remote_config_state(srv)
    if configured != "checking":
        return configured
    try:
        response = await _get_client().get(
            srv["base_url"].rstrip("/") + "/models",
            headers=_remote_headers(srv),
            timeout=httpx.Timeout(8.0, connect=5.0),
        )
        if 200 <= response.status_code < 300:
            return "ready"
        if response.status_code in (401, 403):
            logger.warning("[%s] Provider credential rejected (HTTP %s)", srv["name"], response.status_code)
            return "invalid_credentials"
        logger.warning("[%s] Provider health probe returned HTTP %s", srv["name"], response.status_code)
        return "unavailable"
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("[%s] Provider health probe failed: %s", srv["name"], exc)
        return "unavailable"
    except Exception as exc:
        logger.warning("[%s] Provider health probe error: %s", srv["name"], exc)
        return "unavailable"


def _get_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it on first access."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        )
    return _http_client


def _close_client():
    """Close the shared client, called during shutdown."""
    global _http_client
    if _http_client is not None:
        # Can't await in sync context; schedule in event loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_http_client.aclose())
        except RuntimeError:
            pass
        _http_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.create_task(monitor_servers())
    loop.create_task(start_initial_servers())
    yield
    logger.info("Router shutting down, stopping all servers...")
    await _shutdown_all_servers()
    _close_client()


app = FastAPI(title="OpenAI-Compatible Router", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def start_initial_servers():
    for srv in config.get("servers", []):
        if srv.get("backend", "local") == "openai":
            server_states[srv["name"]] = await _probe_remote_server(srv)
            continue
        if srv.get("backend", "local") == "local" and srv.get("startup", True):
            await _launch_server(srv)
            name = srv["name"]
            logger.info(f"[{name}] Waiting for model to finish loading before starting next ...")
            await _wait_until_ready(name, timeout=600)


async def _launch_server(srv):
    if srv.get("backend", "local") != "local":
        logger.warning("[%s] Remote models are not process-managed", srv.get("name"))
        return None
    name = srv["name"]
    if name in server_processes:
        proc = server_processes[name]
        if proc.poll() is None:
            logger.warning(f"[{name}] Already running with PID {proc.pid}")
            return None

    server_dir = Path(__file__).parent
    script = srv.get("server_script")
    if not script:
        logger.error(f"[{name}] No server_script configured")
        return None

    script_path = server_dir / script
    if not script_path.exists():
        logger.error(f"[{name}] Script {script_path} not found")
        return None

    cmd = [
        sys.executable, str(script_path),
        "--host", str(srv["host"]),
        "--port", str(srv["port"]),
        "--model-path", str(Path(server_dir / srv["model_path"]).resolve()),
        "--config-path", str(Path(server_dir / srv["config_path"]).resolve()),
        "--module-path", str(Path(server_dir / srv["module_path"]).resolve()),
        "--devid", str(srv.get("devid", 0)),
    ]

    if srv.get("type") == "chat":
        cmd.extend(["--video-ratio", str(srv.get("video_ratio", 0.25))])
        cmd.extend(["--max-new-tokens", str(srv.get("max_new_tokens", 512))])
        if srv.get("server_script") == "qwen3_5_history_server.py":
            cmd.extend(["--max-sessions", str(srv.get("max_sessions", 4))])
            cmd.extend([
                "--max-snapshot-bytes",
                str(srv.get("max_snapshot_bytes", 1024 ** 3)),
            ])
    elif srv.get("type") == "audio":
        lang = srv.get("language")
        if lang is not None:
            cmd.extend(["--language", str(lang)])
    elif srv.get("type") == "tts" and "max_input_tokens" in srv:
        cmd.extend(["--max-input-tokens", str(srv["max_input_tokens"])])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MODULE_PATH"] = str(Path(server_dir / srv["module_path"]).resolve())

    model_dir = str(Path(server_dir / srv["module_path"]).resolve().parent)
    for key in ["PYTHONPATH", "PYTHONHOME"]:
        if key in env:
            del env[key]

    server_states[name] = "starting"
    logger.info(f"[{name}] Starting server on {srv['host']}:{srv['port']} ...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(server_dir),
        env=env,
    )
    server_processes[name] = proc
    server_states[name] = "starting"
    managed_pids.append(proc.pid)
    logger.info(f"[{name}] PID: {proc.pid}")

    t = threading.Thread(target=_stream_server_output, args=(proc, name), daemon=True)
    t.start()

    return proc


async def _wait_until_ready(name, timeout=600, poll_interval=3):
    srv = _get_srv_info(name)
    if srv is None:
        return
    host = srv["host"]
    port = srv["port"]
    elapsed = 0
    last_logged_state = None
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        proc = server_processes.get(name)
        if proc is not None and proc.poll() is not None:
            logger.error(f"[{name}] Process exited during loading, will not continue.")
            return
        client = _get_client()
        try:
            resp = await client.get(f"http://{host}:{port}/health")
            data = resp.json()
            state = data.get("status", "unknown")
            if state == "ready":
                logger.info(f"[{name}] Model loaded and ready.")
                return
            elif state == "error":
                logger.error(f"[{name}] Model failed to load.")
                return
            elif state != last_logged_state:
                logger.info(f"[{name}] Loading... (state: {state})")
                last_logged_state = state
        except Exception:
            if last_logged_state != "not_responding":
                logger.info(f"[{name}] Starting up, waiting for health endpoint...")
                last_logged_state = "not_responding"
    logger.warning(f"[{name}] Timed out waiting for model to load after {timeout}s")


def _stream_server_output(proc, name):
    try:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n\r")
            if text:
                logger.info(f"[{name}] {text}")
    except Exception as e:
        logger.error(f"[{name}] Output stream error: {e}")
    finally:
        rc = proc.wait()
        server_states[name] = "stopped"
        if rc < 0:
            sig = -rc
            sig_names = {11: "SIGSEGV", 6: "SIGABRT", 9: "SIGKILL", 15: "SIGTERM"}
            sig_name = sig_names.get(sig, f"signal {sig}")
            logger.error(f"[{name}] Process killed by {sig_name} (exit code {rc})")
        else:
            logger.warning(f"[{name}] Process exited with code {rc}")


async def monitor_servers():
    while True:
        await asyncio.sleep(10)  # 10s poll — responsive enough, far less churn
        client = _get_client()
        for srv in config.get("servers", []):
            name = srv["name"]
            if srv.get("backend", "local") == "openai":
                server_states[name] = await _probe_remote_server(srv)
                continue
            host = srv["host"]
            port = srv["port"]
            proc = server_processes.get(name)
            proc_dead = proc is not None and proc.poll() is not None

            if proc_dead:
                new_state = "stopped"
            else:
                try:
                    resp = await client.get(f"http://{host}:{port}/health")
                    data = resp.json()
                    new_state = data.get("status", "unknown")
                except Exception:
                    if proc is not None:
                        new_state = "starting" if proc.poll() is None else "stopped"
                    else:
                        new_state = "offline"

            old_state = server_states.get(name, "unknown")
            if new_state != old_state:
                if new_state == "ready":
                    logger.info(f"[{name}] Model is ready")
                elif new_state == "stopped":
                    logger.warning(f"[{name}] Process stopped")
                server_states[name] = new_state


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/frontend")
async def serve_frontend():
    """Retired: the unified debugging UI is hosted by Voice Agent on 8766."""
    return HTMLResponse(
        content=(
            "<!doctype html><meta charset='utf-8'><title>页面已迁移</title>"
            "<h1>此页面已废弃</h1>"
            "<p>请使用 Voice Agent 调试台："
            "<a href='http://192.168.10.2:8766/'>http://192.168.10.2:8766/</a></p>"
        ),
        status_code=410,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/v1/agent/traces")
async def get_agent_traces(limit: int = 50):
    """Return a bounded snapshot so a refreshed dashboard can recover."""
    return {"traces": agent_traces.snapshot(limit=limit)}


async def _broadcast_agent_event(event):
    if not agent_trace_viewers:
        return
    message = {"type": "agent_event", "event": event}
    stale = []
    for viewer in tuple(agent_trace_viewers):
        try:
            await viewer.send_json(message)
        except Exception:
            stale.append(viewer)
    for viewer in stale:
        agent_trace_viewers.discard(viewer)


@app.post("/v1/agent/events")
async def ingest_agent_event(request: Request):
    """Receive one observable execution event from the local Agent process."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 256_000:
                return JSONResponse(
                    {"error": {"message": "Agent event is too large"}},
                    status_code=413,
                )
        except ValueError:
            pass
    try:
        event = agent_traces.add(await request.json())
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status_code=400,
        )
    await _broadcast_agent_event(event)
    return {"status": "ok"}


@app.websocket("/v1/agent/events/ws")
async def agent_event_stream(websocket: WebSocket):
    """Push snapshots and live Agent events to a read-only dashboard."""
    await websocket.accept()
    agent_trace_viewers.add(websocket)
    try:
        await websocket.send_json({
            "type": "snapshot",
            "traces": agent_traces.snapshot(limit=50),
        })
        # Waiting for client frames lets Starlette observe disconnects.  The
        # browser normally sends nothing; server-side broadcasts happen above.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        agent_trace_viewers.discard(websocket)


@app.get("/status")
async def get_status():
    result = {}
    for srv in config.get("servers", []):
        name = srv["name"]
        backend = srv.get("backend", "local")
        state = server_states.get(name, "checking") if backend == "openai" else server_states.get(name, "unknown")
        result[name] = {
            "display_name": srv.get("display_name", name),
            "type": srv.get("type", "?"),
            "status": state,
            "backend": backend,
            "provider": srv.get("provider", "local"),
            "available": state == "ready",
            "running": (
                False if backend == "openai"
                else name in server_processes and server_processes[name].poll() is None
            ),
            "host": srv.get("host", ""),
            "port": srv.get("port", 0),
        }
    return {"router": "ok", "servers": result}


@app.get("/v1/models")
async def list_models():
    models = []
    for srv in config.get("servers", []):
        name = srv["name"]
        backend = srv.get("backend", "local")
        state = server_states.get(name, "checking") if backend == "openai" else server_states.get(name, "unknown")
        if state == "ready":
            status = "online"
        elif state in ("starting", "loading"):
            status = "loading"
        else:
            status = "offline"
        models.append({
            "id": srv["display_name"],
            "object": "model",
            "owned_by": srv.get("provider", "local"),
            "status": status,
            "type": srv.get("type", "chat"),
            "backend": backend,
        })
    return {"object": "list", "data": models}


@app.post("/v1/load/{server_name}")
async def load_server(server_name: str):
    srv = _get_srv_info(server_name)
    if srv is None:
        return JSONResponse(
            {"error": {"message": f"Server '{server_name}' not found in config", "type": "invalid_request_error"}},
            status_code=404,
        )
    if srv.get("backend", "local") == "openai":
        state = await _probe_remote_server(srv)
        server_states[server_name] = state
        code = 200 if state == "ready" else 503
        return JSONResponse({"status": state, "name": server_name}, status_code=code)

    existing_proc = server_processes.get(server_name)
    if existing_proc is not None and existing_proc.poll() is None:
        return JSONResponse(
            {"error": {"message": f"Server '{server_name}' is already running (PID {existing_proc.pid})", "type": "conflict"}},
            status_code=409,
        )

    exclusive_group = srv.get("exclusive_group")
    if exclusive_group:
        for other in config.get("servers", []):
            if other["name"] == server_name:
                continue
            if other.get("exclusive_group") != exclusive_group:
                continue
            other_proc = server_processes.get(other["name"])
            if other_proc is not None and other_proc.poll() is None:
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"Server '{server_name}' conflicts with running server "
                                f"'{other['name']}' in exclusive group '{exclusive_group}'. "
                                f"Unload '{other['name']}' first."
                            ),
                            "type": "conflict",
                        }
                    },
                    status_code=409,
                )

    result = await _launch_server(srv)
    if result is None:
        return JSONResponse(
            {"error": {"message": f"Failed to start server '{server_name}'", "type": "server_error"}},
            status_code=500,
        )

    return JSONResponse({
        "status": "starting",
        "name": server_name,
        "pid": result.pid,
        "message": f"Server '{server_name}' is starting, use /status to monitor progress",
    })


@app.post("/v1/unload/{server_name}")
async def unload_server(server_name: str):
    srv = _get_srv_info(server_name)
    if srv is None:
        return JSONResponse(
            {"error": {"message": f"Server '{server_name}' not found in config", "type": "invalid_request_error"}},
            status_code=404,
        )
    if srv.get("backend", "local") == "openai":
        return JSONResponse(
            {"error": {"message": "Remote models cannot be unloaded", "type": "invalid_request_error"}},
            status_code=409,
        )

    proc = server_processes.get(server_name)
    if proc is None or proc.poll() is not None:
        server_states[server_name] = "offline"
        return JSONResponse({"status": "already_stopped", "name": server_name})

    host = srv["host"]
    port = srv["port"]

    logger.info(f"[{server_name}] Sending /shutdown to {host}:{port} ...")
    client = _get_client()
    try:
        await client.post(f"http://{host}:{port}/shutdown")
    except Exception:
        pass

    await asyncio.sleep(2)

    if proc.poll() is None:
        logger.info(f"[{server_name}] Process still alive, terminating PID {proc.pid} ...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    server_states[server_name] = "offline"
    del server_processes[server_name]

    logger.info(f"[{server_name}] Unloaded successfully.")
    return JSONResponse({"status": "unloaded", "name": server_name})


@app.post("/v1/responses")
async def responses_fallback(request: Request):
    return JSONResponse(
        {"error": {"message": "Responses API is not supported by this server", "type": "invalid_request_error"}},
        status_code=404,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )
    model_name = body.get("model", "")

    target = None
    for srv in config.get("servers", []):
        if srv.get("type") == "chat" and (
            srv["name"] == model_name or srv["display_name"] == model_name
        ):
            target = srv
            break

    if target is None:
        return JSONResponse(
            {"error": {"message": f"Model '{model_name}' not found. Available: " + ", ".join(s['display_name'] for s in config.get('servers', []) if s.get('type') == 'chat'), "type": "invalid_request_error"}},
            status_code=400,
        )

    target_state = server_states.get(target["name"], "unknown")
    if target.get("backend", "local") == "openai":
        # The background poll owns normal liveness updates.  Only an initial
        # request made before its first pass blocks for a direct probe.
        if target_state == "checking":
            target_state = await _probe_remote_server(target)
            server_states[target["name"]] = target_state
    if target_state != "ready":
        return JSONResponse(
            {"error": {"message": f"Model '{target['display_name']}' is not ready, current state: {target_state}", "type": "server_error"}},
            status_code=503,
        )

    if target.get("backend", "local") == "openai":
        return await _remote_chat_completion(request, body, target)

    if body.get("prompt_slots") is not None and target.get("server_script") != "qwen3_5_history_server.py":
        try:
            body = dict(body)
            body["messages"] = messages_from_prompt_slots(body.pop("prompt_slots"))
        except ValueError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )

    upstream_url = f"http://{target['host']}:{target['port']}/v1/chat/completions"
    stream = body.get("stream", False)

    headers = {}
    for key, value in request.headers.items():
        if (
            key.lower().startswith("x-session")
            or key.lower() in {"x-clear-history", "x-prompt-slots"}
        ):
            headers[key] = value

    client = _get_client()
    if stream:
        return StreamingResponse(
            _stream_proxy(client, upstream_url, body, headers),
            media_type="text/event-stream",
        )
    try:
        resp = await client.post(upstream_url, json=body, headers=headers)
        try:
            resp_body = resp.json()
        except Exception:
            resp_text = resp.text or ""
            if resp.status_code >= 400:
                return JSONResponse(
                    {"error": {"message": f"Upstream error {resp.status_code}: {resp_text[:500]}", "type": "server_error"}},
                    status_code=resp.status_code,
                )
            return JSONResponse(
                {"error": {"message": f"Upstream returned non-JSON: {resp_text[:500]}", "type": "server_error"}},
                status_code=502,
            )
        return JSONResponse(content=resp_body, status_code=resp.status_code)
    except httpx.ConnectError:
        return JSONResponse(
            {"error": {"message": f"Model server at {upstream_url} is unreachable", "type": "server_error"}},
            status_code=502,
        )
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=502,
        )


def _remote_request_body(body: dict, target: dict) -> dict:
    upstream = dict(body)
    prompt_slots = upstream.pop("prompt_slots", None)
    upstream.pop("benchmark", None)
    if prompt_slots is not None:
        slots = dict(prompt_slots)
        had_image = slots.pop("image", None) is not None
        messages = messages_from_prompt_slots(slots)
        if had_image:
            messages[-1]["content"] += "\n<system-note>当前在线模型未接收本轮图像，请勿假装看到了图像。</system-note>"
        upstream["messages"] = messages
    if not isinstance(upstream.get("messages"), list) or not upstream["messages"]:
        raise ValueError("messages or prompt_slots is required")
    upstream["model"] = target["upstream_model"]
    upstream["thinking"] = {"type": target.get("thinking", "enabled")}
    upstream["reasoning_effort"] = target.get("reasoning_effort", "high")
    return upstream


def _remote_url(target: dict) -> str:
    return target["base_url"].rstrip("/") + "/chat/completions"


def _remote_headers(target: dict) -> dict:
    key = os.environ.get(str(target.get("api_key_env", "")), "")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("retry-after", "")
        try:
            return min(30.0, max(0.0, float(raw)))
        except ValueError:
            pass
    return min(4.0, 0.5 * (2 ** attempt))


async def _remote_chat_completion(request: Request, body: dict, target: dict):
    try:
        upstream_body = _remote_request_body(body, target)
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status_code=400,
        )
    if upstream_body.get("stream", False):
        return StreamingResponse(
            _remote_stream(target, upstream_body), media_type="text/event-stream"
        )

    client = _get_client()
    retries = max(0, int(target.get("max_retries", 2)))
    timeout = httpx.Timeout(
        float(target.get("read_timeout", 300.0)),
        connect=float(target.get("connect_timeout", 10.0)),
    )
    last_error = "remote request failed"
    for attempt in range(retries + 1):
        response = None
        try:
            response = await client.post(
                _remote_url(target), json=upstream_body,
                headers=_remote_headers(target), timeout=timeout,
            )
            if response.status_code != 429 and response.status_code < 500:
                try:
                    return JSONResponse(response.json(), status_code=response.status_code)
                except Exception:
                    return JSONResponse(
                        {"error": {"message": "Remote provider returned non-JSON", "type": "server_error"}},
                        status_code=502,
                    )
            last_error = f"remote HTTP {response.status_code}"
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            await asyncio.sleep(_retry_delay(response, attempt))
    return JSONResponse(
        {"error": {"message": last_error, "type": "server_error"}}, status_code=502
    )


async def _remote_stream(target: dict, body: dict):
    client = _get_client()
    retries = max(0, int(target.get("max_retries", 2)))
    timeout = httpx.Timeout(
        float(target.get("read_timeout", 300.0)),
        connect=float(target.get("connect_timeout", 10.0)),
    )
    for attempt in range(retries + 1):
        response = None
        try:
            async with client.stream(
                "POST", _remote_url(target), json=body,
                headers=_remote_headers(target), timeout=timeout,
            ) as response:
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < retries:
                        await response.aread()
                        await asyncio.sleep(_retry_delay(response, attempt))
                        continue
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:500]
                    payload = {"error": {"message": detail, "type": "server_error"}}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return
                async for chunk in response.aiter_text():
                    yield chunk
                return
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt < retries:
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            payload = {"error": {"message": str(exc), "type": "server_error"}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )
    model_name = body.get("model", "")
    target = _find_server(model_name, "embedding")
    if target is None:
        available = ", ".join(
            srv["display_name"] for srv in config.get("servers", []) if srv.get("type") == "embedding"
        )
        return JSONResponse(
            {"error": {"message": f"Embedding model '{model_name}' not found. Available: {available}", "type": "invalid_request_error"}},
            status_code=400,
        )
    target_state = server_states.get(target["name"], "unknown")
    if target_state != "ready":
        return JSONResponse(
            {"error": {"message": f"Model '{target['display_name']}' is not ready, current state: {target_state}", "type": "server_error"}},
            status_code=503,
        )
    upstream_url = f"http://{target['host']}:{target['port']}/v1/embeddings"
    try:
        response = await _get_client().post(upstream_url, json=body)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.ConnectError:
        return JSONResponse(
            {"error": {"message": f"Model server at {upstream_url} is unreachable", "type": "server_error"}},
            status_code=502,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "server_error"}},
            status_code=502,
        )


async def _stream_proxy(client, url, body, headers):
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for chunk in resp.aiter_text():
                yield chunk
    except httpx.ConnectError:
        error_data = {"error": {"message": f"Model server at {url} is unreachable", "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"


def _find_server(model_name, server_type):
    """Resolve either the internal or public model name for non-OpenAI APIs."""
    return next((srv for srv in config.get("servers", [])
                 if srv.get("type") == server_type
                 and (srv["name"] == model_name or srv["display_name"] == model_name)), None)


async def _json_model_proxy(request: Request, server_type: str, upstream_path: str):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body"}}, status_code=400)
    target = _find_server(body.get("model", ""), server_type)
    if target is None:
        return JSONResponse({"error": {"message": f"Unknown {server_type} model"}}, status_code=400)
    if server_states.get(target["name"]) != "ready":
        return JSONResponse({"error": {"message": f"Model '{target['display_name']}' is not ready"}}, status_code=503)
    try:
        response = await _get_client().post(
            f"http://{target['host']}:{target['port']}{upstream_path}", json=body
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=502)


@app.post("/v1/audio/keywords")
async def audio_keywords(request: Request):
    """Run the configured KWS backend on a base64 PCM/WAV audio block."""
    return await _json_model_proxy(request, "kws", "/v1/audio/keywords")


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """Synthesize speech with the configured TTS backend; response contains WAV base64."""
    return await _json_model_proxy(request, "tts", "/v1/audio/speech")


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    form = await request.form()
    model_name = form.get("model", "")

    target = None
    for srv in config.get("servers", []):
        if srv.get("type") == "audio" and (
            srv["name"] == model_name or srv["display_name"] == model_name
        ):
            target = srv
            break

    if target is None:
        return JSONResponse(
            {"error": {"message": f"Audio model '{model_name}' not found. Available: " + ", ".join(s['display_name'] for s in config.get('servers', []) if s.get('type') == 'audio'), "type": "invalid_request_error"}},
            status_code=400,
        )

    target_state = server_states.get(target["name"], "unknown")
    if target_state != "ready":
        return JSONResponse(
            {"error": {"message": f"Model '{target['display_name']}' is not ready, current state: {target_state}", "type": "server_error"}},
            status_code=503,
        )

    upstream_url = f"http://{target['host']}:{target['port']}/v1/audio/transcriptions"

    files = {}
    data = {}
    file_obj = form.get("file")
    if file_obj:
        content = await file_obj.read()
        files["file"] = (file_obj.filename, content, file_obj.content_type)
    language = form.get("language")
    if language:
        data["language"] = str(language)
    if model_name:
        data["model"] = str(model_name)

    client = _get_client()
    try:
        resp = await client.post(upstream_url, files=files, data=data)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError:
        return JSONResponse(
            {"error": {"message": f"Audio server at {upstream_url} is unreachable", "type": "server_error"}},
            status_code=502,
        )
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=502,
        )

@app.post("/v1/audio/transcriptions/stream")
async def audio_transcriptions_stream(request: Request):
    """Proxy streaming ASR chunks to the audio backend."""
    body = await request.json()
    model_name = body.get("model", "")

    target = None
    for srv in config.get("servers", []):
        if srv.get("type") == "audio" and (
            srv["name"] == model_name or srv["display_name"] == model_name
        ):
            target = srv
            break

    if target is None:
        # Default: use the first available audio server
        for srv in config.get("servers", []):
            if srv.get("type") == "audio":
                target = srv
                break

    if target is None:
        return JSONResponse(
            {"error": {"message": "No audio model available", "type": "invalid_request_error"}},
            status_code=400,
        )

    target_state = server_states.get(target["name"], "unknown")
    if target_state != "ready":
        return JSONResponse(
            {"error": {"message": f"Model '{target['display_name']}' is not ready, current state: {target_state}", "type": "server_error"}},
            status_code=503,
        )

    upstream_url = f"http://{target['host']}:{target['port']}/v1/audio/transcriptions/stream"

    client = _get_client()
    try:
        resp = await client.post(upstream_url, json=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError:
        return JSONResponse(
            {"error": {"message": f"Audio server at {upstream_url} is unreachable", "type": "server_error"}},
            status_code=502,
        )
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=502,
        )

@app.post("/shutdown")
async def router_shutdown():
    logger.info("Router received shutdown signal, stopping all servers...")
    await _shutdown_all_servers()
    asyncio.get_event_loop().call_later(3.0, lambda: os._exit(0))
    return JSONResponse({"status": "shutting_down"})


def _get_srv_info(name):
    for srv in config.get("servers", []):
        if srv["name"] == name or srv.get("display_name") == name:
            return srv
    return None


async def _shutdown_all_servers():
    client = _get_client()
    for name, proc in list(server_processes.items()):
        srv_info = _get_srv_info(name)
        if srv_info:
            logger.info(f"[{name}] Sending /shutdown to {srv_info['host']}:{srv_info['port']}...")
            try:
                await client.post(f"http://{srv_info['host']}:{srv_info['port']}/shutdown")
            except Exception:
                pass
        logger.info(f"[{name}] Terminating PID {proc.pid}...")
        try:
            proc.terminate()
        except Exception:
            pass

    import time as _t
    _t.sleep(3)

    for pid in managed_pids:
        try:
            os.kill(pid, 9)
            logger.warning(f"Force killed orphan PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()

    host = args.host or config["router"]["host"]
    port = args.port or config["router"]["port"]
    uvicorn.run(app, host=host, port=port, log_config={
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": "%(asctime)s [%(levelname)s] %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"}},
        "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stderr"}},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    })
