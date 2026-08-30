#!/usr/bin/env python3
"""
Voice Agent - standalone headless process that mediates between a remote
microphone/speaker device and the model servers (KWS + ASR + LLM).

Start:
    python -m agents.Headless.voice_agent --port 8766

State machine per client:
    LISTENING -> (hotword_hit) -> CAPTURING -> (silence) -> PROCESSING -> LISTENING
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import uuid
import wave
from array import array
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

_AGENTS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))

from agents.agent import Agent, AgentConfig
from agents.Headless.manager import HeadlessManager
from agents.model_selection import select_model
from agents.RemoteAudio import (
    PROTOCOL_VERSION, ProtocolError, RemoteAudioError, RemoteAudioOperations,
)
from agents.RemoteAudio.protocol import message as protocol_message, validate_version

logger = logging.getLogger("voice-agent")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence noisy HTTP libraries
for _noisy in ("httpx", "httpcore", "asyncio", "uvicorn"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLE_WIDTH = 2
AUDIO_PREROLL_SECONDS = 2.0
MAX_CAPTURE_AUDIO_SECONDS = 10.0
AUDIO_PREROLL_BYTES = int(
    AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH * AUDIO_PREROLL_SECONDS
)
# The deployed Qwen3-ASR bmodel/runtime combination is stable for five static
# one-second audio blocks. Six or more blocks reproduce a device-memory copy
# error, so only the final command tail is sent to Qwen3-ASR.
QWEN3_ASR_SAFE_AUDIO_SECONDS = 5.0
QWEN3_ASR_SAFE_AUDIO_BYTES = int(
    AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH * QWEN3_ASR_SAFE_AUDIO_SECONDS
)
QWEN3_ASR_SEGMENT_OVERLAP_SECONDS = 0.5
QWEN3_ASR_SEGMENT_OVERLAP_BYTES = int(
    AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH * QWEN3_ASR_SEGMENT_OVERLAP_SECONDS
)
FAILED_CAPTURE_PATH = Path(
    os.environ.get(
        "VOICE_AGENT_FAILED_CAPTURE_PATH",
        "/tmp/voice-agent-last-failed-capture.wav",
    )
)

ASR_DISCLAIMER = (
    "[语音识别结果，可能含有不准确信息，请结合上下文尝试修正与辨别]\n"
)


def _pcm_rms(pcm: bytes) -> int:
    """Return the RMS level of little-endian signed 16-bit mono PCM."""
    if len(pcm) < AUDIO_SAMPLE_WIDTH:
        return 0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % AUDIO_SAMPLE_WIDTH)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0
    return int((sum(int(value) * int(value) for value in samples) / len(samples)) ** 0.5)


def _clean_qwen_asr_text(text: str) -> str:
    """Remove Qwen3-ASR protocol tokens that are not spoken content."""
    value = str(text or "").strip()
    if "<asr_text>" in value:
        value = value.rsplit("<asr_text>", 1)[-1]
    value = re.sub(
        r"^\s*language\s+[A-Za-z][A-Za-z0-9_-]*\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def _has_meaningful_asr_text(text: str) -> bool:
    """Reject empty/control-only/punctuation-only ASR output."""
    return bool(re.search(r"[A-Za-z0-9\u3400-\u9fff]", str(text or "")))


def _limit_pcm_tail(pcm: bytes, max_bytes: int = QWEN3_ASR_SAFE_AUDIO_BYTES) -> bytes:
    """Limit PCM to a safe Qwen window while retaining the spoken command tail."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    max_bytes -= max_bytes % AUDIO_SAMPLE_WIDTH
    if len(pcm) <= max_bytes:
        return pcm
    return pcm[-max_bytes:]


class VoiceState(Enum):
    LISTENING = "listening"
    CAPTURING = "capturing"
    PROCESSING = "processing"


@dataclass
class VoiceClient:
    """Per-browser-connection state."""
    ws: WebSocket
    headless_session_id: str
    state: VoiceState = VoiceState.LISTENING
    # Sherpa KWS is stateful across PCM chunks.  This must remain stable for
    # the lifetime of a browser voice session, otherwise the KWS service
    # creates a fresh decoder/cache for every chunk and can never match a
    # multi-chunk wake phrase.
    kws_session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    asr_client: Optional[httpx.AsyncClient] = None
    capture_peak_rms: int = 0
    capture_chunk_count: int = 0
    audio_preroll: bytearray = field(default_factory=bytearray)
    capture_audio: bytearray = field(default_factory=bytearray)
    last_audio_sequence: Optional[int] = None
    monitor_stream_id: str = ""
    hotword_hit_word: str = ""
    hotwords: list = field(default_factory=list)
    pending_image: Optional[str] = None
    llm_task: Optional[asyncio.Task] = None
    record_task: Optional[asyncio.Task] = None
    remote_audio: Optional[RemoteAudioOperations] = None
    protocol_ready: bool = False
    device_id: str = ""
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    closed: bool = False
    resetting: bool = False
    chunk_count: int = 0
    cooldown_until: float = 0.0
    processing_started_at: float = 0.0
    pending_input_metadata: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class VoiceAgentServer:
    """FastAPI + WebSocket server that manages voice interaction clients."""

    def __init__(self, host="0.0.0.0", port=8766, router_url="http://127.0.0.1:8000",
                 model="qwen3.5-4b-history", trajectory_dir="/data/structure/trajectories",
                 trajectory_enabled=True):
        self.host = host
        self.port = port
        self.router_url = router_url.rstrip("/")
        self.model = model
        self.trajectory_dir = trajectory_dir
        self.trajectory_enabled = trajectory_enabled
        self.clients: Dict[str, VoiceClient] = {}
        self._agent: Optional[Agent] = None
        self._headless: Optional[HeadlessManager] = None
        self._frontend_html: str = ""
        self._frontend_errors: list[dict] = []

    async def _generate_tts(self, text: str) -> str:
        """Generate audio exclusively through the configured Router VITS model."""
        if not text.strip():
            return ""
        vits_url = f"{self.router_url}/v1/audio/speech"
        last_detail = "VITS 未返回音频"
        # VITS can occasionally acknowledge a request before its audio payload
        # is ready.  Retry only transport/empty-audio cases; do not conceal a
        # persistent model fault behind another TTS engine.
        for attempt in range(1, 3):
            try:
                started = time.monotonic()
                async with httpx.AsyncClient(timeout=60.0) as tts_client:
                    response = await tts_client.post(vits_url, json={
                        "model": "vits-melo-tts", "input": text,
                    })
                if response.status_code == 200:
                    payload = response.json()
                    audio = str(payload.get("audio", ""))
                    if audio:
                        logger.info("PERF: vits_tts=%.3fs attempt=%d",
                                    time.monotonic() - started, attempt)
                        return audio
                    logger.warning("VITS returned HTTP 200 with empty audio (attempt %d)", attempt)
                    last_detail = "VITS 返回了空 audio"
                else:
                    try:
                        detail = str(response.json().get("error", {}).get("message", "")).strip()
                    except ValueError:
                        detail = ""
                    last_detail = detail or f"HTTP {response.status_code}"
                    logger.error("VITS unavailable: %s (attempt %d)", last_detail, attempt)
            except Exception as exc:
                last_detail = str(exc)
                logger.error("VITS request failed (attempt %d): %s", attempt, exc)
            if attempt == 1:
                await asyncio.sleep(0.25)
        raise RemoteAudioError(
            "TTS_UPSTREAM_FAILED", f"VITS 合成失败：{last_detail}", retryable=True
        )

    def _init_agent(self):
        config = AgentConfig(
            server_url=self.router_url,
            model_name=self.model,
            trajectory_dir=self.trajectory_dir,
            trajectory_enabled=self.trajectory_enabled,
        )
        self._agent = Agent(config)
        self._agent.initialize()
        self._headless = HeadlessManager(self._agent)
        self._tts = None  # sherpa-onnx TTS engine
        logger.info("Agent + HeadlessManager initialised")

    def _load_frontend(self):
        frontend_path = Path(__file__).with_name("agent_debug.html")
        self._frontend_html = frontend_path.read_text(encoding="utf-8")

    @staticmethod
    def _debug_error(message: str, status_code: int = 400):
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request_error"}},
            status_code=status_code,
        )

    def _debug_result(self, session_id: str, result) -> dict:
        """Serialize the public, inspectable part of an Agent loop result."""
        commands = []
        for command, command_result in result.commands:
            commands.append({
                "name": command.name,
                "type": command.type,
                "params": command.params,
                "success": command_result.success,
                "result_type": command.type,
                "data": command_result.data,
                "error": command_result.error,
                "duration_ms": command_result.duration_ms,
                "diagnostics": command_result.diagnostics,
            })
        return {
            "session_id": session_id,
            "text": result.text,
            "commands": commands,
            "metrics": result.metrics,
            "turn_end_reason": result.turn_end_reason,
            "session_ended": result.session_ended,
        }

    async def start(self):
        self._init_agent()
        self._load_frontend()

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok", "clients": len(self.clients), "model": self.model}

        @app.get("/")
        async def serve_frontend():
            return HTMLResponse(
                content=self._frontend_html,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/debug")
        async def serve_debug_frontend():
            return await serve_frontend()

        @app.get("/v1/debug/config")
        async def debug_config():
            return {
                "agent_model": self.model,
                "router_url": self.router_url,
                "websocket_protocol": PROTOCOL_VERSION,
            }

        @app.post("/v1/debug/client-errors")
        async def record_debug_client_error(request: Request):
            """Small, bounded browser error sink for remote frontend diagnosis."""
            try:
                payload = await request.json()
            except Exception:
                return self._debug_error("Invalid JSON body")
            if not isinstance(payload, dict):
                return self._debug_error("error payload must be an object")
            item = {
                "timestamp": time.time(),
                "message": str(payload.get("message", ""))[:2000],
                "source": str(payload.get("source", ""))[:500],
                "line": payload.get("line", ""),
                "column": payload.get("column", ""),
            }
            self._frontend_errors.append(item)
            del self._frontend_errors[:-30]
            logger.warning("Browser debug UI error: %s", item)
            return {"status": "recorded"}

        @app.get("/v1/debug/client-errors")
        async def get_debug_client_errors():
            return {"data": list(self._frontend_errors)}

        @app.get("/v1/agent/sessions")
        async def list_agent_sessions():
            return {"data": [item.to_dict() for item in self._headless.list_sessions()]}

        @app.post("/v1/agent/sessions")
        async def create_agent_session(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            session_id = str(body.get("id", "")).strip()
            tags = body.get("tags", ["web-debug"])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                return self._debug_error("tags must be an array of strings")
            if len(session_id) > 64 or len(tags) > 8:
                return self._debug_error("session id or tags are too long")
            try:
                created = self._headless.create_session(tags=tags, session_id=session_id or None)
            except ValueError as exc:
                return self._debug_error(str(exc), 409)
            return {"id": created}

        @app.delete("/v1/agent/sessions/{session_id}")
        async def delete_agent_session(session_id: str):
            return {"deleted": self._headless.delete_session(session_id)}

        @app.get("/v1/agent/sessions/{session_id}/history")
        async def agent_session_history(session_id: str, tail: int = 50):
            try:
                messages = self._headless.get_history(session_id, tail=max(1, min(tail, 200)))
            except KeyError as exc:
                return self._debug_error(str(exc), 404)
            return {"data": [{
                "role": message.role,
                "content": message.content,
                "has_image": message.image is not None,
                "source": message.metadata.get("source", ""),
            } for message in messages if message.role != "system"]}

        @app.post("/v1/agent/prompt")
        async def submit_agent_prompt(request: Request):
            try:
                body = await request.json()
            except Exception:
                return self._debug_error("Invalid JSON body")
            session_id = str(body.get("session_id", "")).strip()
            text = str(body.get("text", "")).strip()
            image = body.get("image")
            if not session_id or not text:
                return self._debug_error("session_id and text are required")
            if len(text) > 20_000:
                return self._debug_error("text is too long")
            if image is not None and (not isinstance(image, str) or len(image) > 12_000_000):
                return self._debug_error("image must be a base64 data URL smaller than 12 MB")
            try:
                result = await self._headless.submit(session_id, text, image=image)
            except KeyError as exc:
                return self._debug_error(str(exc), 404)
            except Exception as exc:
                logger.exception("Web debug Agent request failed")
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "agent_error"}}, status_code=500
                )
            return self._debug_result(session_id, result)

        async def proxy_router(path: str, request: Request):
            """Same-origin bridge for browser debugging when Router is bound to localhost."""
            url = f"{self.router_url}{path}"
            try:
                content = await request.body()
                headers = {"content-type": request.headers.get("content-type", "application/json")}
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    response = await client.request(request.method, url, content=content, headers=headers)
                return JSONResponse(response.json(), status_code=response.status_code)
            except httpx.HTTPError as exc:
                return JSONResponse({"error": {"message": str(exc), "type": "router_unreachable"}}, status_code=502)
            except ValueError:
                return JSONResponse({"error": {"message": "Router returned non-JSON", "type": "server_error"}}, status_code=502)

        @app.api_route("/v1/debug/router/{path:path}", methods=["GET", "POST"])
        async def debug_router_proxy(path: str, request: Request):
            allowed_paths = {
                "status", "v1/models", "v1/chat/completions", "v1/audio/transcriptions",
                "v1/audio/speech", "v1/agent/traces",
            }
            normalized = path.strip("/")
            if normalized not in allowed_paths:
                return self._debug_error("This Router endpoint is not available in the debug bridge", 403)
            return await proxy_router("/" + normalized, request)

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            client = await self._on_connect(ws)
            try:
                while True:
                    raw = await ws.receive_text()
                    client.last_activity = time.time()
                    await self._handle_message(client, raw)
            except WebSocketDisconnect:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"WS error: {type(e).__name__}: {e}")
            finally:
                await self._on_disconnect(client)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            ws="wsproto",
        )
        server = uvicorn.Server(config)
        logger.info(f"Voice Agent listening on ws://{self.host}:{self.port}/ws")
        await server.serve()

    async def shutdown(self):
        for cid in list(self.clients.keys()):
            await self._on_disconnect(self.clients[cid])
        if self._agent:
            await self._agent.shutdown()

    async def _on_connect(self, ws):
        hs_id = self._headless.create_session()
        client = VoiceClient(
            ws=ws,
            headless_session_id=hs_id,
            asr_client=httpx.AsyncClient(timeout=httpx.Timeout(15.0)),
        )
        client.hotwords = ["小麦"]  # default hotword

        async def send_remote(payload: dict):
            await self._send(client, payload)

        client.remote_audio = RemoteAudioOperations(
            send=send_remote,
            synthesize=self._generate_tts,
            record_max_seconds=MAX_CAPTURE_AUDIO_SECONDS,
        )

        cid = uuid.uuid4().hex[:8]
        self.clients[cid] = client
        logger.info(f"Client {cid} connected (hs={hs_id})")
        await self._send(client, protocol_message(
            "hello.required", supported_protocols=[PROTOCOL_VERSION]
        ))
        return client

    async def _on_disconnect(self, client):
        if client.closed:
            return
        client.closed = True
        tasks = []
        current = asyncio.current_task()
        for task in (client.llm_task, client.record_task):
            if task and task is not current and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if client.remote_audio:
            await client.remote_audio.close()
        if client.asr_client:
            await client.asr_client.aclose()
        self._headless.delete_session(client.headless_session_id)
        for cid, c in list(self.clients.items()):
            if c is client:
                del self.clients[cid]
                logger.info(f"Client {cid} disconnected")
                break

    async def _handle_message(self, client, raw):
        if client.closed:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        t = str(msg.get("type", ""))
        if not client.protocol_ready:
            try:
                device_id = RemoteAudioOperations.validate_hello(msg)
                client.protocol_ready = True
                client.device_id = device_id
                await self._send(client, RemoteAudioOperations.hello_ack(device_id))
                await self._send(client, protocol_message("state", state="listening"))
                logger.info("Remote audio device ready: %s", device_id)
            except ProtocolError as exc:
                await self._send_protocol_error(client, exc)
            return
        try:
            validate_version(msg)
        except ProtocolError as exc:
            await self._send_protocol_error(client, exc)
            return

        if t == "monitor.audio":
            audio = msg.get("audio") if isinstance(msg.get("audio"), dict) else {}
            expected_monitor = {
                "encoding": "pcm_s16le", "sample_rate": AUDIO_SAMPLE_RATE, "channels": 1,
            }
            if any(audio.get(key) != wanted for key, wanted in expected_monitor.items()):
                await self._send_protocol_error(client, ProtocolError(
                    "UNSUPPORTED_AUDIO_FORMAT", "monitor.audio 必须是 16kHz 单声道 PCM S16LE"
                ))
                return
            await self._handle_audio(
                client,
                audio.get("data", ""),
                sequence=msg.get("seq"),
                sample_rate=audio.get("sample_rate"),
                stream_id=str(msg.get("stream_id", "")),
            )
        elif t == "hotwords":
            client.hotwords = msg.get("words", [])
            logger.debug(f"Hotwords updated: {client.hotwords}")
        elif t == "image":
            client.pending_image = msg.get("data", "")
            logger.debug(f"Image stored ({len(client.pending_image)} chars b64)")
        elif t == "control.reset":
            await self._handle_reset(client)
        elif t.startswith(("operation.", "playback.", "record.")):
            try:
                await client.remote_audio.handle_message(msg)
            except ProtocolError as exc:
                client.remote_audio.fail_protocol(exc)
                await self._send_protocol_error(client, exc)
        elif t != "hello":
            await self._send_protocol_error(
                client, ProtocolError("UNKNOWN_MESSAGE_TYPE", f"不支持的消息: {t}")
            )

    async def _send_protocol_error(self, client, exc: ProtocolError):
        await self._send(client, protocol_message(
            "protocol.error",
            operation_id=exc.operation_id,
            error={"code": exc.code, "message": str(exc), "retryable": False},
        ))

    async def _handle_reset(self, client):
        """Cancel in-flight work and restore a clean per-client session."""
        if client.closed or client.resetting:
            return
        client.resetting = True
        try:
            tasks = []
            current = asyncio.current_task()
            for task in (client.llm_task, client.record_task):
                if task and task is not current and not task.done():
                    task.cancel()
                    tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            client.llm_task = None
            client.record_task = None

            if client.remote_audio:
                await client.remote_audio.close("设备会话已重置")
            old_hs = client.headless_session_id
            client.headless_session_id = self._headless.create_session()
            self._headless.delete_session(old_hs)
            client.kws_session_id = uuid.uuid4().hex

            client.state = VoiceState.LISTENING
            client.audio_preroll.clear()
            client.capture_audio.clear()
            client.capture_peak_rms = 0
            client.capture_chunk_count = 0
            client.last_audio_sequence = None
            client.monitor_stream_id = ""
            client.hotword_hit_word = ""
            client.pending_image = None
            client.cooldown_until = 0.0
            client.processing_started_at = 0.0
            await self._send(client, protocol_message("asr.result", text="", final=True))
            await self._send(client, protocol_message("control.reset_completed"))
            await self._send(client, protocol_message("state", state="listening"))
            logger.info("Client session reset")
        finally:
            client.resetting = False

    # ── audio handling ─────────────────────────────────────────────────

    def _ingest_audio_chunk(
        self,
        client,
        b64_data,
        sequence: Optional[int] = None,
        sample_rate: Optional[int] = None,
        stream_id: str = "",
    ) -> bool:
        """Decode one non-overlapping remote-audio.v1 monitor PCM chunk."""
        try:
            pcm = base64.b64decode(b64_data, validate=True)
        except Exception as e:
            logger.warning(f"Invalid audio chunk: {e}")
            return False

        if len(pcm) % AUDIO_SAMPLE_WIDTH:
            logger.warning("Odd-length PCM chunk (%d bytes), rejected", len(pcm))
            return False
        if not pcm:
            return False

        if sample_rate != AUDIO_SAMPLE_RATE:
            return False

        if not stream_id:
            logger.warning("monitor.audio 缺少 stream_id")
            return False
        if stream_id != client.monitor_stream_id:
            client.monitor_stream_id = stream_id
            client.last_audio_sequence = None
            client.kws_session_id = uuid.uuid4().hex
            client.audio_preroll.clear()

        if not isinstance(sequence, int) or sequence < 0:
            logger.warning("monitor.audio seq 必须是非负整数")
            return False
        contiguous = client.last_audio_sequence is None or sequence == client.last_audio_sequence + 1
        if not contiguous:
            logger.info(
                "Monitor audio sequence gap: previous=%s current=%s; dropping chunk",
                client.last_audio_sequence,
                sequence,
            )
            return False
        client.last_audio_sequence = sequence

        if client.state == VoiceState.LISTENING:
            client.audio_preroll.extend(pcm)
            excess = len(client.audio_preroll) - AUDIO_PREROLL_BYTES
            if excess > 0:
                del client.audio_preroll[:excess]
        return True

    async def _handle_audio(
        self,
        client,
        b64_data,
        sequence: Optional[int] = None,
        sample_rate: Optional[int] = None,
        stream_id: str = "",
    ):
        """Forward audio to ASR, detect hotword hits, manage capture state."""
        if client.closed or not b64_data:
            return

        if not self._ingest_audio_chunk(
            client,
            b64_data,
            sequence=sequence,
            sample_rate=sample_rate,
            stream_id=stream_id,
        ):
            return

        # Debug throttle
        client.chunk_count += 1
        cnt = client.chunk_count
        if cnt % 50 == 1:
            logger.debug(f"Audio chunk #{cnt} state={client.state.value}")

        # Active recording/processing cannot be interrupted by another wakeup.
        if client.state != VoiceState.LISTENING:
            return

        # ── KWS is independent of ASR text.  This is the hotword critical
        # path: the service receives raw 16 kHz PCM and returns keyword hits.
        body = {
            "audio": b64_data,
            "model": "sherpa-kws",
            "sample_rate": AUDIO_SAMPLE_RATE,
            "hotwords": client.hotwords,
            "session_id": client.kws_session_id,
        }

        try:
            started = time.monotonic()
            url = f"{self.router_url}/v1/audio/keywords"
            resp = await client.asr_client.post(url, json=body)
            if resp.status_code != 200:
                logger.error(f"ASR HTTP {resp.status_code}: {resp.text[:150]}")
                return
            data = resp.json()
            logger.info("PERF: sherpa_kws=%.3fs", time.monotonic() - started)
        except Exception as e:
            logger.error(f"ASR forward error: {e}")
            return

        if data.get("error"):
            logger.warning(f"ASR error: {data['error']}")
            return

        hits = data.get("hotword_hits", [])

        # Debug log every ASR response
        if hits:
            logger.debug("KWS hits=%s", hits)

        # ── LISTENING: watch for hotword ──
        if client.state == VoiceState.LISTENING:
            if hits:
                # Cooldown: ignore hotword within 3s of last LLM response
                cooldown = client.cooldown_until
                if time.time() < cooldown:
                    logger.debug(f"Hotword ignored (cooldown until {cooldown})")
                else:
                    hw = hits[0]
                    client.hotword_hit_word = hw
                    logger.info(f"HOTWORD '{hw}' -> CAPTURING")
                    self._transition_to(client, VoiceState.CAPTURING)

                    client.capture_peak_rms = 0
                    client.capture_chunk_count = 0
                    client.capture_audio = bytearray(client.audio_preroll)
                    client.audio_preroll.clear()
                    logger.info(
                        "Audio capture started: pre_roll=%.3fs",
                        len(client.capture_audio)
                        / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
                    )

                    await self._send(client, protocol_message(
                        "hotword.hit", word=hw
                    ))
                    await self._send(client, protocol_message(
                        "state", state="recording"
                    ))
                    if client.record_task and not client.record_task.done():
                        logger.warning("Hotword ignored because record operation is active")
                    else:
                        client.record_task = asyncio.create_task(
                            self._run_hotword_record(client)
                        )
        # CAPTURING is driven exclusively by remote record chunks and WebRTC VAD.

    async def _run_hotword_record(self, client):
        """Run the protocol-level record operation without blocking WS receive."""
        try:
            if not client.remote_audio:
                raise RuntimeError("远程音频控制器未初始化")
            result = await client.remote_audio.record()
            client.capture_audio.extend(result.pcm)
            client.capture_chunk_count = max(
                1, len(result.pcm) // (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH // 50)
            )
            client.capture_peak_rms = _pcm_rms(result.pcm)
            logger.info(
                "Remote record completed: operation_id=%s duration=%.2fs reason=%s",
                result.operation_id, result.duration_seconds, result.stop_reason,
            )
            client.pending_input_metadata = {
                "source": "hotword_record",
                "operation_id": result.operation_id,
                "record_completed_at": time.time(),
                "record_duration_seconds": result.duration_seconds,
                "record_bytes": len(result.pcm),
                "record_sha256": hashlib.sha256(result.pcm).hexdigest(),
                "stop_reason": result.stop_reason,
            }
            await self._finalize_capture(client)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Remote record failed: %s", exc)
            if not client.closed and not client.resetting:
                self._transition_to(client, VoiceState.LISTENING)
                await self._send(client, protocol_message(
                    "state", state="listening",
                    error={"code": getattr(exc, "code", "RECORD_FAILED"), "message": str(exc)},
                ))

    # ── finalize capture & dispatch to LLM ────────────────────────────

    @staticmethod
    def _pcm_to_wav(pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(AUDIO_SAMPLE_WIDTH)
            wav.setframerate(AUDIO_SAMPLE_RATE)
            wav.writeframes(pcm)
        return output.getvalue()

    def _save_failed_capture(self, pcm: bytes, reason: str) -> None:
        """Keep one reproducible WAV when neither ASR can recover an utterance."""
        try:
            FAILED_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FAILED_CAPTURE_PATH.write_bytes(self._pcm_to_wav(pcm))
            logger.error(
                "Saved failed capture: path=%s reason=%s bytes=%d sha256=%s",
                FAILED_CAPTURE_PATH,
                reason,
                len(pcm),
                hashlib.sha256(pcm).hexdigest()[:16],
            )
        except Exception:
            logger.exception("Unable to save failed capture to %s", FAILED_CAPTURE_PATH)

    async def _transcribe_with_qwen3_asr(self, client, pcm: bytes) -> str:
        """Send captured 16 kHz mono PCM to Qwen3-ASR through the router."""
        wav_data = self._pcm_to_wav(pcm)
        url = f"{self.router_url}/v1/audio/transcriptions"
        started = time.monotonic()
        resp = await client.asr_client.post(
            url,
            files={"file": ("capture.wav", wav_data, "audio/wav")},
            data={"model": "qwen3-asr"},
            timeout=300.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Qwen3-ASR HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Qwen3-ASR error: {data['error']}")
        raw_text = str(data.get("text", "")).strip()
        text = _clean_qwen_asr_text(raw_text)
        if raw_text != text:
            logger.info("Qwen3-ASR control prefix removed: raw='%s'", raw_text[:100])
        logger.info(
            "PERF: qwen3_asr=%.2fs audio=%.2fs text='%s'",
            time.monotonic() - started,
            len(pcm) / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
            text[:100],
        )
        return text

    async def _transcribe_followup_segments(self, client, pcm: bytes) -> str:
        """Transcribe a long follow-up without discarding its opening words.

        The deployed ASR bmodel safely accepts at most five seconds per call.
        A previous tail-only workaround lost the beginning of a 10-second
        answer, precisely where users often say "医生明确诊断".  Segment and
        join the independently decoded windows instead.
        """
        window = QWEN3_ASR_SAFE_AUDIO_BYTES
        if len(pcm) <= window:
            return await self._transcribe_with_qwen3_asr(client, pcm)
        overlap = min(QWEN3_ASR_SEGMENT_OVERLAP_BYTES, window // 2)
        stride = window - overlap
        texts = []
        for start in range(0, len(pcm), stride):
            piece = pcm[start:start + window]
            text = await self._transcribe_with_qwen3_asr(client, piece)
            if _has_meaningful_asr_text(text):
                texts.append(text.strip())
            if start + window >= len(pcm):
                break
        if not texts:
            return ""
        # Overlapped windows can decode their shared boundary twice.  Remove
        # the longest exact character overlap before joining the segments.
        merged = []
        for text in texts:
            if not merged:
                merged.append(text)
                continue
            previous = merged[-1]
            overlap_chars = 0
            for size in range(min(len(previous), len(text), 24), 0, -1):
                if previous[-size:] == text[:size]:
                    overlap_chars = size
                    break
            text = text[overlap_chars:].lstrip("，。！？；、 ")
            if text:
                merged.append(text)
        combined = "，".join(merged)
        logger.info(
            "Qwen3-ASR segmented follow-up: %.2fs -> %d windows overlap=%.1fs text='%s'",
            len(pcm) / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
            len(texts), QWEN3_ASR_SEGMENT_OVERLAP_SECONDS, combined[:120],
        )
        return combined

    async def _finalize_capture(self, client):
        """Transcribe captured PCM with Qwen3-ASR and dispatch it to the LLM."""
        logger.info("_finalize_capture: ENTER")
        try:
            if client.state != VoiceState.CAPTURING:
                logger.debug("_finalize_capture: ignored outside CAPTURING state")
                return

            pcm = bytes(client.capture_audio)
            client.capture_audio.clear()

            if not pcm:
                logger.info("_finalize_capture: empty audio -> LISTENING")
                self._transition_to(client, VoiceState.LISTENING)
                await self._send(client, protocol_message("state", state="listening"))
                return

            qwen_pcm = _limit_pcm_tail(pcm)
            logger.info(
                "_finalize_capture: audio raw=%.2fs qwen=%.2fs bytes=%d "
                "chunks=%d peak_rms=%d sha256=%s",
                len(pcm) / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
                len(qwen_pcm) / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
                len(pcm),
                client.capture_chunk_count,
                client.capture_peak_rms,
                hashlib.sha256(pcm).hexdigest()[:16],
            )
            if len(qwen_pcm) < len(pcm):
                logger.warning(
                    "Qwen3-ASR input limited to final %.1fs: discarded %.2fs pre-roll/old audio",
                    QWEN3_ASR_SAFE_AUDIO_SECONDS,
                    (len(pcm) - len(qwen_pcm))
                    / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
                )
            self._transition_to(client, VoiceState.PROCESSING)
            client.processing_started_at = time.monotonic()
            await self._send(client, protocol_message("state", state="processing"))

            qwen_error = None
            asr_started_at = time.time()
            try:
                utterance = await self._transcribe_with_qwen3_asr(
                    client, qwen_pcm
                )
            except Exception as exc:
                qwen_error = exc
                utterance = ""
                logger.exception("Qwen3-ASR transcription failed")

            if not _has_meaningful_asr_text(utterance):
                reason = (
                    f"Qwen3-ASR error: {qwen_error}"
                    if qwen_error is not None
                    else f"Qwen3-ASR unusable text: {utterance!r}"
                )
                self._save_failed_capture(pcm, reason)
                raise RuntimeError(reason)

            logger.info(f"_finalize_capture: Qwen utterance -> LLM: '{utterance[:80]}'")
            await self._send(client, protocol_message(
                "asr.result", text=utterance, final=True
            ))

            client.pending_input_metadata = {
                **client.pending_input_metadata,
                "source": "hotword_record",
                "asr_text": utterance,
                "audio_bytes": len(pcm),
                "audio_sha256": hashlib.sha256(pcm).hexdigest(),
                "audio_duration_seconds": len(pcm) / (AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH),
                "asr_started_at": asr_started_at,
                "asr_completed_at": time.time(),
            }
            # ASR provenance belongs in trajectory metadata, not in the
            # medical query itself: its warning words can be mistaken for
            # clinical entities (for example "修正").
            client.llm_task = asyncio.create_task(self._llm_dispatch(client, utterance))
            logger.info("_finalize_capture: LLM task created, DONE")
        except Exception as e:
            logger.exception(f"_finalize_capture: EXCEPTION {e}")
            self._transition_to(client, VoiceState.LISTENING)
            await self._send(client, protocol_message("state", state="listening"))

    # ── LLM dispatch ───────────────────────────────────────────────────

    async def _llm_dispatch(self, client, utterance):
        """Submit to HeadlessManager; SpeakTool performs remote playback inline."""
        try:
            current_input = utterance
            while True:
                hs = self._headless.get_session(client.headless_session_id)
                sys_len = 0
                if hs and hs.session_obj:
                    for m in hs.session_obj.history:
                        if m.role == "system":
                            sys_len = len(m.content)
                            break
                logger.info(f"LLM_INPUT: sys_len={sys_len} user='{current_input[:120]}'")
                _t0 = time.time()
                result = await self._headless.submit(
                    session_id=client.headless_session_id,
                    text=current_input,
                    image=client.pending_image,
                    tool_context_extra={
                        "remote_audio": client.remote_audio,
                        "input_metadata": dict(client.pending_input_metadata),
                    },
                )
                client.pending_image = None
                client.pending_input_metadata = {}
                _t1 = time.time()
                logger.info(f"PERF: agent_loop={_t1 - _t0:.2f}s (submit → result)")

                full_text = result.text or ""
                logger.info(f"LLM_OUTPUT: len_text={len(full_text)} text='{full_text[:200]}'")
                if full_text.strip():
                    await self._send(client, protocol_message(
                        "agent.output", text=full_text, final=True,
                        turn_end_reason=result.turn_end_reason,
                        session_ended=result.session_ended,
                    ))

                if result.turn_end_reason == "query" and result.continuation_pcm:
                    asr_started_at = time.time()
                    followup = await self._transcribe_followup_segments(
                        client, result.continuation_pcm
                    )
                    if not _has_meaningful_asr_text(followup):
                        raise RuntimeError("query 录音未识别出有效文本")
                    await self._send(client, protocol_message(
                        "asr.result", text=followup, final=True, source="query"
                    ))
                    audio = dict(result.continuation_audio)
                    client.pending_input_metadata = {
                        **audio,
                        "source": "query_record",
                        "asr_text": followup,
                        "asr_started_at": asr_started_at,
                        "asr_completed_at": time.time(),
                    }
                    current_input = followup
                    continue

                if result.session_ended:
                    old_hs = client.headless_session_id
                    client.headless_session_id = self._headless.create_session()
                    self._headless.delete_session(old_hs)
                    logger.info("Voice session ended by speak; new hotword session=%s", client.headless_session_id)
                elif result.turn_end_reason != "query":
                    # An abnormal turn cannot leak into a fourth user turn.  The
                    # trajectory remains on disk with SESSION_NOT_TERMINATED.
                    old_hs = client.headless_session_id
                    client.headless_session_id = self._headless.create_session()
                    self._headless.delete_session(old_hs)
                    logger.warning(
                        "Voice session reset after non-terminal turn; new session=%s",
                        client.headless_session_id,
                    )
                break
        except Exception as e:
            logger.error(f"LLM dispatch error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await self._send(client, protocol_message(
                "agent.output", text=f"[错误: {e}]", final=True
            ))
        finally:
            client.cooldown_until = time.time() + 3.0
            if not client.closed and not client.resetting:
                # sherpa-kws keeps decoder state by session_id.  Reusing the
                # just-detected stream makes a subsequent wake word look like
                # continuation audio and can permanently suppress its next
                # hit.  A spoken turn is a hard KWS boundary.
                client.kws_session_id = uuid.uuid4().hex
                client.audio_preroll.clear()
                # Keep the isolated headless session so follow-up utterances
                # retain bounded conversational context.
                self._transition_to(client, VoiceState.LISTENING)
                await self._send(client, protocol_message("state", state="listening"))

    def _transition_to(self, client, new_state):
        old = client.state
        client.state = new_state
        if old != new_state:
            logger.debug(f"State: {old.value} -> {new_state.value}")

    async def _send(self, client, data):
        if client.closed:
            return
        try:
            payload = json.dumps(data, ensure_ascii=False)
            async with client.send_lock:
                await client.ws.send_text(payload)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"WebSocket send failed: {type(e).__name__}: {e}")


def _extract_after_hotword(text, hotword):
    idx = text.rfind(hotword)
    if idx == -1:
        return ""
    after = text[idx + len(hotword):]
    return re.sub(r"^[\s,，。.!！？?~～、]+", "", after).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Agent Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--trajectory-dir", default="/data/structure/trajectories")
    parser.add_argument("--no-trajectory", action="store_true")
    args = parser.parse_args()
    model = select_model(
        args.router_url,
        requested=args.model,
        default="qwen3.5-4b-history",
        interactive=True,
    )
    server = VoiceAgentServer(
        host=args.host, port=args.port, router_url=args.router_url,
        model=model, trajectory_dir=args.trajectory_dir,
        trajectory_enabled=not args.no_trajectory,
        
    )
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(server.shutdown())


if __name__ == "__main__":
    main()
