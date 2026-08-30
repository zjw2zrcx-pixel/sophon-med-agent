#!/usr/bin/env python3
"""Self-contained HTTP service for the BM1684X TPU KWS deployment."""
import argparse
import asyncio
import base64
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sail_kws_runner import SailKwsRunner


ROOT = Path(__file__).resolve().parent
state = "initializing"
error = ""
runner_args: tuple[Path, Path] | None = None


@dataclass
class Session:
    runner: SailKwsRunner
    clock: float = 0.0


sessions: dict[str, Session] = {}


def load() -> None:
    global error, runner_args, state
    try:
        state = "loading"
        model = ROOT / "kws_transducer_chunk16_bm1684x_f32.bmodel"
        tokens = ROOT / "runtime" / "tokens.txt"
        # Construct once before readiness so a missing TPU or extension cannot
        # produce a false healthy response.
        SailKwsRunner(model, tokens)
        runner_args = (model, tokens)
        state = "ready"
    except Exception as exc:  # surfaced through /health
        error = str(exc)
        state = "error"


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.get_running_loop().run_in_executor(None, load)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": state, "error": error or None}


@app.post("/v1/audio/keywords")
async def keywords(request: Request):
    if state != "ready" or runner_args is None:
        return JSONResponse({"error": {"message": error or state}}, status_code=503)
    body = await request.json()
    session_id = str(body.get("session_id") or uuid.uuid4().hex)
    pcm = base64.b64decode(body["audio"])
    session = sessions.get(session_id)
    if session is None:
        session = Session(SailKwsRunner(*runner_args))
        sessions[session_id] = session
    start = session.clock
    session.clock += len(pcm) / 32000.0  # signed 16-bit, 16 kHz mono
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    tokens = session.runner.accept_waveform(samples)
    # The runner emits this only after the complete Sherpa ContextGraph phrase
    # has met its confidence and trailing-blank requirements.
    hit = bool(tokens)
    return {
        "session_id": session_id,
        "hotword_hits": ["小麦小麦"] if hit else [],
        "block_start_s": start,
        "block_end_s": session.clock,
    }


@app.post("/shutdown")
async def shutdown():
    asyncio.get_running_loop().call_later(0.2, lambda: os._exit(0))
    return {"status": "shutting_down"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    # Accepted for compatibility with server/config.yaml; artifacts are always
    # resolved relative to this self-contained deployment directory.
    parser.add_argument("--model-path")
    parser.add_argument("--config-path")
    parser.add_argument("--module-path")
    parser.add_argument("--devid", type=int, default=0)
    args = parser.parse_args()
    # Router polls /health periodically; the access lines add noise without
    # conveying a state transition. Startup and failure logs remain enabled.
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
