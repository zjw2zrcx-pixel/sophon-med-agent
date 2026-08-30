import sys
import os
import uuid
import time
import asyncio
import argparse
import tempfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

import atexit

import logging

from logging_utils import setup_colored_logging

logger = setup_colored_logging("qwen3-asr")

pipeline = None
server_state = "initializing"
executor = ThreadPoolExecutor(max_workers=1)


class SimpleArgs:
    devid = 0
    model_path = ""
    config_path = ""
    language = None


def make_args(cfg: dict):
    args = SimpleArgs()
    args.devid = cfg.get("devid", 0)
    args.model_path = cfg["model_path"]
    args.config_path = cfg["config_path"]
    args.language = cfg.get("language")
    return args


async def load_model():
    global pipeline, server_state
    module_path = os.environ.get("MODULE_PATH", "../Qwen3_ASR/python_demo")
    cfg = app.state.cfg

    try:
        server_state = "loading"
        logger.info(f"Adding module path: {module_path}")
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        import chat
        import qwen_asr
        from pipeline import Qwen3_ASR

        args = make_args(cfg)
        logger.info(f"Loading model from {args.model_path} ...")
        loop = asyncio.get_running_loop()
        pipeline = await loop.run_in_executor(None, _load_model_sync, args, Qwen3_ASR)
        server_state = "ready"
        logger.info("Model loaded successfully.")
    except Exception as e:
        server_state = "error"
        logger.exception(f"Failed to load model: {e}")
        if pipeline is not None:
            try:
                pipeline.model.deinit()
            except Exception:
                pass
            pipeline = None


def _load_model_sync(args, cls):
    return cls(args)


def _cleanup_model():
    global pipeline
    if pipeline is not None:
        try:
            logger.info("Calling model.deinit() (atexit) ...")
            pipeline.model.deinit()
        except Exception:
            pass
        pipeline = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.get_running_loop().create_task(load_model())
    yield
    _cleanup_model()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": server_state}


@app.get("/status")
async def status_detail():
    return {
        "status": server_state,
        "model": "qwen3-asr",
        "type": "audio",
    }


@app.post("/shutdown")
async def do_shutdown():
    _cleanup_model()
    logger.info("Shutting down...")
    asyncio.get_running_loop().call_later(1.0, lambda: os._exit(0))
    return JSONResponse({"status": "shutting_down"})


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr"),
    language: str = Form(None),
):
    if server_state != "ready" or pipeline is None:
        return JSONResponse(
            {"error": {"message": f"Server not ready, current state: {server_state}", "type": "server_error"}},
            status_code=503,
        )

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    if not suffix:
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        context_str = ""
        lang = language if language else None

        loop = asyncio.get_running_loop()
        full_text = await loop.run_in_executor(
            executor, _collect_transcribe, context_str, tmp.name, lang
        )

        return JSONResponse({"text": full_text})
    except Exception as e:
        logger.exception("ASR transcription failed")
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _collect_transcribe(context_str, audio_path, lang):
    # Qwen3_ASR.transcribe() currently accepts ``language`` but does not apply it;
    # build_text_prompt() reads pipeline.language instead. Inference is serialized
    # by the single-worker executor, so temporarily overriding it is safe.
    previous_language = pipeline.language
    if lang is not None:
        pipeline.language = lang

    try:
        chunks = pipeline.transcribe(
            context_str,
            audio_path,
            language=lang,
            clear_history_flag=True,
        )
        full_text = "".join(chunks)
    finally:
        pipeline.language = previous_language

    logger.info(f"ASR done: {len(full_text)} chars")
    return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model-path", type=str, default="../Qwen3_ASR/qwen3_asr.bmodel")
    parser.add_argument("--config-path", type=str, default="../Qwen3_ASR/config")
    parser.add_argument("--module-path", type=str, default="../Qwen3_ASR/python_demo")
    parser.add_argument("--devid", type=int, default=0)
    parser.add_argument("--language", type=str, default=None)
    args = parser.parse_args()

    app.state.cfg = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "devid": args.devid,
        "language": args.language,
    }
    os.environ["MODULE_PATH"] = args.module_path

    atexit.register(_cleanup_model)

    uvicorn.run(app, host=args.host, port=args.port, log_config={
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
