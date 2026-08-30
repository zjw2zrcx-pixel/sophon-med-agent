#!/usr/bin/env python3
"""OpenAI-compatible embedding server for the Qwen3-Embedding BM1684X bundle."""

import argparse
import asyncio
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from logging_utils import setup_colored_logging


logger = setup_colored_logging("qwen3-embedding")

MODEL_NAME = "qwen3-embedding-0.6b"
MODEL_VARIANT = "bf16"
EXPECTED_MODEL_ARTIFACT = "qwen3_embedding_bf16_seq512_bm1684x.bmodel"
encoder = None
server_state = "initializing"
executor = ThreadPoolExecutor(max_workers=1)


async def load_model():
    global encoder, server_state
    cfg = app.state.cfg
    module_path = os.environ.get("MODULE_PATH", "../Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512")
    try:
        server_state = "loading"
        model_path = Path(cfg["model_path"]).expanduser().resolve()
        if model_path.name != EXPECTED_MODEL_ARTIFACT:
            raise RuntimeError(
                "Refusing to load a non-BF16 embedding artifact: "
                f"expected '{EXPECTED_MODEL_ARTIFACT}', got '{model_path.name}'"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"BF16 embedding artifact not found: {model_path}")
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        from embed_sail import Qwen3Embedding

        logger.info("Loading %s embedding model from %s ...", MODEL_VARIANT, model_path)
        loop = asyncio.get_running_loop()
        encoder = await loop.run_in_executor(
            executor,
            Qwen3Embedding,
            model_path,
            cfg["config_path"],
            cfg["devid"],
        )
        # Do not report ready until one real TPU inference has run; otherwise
        # the first medical query pays the device initialisation cost.
        await loop.run_in_executor(executor, _warm_encoder)
        server_state = "ready"
        logger.info("Embedding model loaded successfully.")
    except Exception:
        server_state = "error"
        logger.exception("Failed to load embedding model")
        encoder = None


def _encode_many(inputs: list[str], dimensions: int):
    """The exported bmodel is static batch=1, so execute inputs serially."""
    vectors = []
    prompt_tokens = 0
    for text in inputs:
        tokens = encoder.tokenizer(
            text,
            truncation=True,
            max_length=encoder.seq_length,
            return_tensors="np",
        )
        prompt_tokens += int(tokens["attention_mask"].sum())
        vectors.append(encoder.encode(text, dimensions).tolist())
    return vectors, prompt_tokens


def _warm_encoder() -> None:
    """Prime the embedding TPU path immediately after model mapping."""
    if os.environ.get("EMBEDDING_MODEL_PREWARM", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        logger.info("Embedding model prewarm disabled by EMBEDDING_MODEL_PREWARM")
        return
    started = time.perf_counter()
    text = os.environ.get("EMBEDDING_PREWARM_TEXT", "医疗检索预热")
    _encode_many([text], 256)
    logger.info(
        "Embedding TPU inference prewarm completed: %.1f ms",
        (time.perf_counter() - started) * 1000,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.get_running_loop().create_task(load_model())
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Qwen3-Embedding OpenAI-Compatible API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": server_state}


@app.get("/status")
async def status():
    return {
        "status": server_state,
        "model": MODEL_NAME,
        "type": "embedding",
        "variant": MODEL_VARIANT,
        "artifact": EXPECTED_MODEL_ARTIFACT,
    }


@app.post("/shutdown")
async def shutdown():
    global encoder
    encoder = None
    logger.info("Shutting down...")
    asyncio.get_running_loop().call_later(1.0, lambda: os._exit(0))
    return JSONResponse({"status": "shutting_down"})


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    if server_state != "ready" or encoder is None:
        return JSONResponse(
            {"error": {"message": f"Model '{MODEL_NAME}' is not ready, current state: {server_state}", "type": "server_error"}},
            status_code=503,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}}, status_code=400)

    requested_model = body.get("model", MODEL_NAME)
    if requested_model not in (MODEL_NAME, "qwen3-embedding"):
        return JSONResponse(
            {"error": {"message": f"Model '{requested_model}' not found", "type": "invalid_request_error"}},
            status_code=400,
        )
    value = body.get("input")
    inputs = [value] if isinstance(value, str) else value
    if not isinstance(inputs, list) or not inputs or not all(isinstance(item, str) for item in inputs):
        return JSONResponse(
            {"error": {"message": "'input' must be a non-empty string or list of strings", "type": "invalid_request_error"}},
            status_code=400,
        )
    dimensions = body.get("dimensions", 256)
    if not isinstance(dimensions, int) or not 32 <= dimensions <= 1024:
        return JSONResponse(
            {"error": {"message": "'dimensions' must be an integer in [32, 1024]", "type": "invalid_request_error"}},
            status_code=400,
        )
    encoding_format = body.get("encoding_format", "float")
    if encoding_format not in ("float", "base64"):
        return JSONResponse(
            {"error": {"message": "'encoding_format' must be 'float' or 'base64'", "type": "invalid_request_error"}},
            status_code=400,
        )

    try:
        loop = asyncio.get_running_loop()
        vectors, prompt_tokens = await loop.run_in_executor(executor, _encode_many, inputs, dimensions)
    except Exception as exc:
        logger.exception("Embedding inference failed")
        return JSONResponse({"error": {"message": str(exc), "type": "server_error"}}, status_code=500)

    data = []
    for index, vector in enumerate(vectors):
        embedding = vector
        if encoding_format == "base64":
            import numpy as np
            embedding = base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")
        data.append({"object": "embedding", "embedding": embedding, "index": index})
    return {
        "object": "list",
        "data": data,
        "model": MODEL_NAME,
        "model_variant": MODEL_VARIANT,
        "model_artifact": EXPECTED_MODEL_ARTIFACT,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True, help="Tokenizer directory")
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--devid", type=int, default=0)
    args = parser.parse_args()
    app.state.cfg = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "devid": args.devid,
    }
    os.environ["MODULE_PATH"] = args.module_path
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
