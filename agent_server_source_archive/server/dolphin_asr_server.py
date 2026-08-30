#!/usr/bin/env python3
"""
Dolphin-CN ASR Server - OpenAI-compatible HTTP API
===================================================
Replaces the Qwen3-ASR server with the Dolphin-CN streaming model,
exposing the same /v1/audio/transcriptions endpoint on port 8002.

Supports:
  - multipart/form-data  (file upload, same as Qwen3-ASR)
  - application/json     (base64-encoded audio)

The Dolphin model is inherently streaming (chunked TPU encoder + CPU decode).
For the non-streaming API, we read the entire audio file, process it chunk-by-
chunk, and return the accumulated text.
  - application/json  streaming chunks  (POST /v1/audio/transcriptions/stream)
    with session management for real-time ASR.
"""
import sys
import os
import math
import json
import base64
import time
import asyncio
import argparse
import tempfile
import types
import logging
import ctypes
import gc
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import yaml
import numpy as np
import torch
import torchaudio
import uuid
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import atexit

from logging_utils import setup_colored_logging

logger = setup_colored_logging("dolphin-asr")

# ---------------------------------------------------------------------------
# Constants (matching streaming_file_tpu.py / streaming_tpu_server.py)
# ---------------------------------------------------------------------------
NUM_LAYERS = 12
MODEL_DIM = 768
NUM_HEADS = 12
D_K = 64
MAX_CACHE_T = 256
CHUNK_SIZE = 16
SAMPLE_RATE = 16000
SUBSAMPLING = 4
RIGHT_CONTEXT = 6
CONTEXT = RIGHT_CONTEXT + 1
STRIDE = SUBSAMPLING * CHUNK_SIZE
DECODING_WINDOW = (CHUNK_SIZE - 1) * SUBSAMPLING + CONTEXT
FRAME_SHIFT = 160
FRAME_LENGTH = 400
CHUNK_SAMPLES = (DECODING_WINDOW - 1) * FRAME_SHIFT + FRAME_LENGTH
STRIDE_SAMPLES = STRIDE * FRAME_SHIFT
GRAPH_NAMES = [f"encoder_layer_{i}" for i in range(NUM_LAYERS)]

# ---------------------------------------------------------------------------
# --- Auto-reset thresholds ---
VAD_SPEECH_THRESHOLD = 800.0
SILENCE_TIMEOUT_MS = 10000
MAX_SAFE_OFFSET = 45000
CHUNK_TIME_MS = int(STRIDE_SAMPLES / SAMPLE_RATE * 1000)  # ~640ms

# Global state
# ---------------------------------------------------------------------------
engine_obj = None          # The Dolphin inference engine
server_state = "initializing"
executor = ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------------------------
# Streaming session management
# ---------------------------------------------------------------------------
HOTWORD_BIAS = 3.0


@dataclass
class StreamingSession:
    """Holds per-client state for streaming ASR."""
    att_cache: np.ndarray = field(default_factory=lambda: np.zeros(
        (NUM_LAYERS, 1, NUM_HEADS, MAX_CACHE_T, D_K * 2), dtype=np.float32))
    cnn_cache: np.ndarray = field(default_factory=lambda: np.zeros(
        (NUM_LAYERS, 1, MODEL_DIM * 2, 30), dtype=np.float32))
    offset: int = 0
    prev_tokens: Optional[List[int]] = None
    last_partial_text: str = ""
    chunk_count: int = 0
    last_activity: float = field(default_factory=time.time)
    hotwords: List[str] = field(default_factory=list)
    hw_bias: Optional[np.ndarray] = None
    hit_hotwords: set = field(default_factory=set)
    silence_ms: int = 0
    had_speech: bool = False
    reset_count: int = 0


streaming_sessions: Dict[str, StreamingSession] = {}
SESSION_TTL = 300  # 5 minutes


def _evict_stale_sessions():
    """Remove sessions idle > SESSION_TTL."""
    now = time.time()
    stale = [sid for sid, s in streaming_sessions.items()
             if now - s.last_activity > SESSION_TTL]
    for sid in stale:
        del streaming_sessions[sid]
    if stale:
        logger.debug("Evicted %d stale streaming sessions", len(stale))


def _build_hotword_bias(tokenizer, words: List[str]) -> Optional[np.ndarray]:
    """Build a CTC logprob bias vector for hotword characters."""
    if not words or tokenizer is None:
        return None
    t2i = {k: int(v) for k, v in tokenizer._symbol_table.items()}
    bias = np.zeros(18173, dtype=np.float32)
    for w in words:
        for ch in w:
            bias[t2i.get(ch, 0)] = HOTWORD_BIAS
    return bias


def _apply_hotword_bias(logprobs: np.ndarray, hw_bias: Optional[np.ndarray]) -> np.ndarray:
    """Add hotword bias to log-probabilities."""
    if hw_bias is None:
        return logprobs
    result = logprobs.copy()
    result[0, :, :] = logprobs[0, :, :] + hw_bias
    return result



# ---------------------------------------------------------------------------
# Model loading helpers (adapted from streaming_file_tpu.py)
# ---------------------------------------------------------------------------

def _load_cpu_model_and_tokenizer(dolphin_dir, cpu_dir):
    """Load minimal CPU model (CMVN, PE, after_norm) and tokenizer."""
    dd = os.path.join(dolphin_dir, "Dolphin", "dolphin")
    dp = types.ModuleType('dolphin')
    dp.__path__ = [dd]
    dp.__file__ = os.path.join(dd, '__init__.py')
    sys.modules['dolphin'] = dp

    for sub in ['common', 'mask', 'search', 'tokenizer', 'hotword',
                'audio', 'constants', 'languages', 'processor']:
        sp = spec_from_file_location(
            f'dolphin.{sub}',
            os.path.join(dd, f'{sub}.py'),
            submodule_search_locations=[],
        )
        m = module_from_spec(sp)
        sys.modules[f'dolphin.{sub}'] = m
        sp.loader.exec_module(m)

    sp = spec_from_file_location(
        'dolphin.model',
        os.path.join(dd, 'model.py'),
        submodule_search_locations=[],
    )
    dm = module_from_spec(sp)
    sys.modules['dolphin.model'] = dm
    sp.loader.exec_module(dm)

    from dolphin.model import EBranchformerEncoder, TransformerDecoder
    oe = EBranchformerEncoder.__init__

    def _enc(s, *a, **kw):
        oe(s, *a, **kw)
        s.encoders = torch.nn.ModuleList([])

    EBranchformerEncoder.__init__ = _enc
    od = TransformerDecoder.__init__

    def _dec(s, *a, **kw):
        od(s, *a, **kw)
        s.decoders = torch.nn.ModuleList([])

    TransformerDecoder.__init__ = _dec

    with open(os.path.join(cpu_dir, 'train.yaml')) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg['cmvn_conf']['cmvn_file'] = os.path.join(cpu_dir, 'global_cmvn')
    cfg['tokenizer_conf']['symbol_table_path'] = os.path.join(cpu_dir, 'units.txt')

    model = dm.init_speech_model(cfg)
    sd = torch.load(
        os.path.join(cpu_dir, 'cpu_components_minimal.pt'),
        map_location='cpu',
    )
    model.load_state_dict(sd, strict=False)
    model.eval()

    EBranchformerEncoder.__init__ = oe
    TransformerDecoder.__init__ = od

    # Expand positional encoding to 50000 (for long streaming sessions)
    pe = model.encoder.embed.pos_enc
    if hasattr(pe, 'pe'):
        om = pe.max_len
        pe.max_len = 50000
        op = pe.pe
        np1 = torch.zeros(1, 50000, op.size(2))
        np1[:, :om, :] = op
        d_dim = pe.d_model
        n_new = 50000 - om
        i = torch.arange(om, 50000, dtype=torch.float64).unsqueeze(1)
        j = torch.arange(0, d_dim // 2, dtype=torch.float64).unsqueeze(0)
        div = 10000.0 ** (2 * j / d_dim)
        vals = i / div
        np1[0, om:, 0:d_dim:2] = torch.sin(vals).float()
        np1[0, om:, 1:d_dim:2] = torch.cos(vals).float()
        pe.pe = np1

    import dolphin.tokenizer as dt
    return model, dt.init_tokenizer(cfg)


def _process_stream_chunk(session, chunk_int16, engine):
    """Process a single PCM chunk using session state, return partial results."""
    from dolphin.search import ctc_greedy_search

    model = engine.model
    sail_eng = engine.sail_engine
    tokenizer = engine.tokenizer

    N = NUM_LAYERS
    session.chunk_count += 1

    # Ensure correct size
    if len(chunk_int16) != CHUNK_SAMPLES:
        chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
        copy_len = min(len(chunk_int16), CHUNK_SAMPLES)
        chunk[:copy_len] = chunk_int16[:copy_len]
        chunk_int16 = chunk

    # Fbank
    wav = torch.from_numpy(chunk_int16.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        feats = torchaudio.compliance.kaldi.fbank(
            waveform=wav, num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0, sample_frequency=SAMPLE_RATE,
        ).unsqueeze(0)

    # CPU: CMVN
    with torch.no_grad():
        xs = model.encoder.global_cmvn(feats)

    # bmodel: embed_conv
    fn = xs.unsqueeze(1).cpu().numpy().astype(np.float32)
    out = sail_eng.process('embed_conv', {'x.1': fn})
    cv = out['28']
    ct = np.transpose(cv, (0, 2, 1, 3)).reshape(
        1, CHUNK_SIZE, -1).astype(np.float32)

    # bmodel: embed_linear
    out = sail_eng.process('embed_linear', {'x.1': ct})
    xs_lin = torch.from_numpy(out['4'])

    # CPU: xscale + correct PE
    pc = model.encoder.embed.position_encoding(
        offset=session.offset, size=CHUNK_SIZE)
    xs = xs_lin * math.sqrt(MODEL_DIM) + pc

    t_out = xs.size(1)
    aks = MAX_CACHE_T + t_out
    et = min(session.offset, MAX_CACHE_T)
    vs = MAX_CACHE_T - et

    # CPU: position encoding + attention mask
    pe_enc = model.encoder.embed.position_encoding(
        offset=session.offset - et, size=aks)
    am = torch.full((1, t_out, aks), float('-inf'), dtype=torch.float32)
    for i in range(t_out):
        am[0, i, vs:MAX_CACHE_T + i + 1] = 0.0

    xn = xs.cpu().numpy().astype(np.float32)
    an = am.cpu().numpy().astype(np.float32)
    pn = pe_enc.cpu().numpy().astype(np.float32)

    # bmodel: 12 encoder layers
    for li in range(N):
        aksl = session.att_cache[li]
        inp = {
            'x.5': xn,
            'att_mask.1': an,
            'pos_emb.1': pn,
            'k_cache.1': aksl[:, :, :, :D_K],
            'v_cache.1': aksl[:, :, :, D_K:],
            'cnn_cache.1': session.cnn_cache[li],
        }
        out = sail_eng.process(GRAPH_NAMES[li], inp)
        xn = out['320']
        nk = np.concatenate([out['52'], out['53']], axis=-1)
        if nk.shape[2] > MAX_CACHE_T:
            nk = nk[:, :, nk.shape[2] - MAX_CACHE_T:, :]
        session.att_cache[li] = nk
        session.cnn_cache[li] = out['58']

    xs_out = torch.from_numpy(xn)
    session.offset += t_out

    # CPU: after_norm
    new_enc = xs_out
    with torch.no_grad():
        if model.encoder.normalize_before and model.encoder.final_norm:
            new_enc = model.encoder.after_norm(new_enc)

    # bmodel: ctc_proj
    cn = new_enc.cpu().numpy().astype(np.float32)
    out = sail_eng.process('ctc_proj', {'x.1': cn})
    cp = torch.from_numpy(out['7']).log_softmax(dim=2)

    # Apply hotword bias
    if session.hw_bias is not None:
        cp_np = cp.cpu().numpy()
        cp_np = _apply_hotword_bias(cp_np, session.hw_bias)
        cp = torch.from_numpy(cp_np)

    cl = torch.tensor([cp.size(1)])
    r = ctc_greedy_search(cp, cl)
    new_tokens = r[0].tokens[:int(cl[0])]

    if session.prev_tokens is None:
        session.prev_tokens = []
    if session.prev_tokens and new_tokens and new_tokens[0] == session.prev_tokens[-1]:
        combined = session.prev_tokens + new_tokens[1:]
    else:
        combined = session.prev_tokens + new_tokens
    session.prev_tokens = combined

    text, _ = tokenizer.detokenize(combined)

    result = {"full_text": text, "new_partial": "", "hotword_hits": []}
    # VAD disabled — voice agent handles silence detection itself.
    # Always report "speech" so the streaming pipeline stays active.
    session.silence_ms = 0
    session.had_speech = True
    result["vad"] = "speech"
    result["silence_duration"] = 0

    if text != session.last_partial_text:
        c = sum(1 for a, b in zip(session.last_partial_text, text) if a == b)
        np_ = text[c:]
        if np_:
            result["new_partial"] = np_
            result["full_text"] = text
            # Hotword check only on NEW text (fire once, not every stale response)
            new_hits = []
            for hw in session.hotwords:
                if hw in text and hw not in session.hit_hotwords:
                    new_hits.append(hw)
                    session.hit_hotwords.add(hw)
                    logger.info("[HOTWORD] hit: %s", hw)
            if new_hits:
                result["hotword_hits"] = new_hits
        session.last_partial_text = text

    # Periodic GC
    if session.chunk_count % 500 == 0:
        gc.collect()

    return result

class DolphinInferenceEngine:
    """Encapsulates the TPU+CPU Dolphin-CN model for file-level inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sail_engine = None
        self._ready = False

    def load(self, model_path, config_path, module_path, devid=0):
        """Load CPU components + TPU bmodel."""
        logger.info("Loading Dolphin-CN CPU components...")
        t0 = time.time()

        module_root = os.path.abspath(os.path.expanduser(module_path))
        cpu_dir = os.path.abspath(os.path.expanduser(config_path))
        bmodel_dir = os.path.abspath(os.path.expanduser(model_path))

        if module_root not in sys.path:
            sys.path.insert(0, module_root)

        bmodel = os.path.join(bmodel_dir, "encoder_full.bmodel")
        if not os.path.exists(bmodel):
            bmodel = os.path.join(module_root, "output", "encoder_full.bmodel")
        if not os.path.exists(cpu_dir):
            cpu_dir = os.path.join(module_root, "cpu_components")

        logger.info("  bmodel:  %s", bmodel)
        logger.info("  cpu_dir: %s", cpu_dir)
        logger.info("  module:  %s", module_root)

        self.model, self.tokenizer = _load_cpu_model_and_tokenizer(
            module_root, cpu_dir
        )
        logger.info("  CPU components loaded (%dms)", int((time.time() - t0) * 1000))

        t1 = time.time()
        import sophon.sail as sail
        self.sail_engine = sail.Engine(bmodel, devid, sail.SYSIO)
        logger.info("  TPU bmodel loaded (%dms)", int((time.time() - t1) * 1000))

        self._ready = True
        logger.info("  Total load time: %dms", int((time.time() - t0) * 1000))

    def deinit(self):
        """Release TPU and CPU resources."""
        self._ready = False
        self.model = None
        self.tokenizer = None
        self.sail_engine = None
        gc.collect()
        logger.info("Dolphin-CN model released.")

    @property
    def is_ready(self):
        return self._ready

    # -- audio loading -------------------------------------------------

    @staticmethod
    def load_audio_to_pcm(audio_path):
        """Load any audio file to 16kHz int16 mono PCM."""
        data, sr = sf.read(audio_path, dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            data = resample(data, int(len(data) * SAMPLE_RATE / sr))
        pcm = (data * 32767).clip(-32768, 32767).astype(np.int16)
        return pcm

    @staticmethod
    def load_pcm_from_bytes(data):
        """Load audio bytes (any format) to 16kHz int16 mono PCM."""
        import io
        bio = io.BytesIO(data)
        data_f, sr = sf.read(bio, dtype='float64')
        if data_f.ndim > 1:
            data_f = data_f.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            data_f = resample(data_f, int(len(data_f) * SAMPLE_RATE / sr))
        pcm = (data_f * 32767).clip(-32768, 32767).astype(np.int16)
        return pcm

    # -- inference ----------------------------------------------------

    def transcribe(self, pcm):
        """Run streaming Dolphin inference on int16 PCM array, return text."""
        from dolphin.search import ctc_greedy_search

        N = NUM_LAYERS
        ak = np.zeros((N, 1, NUM_HEADS, MAX_CACHE_T, D_K * 2), dtype=np.float32)
        ck = np.zeros((N, 1, MODEL_DIM * 2, 30), dtype=np.float32)
        offset = 0
        prev_tokens = None
        last_partial_text = ""
        idx = 0

        while idx < len(pcm):
            chunk = pcm[idx:idx + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            wav = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                feats = torchaudio.compliance.kaldi.fbank(
                    waveform=wav, num_mel_bins=80, frame_length=25,
                    frame_shift=10, dither=0.0, sample_frequency=SAMPLE_RATE,
                ).unsqueeze(0)

            with torch.no_grad():
                xs = self.model.encoder.global_cmvn(feats)

            fn = xs.unsqueeze(1).cpu().numpy().astype(np.float32)
            out = self.sail_engine.process('embed_conv', {'x.1': fn})
            cv = out['28']
            ct = np.transpose(cv, (0, 2, 1, 3)).reshape(
                1, CHUNK_SIZE, -1).astype(np.float32)

            out = self.sail_engine.process('embed_linear', {'x.1': ct})
            xs_lin = torch.from_numpy(out['4'])

            pw = self.model.encoder.embed.position_encoding(
                offset=0, size=CHUNK_SIZE)
            pc = self.model.encoder.embed.position_encoding(
                offset=offset, size=CHUNK_SIZE)
            xs = xs_lin * math.sqrt(MODEL_DIM) + pc

            t_out = xs.size(1)
            aks = MAX_CACHE_T + t_out
            et = min(offset, MAX_CACHE_T)
            vs = MAX_CACHE_T - et

            pe_enc = self.model.encoder.embed.position_encoding(
                offset=offset - et, size=aks)
            am = torch.full((1, t_out, aks), float('-inf'), dtype=torch.float32)
            for i in range(t_out):
                am[0, i, vs:MAX_CACHE_T + i + 1] = 0.0

            xn = xs.cpu().numpy().astype(np.float32)
            an = am.cpu().numpy().astype(np.float32)
            pn = pe_enc.cpu().numpy().astype(np.float32)

            for li in range(N):
                aksl = ak[li]
                inp = {
                    'x.5': xn,
                    'att_mask.1': an,
                    'pos_emb.1': pn,
                    'k_cache.1': aksl[:, :, :, :D_K],
                    'v_cache.1': aksl[:, :, :, D_K:],
                    'cnn_cache.1': ck[li],
                }
                out = self.sail_engine.process(GRAPH_NAMES[li], inp)
                xn = out['320']
                nk = np.concatenate([out['52'], out['53']], axis=-1)
                if nk.shape[2] > MAX_CACHE_T:
                    nk = nk[:, :, nk.shape[2] - MAX_CACHE_T:, :]
                ak[li] = nk
                ck[li] = out['58']

            xs_out = torch.from_numpy(xn)
            offset += t_out

            new_enc = xs_out
            with torch.no_grad():
                if (self.model.encoder.normalize_before
                        and self.model.encoder.final_norm):
                    new_enc = self.model.encoder.after_norm(new_enc)

            cn = new_enc.cpu().numpy().astype(np.float32)
            out = self.sail_engine.process('ctc_proj', {'x.1': cn})
            cp = torch.from_numpy(out['7']).log_softmax(dim=2)
            cl = torch.tensor([cp.size(1)])

            r = ctc_greedy_search(cp, cl)
            new_tokens = r[0].tokens[:int(cl[0])]

            if prev_tokens is None:
                prev_tokens = []
            if prev_tokens and new_tokens and new_tokens[0] == prev_tokens[-1]:
                combined = prev_tokens + new_tokens[1:]
            else:
                combined = prev_tokens + new_tokens
            prev_tokens = combined

            text, _ = self.tokenizer.detokenize(combined)
            last_partial_text = text

            idx += STRIDE_SAMPLES

        logger.info("ASR done: %d chars", len(last_partial_text))
        return last_partial_text


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def _deinit_model():
    global engine_obj
    if engine_obj is not None:
        engine_obj.deinit()
        engine_obj = None


@asynccontextmanager
async def lifespan(fastapi_app):
    global server_state
    asyncio.get_event_loop().create_task(_async_load())
    yield
    _deinit_model()


app = FastAPI(lifespan=lifespan)


async def _async_load():
    global server_state, engine_obj
    cfg = app.state.cfg
    engine_obj = DolphinInferenceEngine()
    loop = asyncio.get_event_loop()
    try:
        server_state = "loading"
        await loop.run_in_executor(
            executor,
            engine_obj.load,
            cfg["model_path"],
            cfg["config_path"],
            cfg["module_path"],
            cfg.get("devid", 0),
        )
        server_state = "ready"
        logger.info("Dolphin-CN ASR server ready.")
    except Exception as e:
        server_state = "error"
        logger.error("Failed to load Dolphin-CN model: %s", e)
        import traceback
        traceback.print_exc()


# -- REST endpoints ----------------------------------------------------

@app.get("/health")
async def health():
    return {"status": server_state}


@app.get("/status")
async def status_detail():
    return {
        "status": server_state,
        "model": "dolphin-cn",
        "type": "audio",
    }


@app.post("/shutdown")
async def do_shutdown():
    _deinit_model()
    logger.info("Shutting down...")
    asyncio.get_event_loop().call_later(1.0, lambda: os._exit(0))
    return JSONResponse({"status": "shutting_down"})


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(None),
    model: str = Form("dolphin-asr"),
    language: str = Form(None),
    request: Request = None,
):
    """OpenAI-compatible audio transcription endpoint.

    Accepts:
      - multipart/form-data with ``file`` field (standard OpenAI format)
      - application/json with ``{"audio": "<base64>", "model": "dolphin-asr"}``
    """
    global engine_obj, server_state

    if server_state != "ready" or engine_obj is None or not engine_obj.is_ready:
        return JSONResponse(
            {"error": {"message": "Server not ready, current state: %s" % server_state,
                       "type": "server_error"}},
            status_code=503,
        )

    content_type = ""
    if request:
        content_type = request.headers.get("content-type", "")

    loop = asyncio.get_event_loop()

    # -- JSON body (base64) ---------------------------------
    if content_type.startswith("application/json") or (file is None and request is not None):
        try:
            body = await request.body()
            data = json.loads(body)
            b64_audio = data.get("audio", "")
            if not b64_audio:
                return JSONResponse(
                    {"error": {"message": "Missing 'audio' field (base64)",
                               "type": "invalid_request_error"}},
                    status_code=400,
                )
            audio_bytes = base64.b64decode(b64_audio)
            pcm = await loop.run_in_executor(
                executor, engine_obj.load_pcm_from_bytes, audio_bytes)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                {"error": {"message": "Failed to decode base64 audio: %s" % str(e),
                           "type": "server_error"}},
                status_code=400,
            )

    # -- multipart file upload -------------------------------
    elif file is not None:
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        if not suffix:
            suffix = ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()
            pcm = await loop.run_in_executor(
                executor, engine_obj.load_audio_to_pcm, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    else:
        return JSONResponse(
            {"error": {"message": "Provide either 'file' (multipart) or 'audio' (JSON base64)",
                       "type": "invalid_request_error"}},
            status_code=400,
        )

    # -- Run transcription ----------------------------------
    try:
        text = await loop.run_in_executor(executor, engine_obj.transcribe, pcm)
        return JSONResponse({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


# ---------------------------------------------------------------------------

# -- Streaming endpoint -------------------------------------------------

@app.post("/v1/audio/transcriptions/stream")
async def audio_transcriptions_stream(request: Request):
    """Streaming ASR endpoint for real-time recognition.

    Accepts JSON body:
      {"audio": "<base64_int16_pcm>", "session_id": "<uuid_or_null>",
       "hotwords": ["word1", ...], "finalize": false, "reset": false}

    Returns JSON:
      {"text": "...", "session_id": "<uuid>", "is_final": false,
       "vad": "speech|silence", "new_partial": "..."}
    """
    global engine_obj, server_state

    if server_state != "ready" or engine_obj is None or not engine_obj.is_ready:
        return JSONResponse(
            {"error": {"message": "Server not ready, state: %s" % server_state,
                       "type": "server_error"}},
            status_code=503,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body",
                       "type": "invalid_request_error"}},
            status_code=400,
        )

    b64_audio = body.get("audio", "")
    session_id = body.get("session_id")
    hotwords = body.get("hotwords", [])
    finalize = body.get("finalize", False)
    reset = body.get("reset", False)

    # Periodic stale session eviction
    if len(streaming_sessions) > 20 and len(streaming_sessions) % 10 == 0:
        _evict_stale_sessions()

    # Create or retrieve session
    if session_id is None or session_id not in streaming_sessions:
        session_id = str(uuid.uuid4())
        session = StreamingSession()
        streaming_sessions[session_id] = session
    else:
        session = streaming_sessions[session_id]

    session.last_activity = time.time()

    # Update hotwords
    if hotwords:
        session.hotwords = list(hotwords)
        session.hit_hotwords = set()
        if engine_obj.tokenizer is not None:
            session.hw_bias = _build_hotword_bias(engine_obj.tokenizer, hotwords)
        else:
            session.hw_bias = None

    # Reset stream state
    if reset:
        session.att_cache = np.zeros(
            (NUM_LAYERS, 1, NUM_HEADS, MAX_CACHE_T, D_K * 2), dtype=np.float32)
        session.cnn_cache = np.zeros(
            (NUM_LAYERS, 1, MODEL_DIM * 2, 30), dtype=np.float32)
        session.offset = 0
        session.prev_tokens = None
        session.last_partial_text = ""
        session.chunk_count = 0
        session.hit_hotwords = set()
        logger.debug("Session %s reset", session_id[:8])
        if not b64_audio and not finalize:
            return JSONResponse({
                "text": "", "session_id": session_id,
                "is_final": False, "vad": "silence", "new_partial": "",
            })

    # Finalize
    if finalize:
        text = session.last_partial_text
        logger.info("Session %s finalized: %d chars, %d chunks",
                     session_id[:8], len(text), session.chunk_count)
        del streaming_sessions[session_id]
        return JSONResponse({
            "text": text, "session_id": session_id,
            "is_final": True, "vad": "silence", "new_partial": "",
            "chunks": session.chunk_count,
        })

    # Process chunk
    if not b64_audio:
        return JSONResponse(
            {"error": {"message": "Missing 'audio' field (base64)",
                       "type": "invalid_request_error"}},
            status_code=400,
        )

    try:
        audio_bytes = base64.b64decode(b64_audio)
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    except Exception as e:
        return JSONResponse(
            {"error": {"message": "Failed to decode base64 audio: %s" % str(e),
                       "type": "server_error"}},
            status_code=400,
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, _process_stream_chunk, session, pcm, engine_obj)

    # Auto-reset: silence timeout + max offset safety
    auto_finalize = False
    if session.silence_ms >= SILENCE_TIMEOUT_MS and session.had_speech:
        auto_finalize = True
        logger.info("Session %s auto-finalized after %dms silence",
                     session_id[:8], session.silence_ms)
    if session.offset >= MAX_SAFE_OFFSET:
        auto_finalize = True
        logger.warning("Session %s force-reset at offset %d",
                       session_id[:8], session.offset)

    if auto_finalize:
        final_text = session.last_partial_text
        session.att_cache = np.zeros(
            (NUM_LAYERS, 1, NUM_HEADS, MAX_CACHE_T, D_K * 2), dtype=np.float32)
        session.cnn_cache = np.zeros(
            (NUM_LAYERS, 1, MODEL_DIM * 2, 30), dtype=np.float32)
        session.offset = 0
        session.prev_tokens = None
        session.last_partial_text = ""
        session.chunk_count = 0
        session.silence_ms = 0
        session.had_speech = False
        session.hit_hotwords = set()
        session.reset_count += 1
        return JSONResponse({
            "text": final_text,
            "session_id": session_id,
            "is_final": True,
            "vad": "silence",
            "new_partial": "",
            "hotword_hits": [],
        })

    return JSONResponse({
        "text": result.get("full_text", ""),
        "session_id": session_id,
        "is_final": False,
        "vad": result.get("vad", "silence"),
        "new_partial": result.get("new_partial", ""),
        "hotword_hits": result.get("hotword_hits", []),
    })

# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model-path", type=str,
                        default="../Dolphin_CN_Streaming/output")
    parser.add_argument("--config-path", type=str,
                        default="../Dolphin_CN_Streaming/cpu_components")
    parser.add_argument("--module-path", type=str,
                        default="../Dolphin_CN_Streaming")
    parser.add_argument("--devid", type=int, default=0)
    parser.add_argument("--language", type=str, default=None)
    args = parser.parse_args()

    app.state.cfg = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "module_path": args.module_path,
        "devid": args.devid,
        "language": args.language,
    }
    os.environ["MODULE_PATH"] = args.module_path

    atexit.register(_deinit_model)

    uvicorn.run(app, host=args.host, port=args.port, log_config={
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO",
                        "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO",
                              "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING",
                               "propagate": False},
        },
    })
