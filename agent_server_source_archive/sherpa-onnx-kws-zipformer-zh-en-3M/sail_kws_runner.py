#!/usr/bin/env python3
"""Minimal BM1684X SAIL runner for the bundled sherpa Zipformer KWS graph.

The bmodel contains encoder, decoder and joiner graphs.  Feature extraction is
kept on CPU (80-bin Kaldi-style log fbank); neural graph execution is on TPU.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from sophon import sail

# The extension is deliberately local to this deployable directory.  It ports
# Sherpa's CPU-side ContextGraph/modified-beam state machine; SAIL still runs
# every neural graph invocation on the TPU.
_SEARCH_DIR = Path(__file__).resolve().parent / "sail_keyword_search"
if str(_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_SEARCH_DIR))
try:
    from sail_keyword_search import KeywordSearch
except ImportError as exc:
    raise RuntimeError(
        "Missing sail_keyword_search extension. Build it with "
        "/data/env310/bin/python sail_keyword_search/setup.py build_ext --inplace"
    ) from exc


class SailKwsRunner:
    CHUNK_FRAMES = 45
    # The encoder consumes a 45-frame look-ahead/left-context window and
    # emits eight 40-ms states, i.e. 32 newly consumed 10-ms fbank frames.
    # `processed_lens` advances by 16 because it is in the encoder's
    # subsampled units, not because the input fbank hop is 16.
    CHUNK_SHIFT_FRAMES = 32
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 400
    FRAME_SHIFT = 160
    N_MELS = 80
    BLANK = 0
    KEYWORD_SCORE = 2.0
    KEYWORD_THRESHOLD = 0.18
    NUM_TRAILING_BLANKS = 1
    MAX_ACTIVE_PATHS = 4

    def __init__(self, model_path: str | Path, tokens_path: str | Path, device_id: int = 0):
        self.engine = sail.Engine(str(model_path), device_id, sail.IOMode.SYSIO)
        self.enc_graph = "kws_encoder_chunk16"
        self.dec_graph = "kws_decoder_chunk16"
        self.join_graph = "kws_joiner_chunk16"
        graphs = set(self.engine.get_graph_names())
        missing = {self.enc_graph, self.dec_graph, self.join_graph} - graphs
        if missing:
            raise RuntimeError(f"KWS bmodel missing graphs: {sorted(missing)}")
        self.tokens = self._load_tokens(tokens_path)
        self.keyword_tokens = self._load_keyword_tokens(Path(tokens_path).parent / "keywords.txt")
        self.keyword_search = (
            KeywordSearch(
                list(self.keyword_tokens), self.KEYWORD_SCORE, self.KEYWORD_THRESHOLD,
                self.MAX_ACTIVE_PATHS, self.NUM_TRAILING_BLANKS,
            ) if self.keyword_tokens else None
        )
        self.reset()

    @staticmethod
    def _load_tokens(path: str | Path) -> Dict[int, str]:
        result: Dict[int, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[int(parts[1])] = parts[0]
        return result

    def _load_keyword_tokens(self, path: Path) -> Tuple[int, ...]:
        """Load the production phrase from sherpa's tokenized keywords file.

        The target runtime deliberately ships the phrase ``小麦小麦`` as phone
        tokens.  Keeping it next to ``tokens.txt`` makes the TPU bundle
        self-contained; no files are read from the old CPU reference project.
        """
        if not path.is_file():
            return ()
        for line in path.read_text(encoding="utf-8").splitlines():
            phones = line.split(":", 1)[0].strip().split()
            ids = [next((i for i, token in self.tokens.items() if token == p), None) for p in phones]
            if phones and all(i is not None for i in ids):
                return tuple(ids)  # type: ignore[arg-type]
        return ()

    def reset(self) -> None:
        self.cache: Dict[str, np.ndarray] = {}
        for name in self.engine.get_input_names(self.enc_graph):
            shape = self.engine.get_input_shape(self.enc_graph, name)
            dtype = np.int32 if name == "processed_lens" else np.float32
            self.cache[name] = np.zeros(shape, dtype=dtype)
        self.processed_samples = 0
        self.pending = np.empty(0, np.float32)
        # For snip_edges=False, the first frame of every chunk after the
        # first must see actual samples immediately preceding the chunk.  Keep
        # one 10 ms shift so local fbank frame 1 equals the next global frame.
        self.fbank_left = np.empty(0, np.float32)
        self.keyword_dec_cache: Dict[Tuple[int, int], np.ndarray] = {}
        if self.keyword_search is not None:
            self.keyword_search.reset()
        self.pending_keyword_tokens: List[int] = []

    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        """Match sherpa-onnx 1.13.4's 80-bin online fbank contract.

        The previous NumPy approximation used a different mel range, window
        and frame boundary rule.  Kaldi's Povey-window implementation here
        matches the compiler golden while keeping feature extraction on CPU.
        """
        if len(samples) < self.FRAME_LENGTH:
            samples = np.pad(samples, (0, self.FRAME_LENGTH - len(samples)))
        try:
            import torch
            from torchaudio.compliance.kaldi import fbank
        except ImportError as exc:
            raise RuntimeError(
                "KWS requires torch and torchaudio for sherpa-compatible fbank. "
                "Use the project runtime: /data/env310/bin/python"
            ) from exc
        features = fbank(
            torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).reshape(1, -1),
            sample_frequency=float(self.SAMPLE_RATE),
            num_mel_bins=self.N_MELS,
            frame_length=25.0,
            frame_shift=10.0,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,
            snip_edges=False,
            remove_dc_offset=True,
            preemphasis_coefficient=0.97,
            window_type="povey",
            use_energy=False,
            round_to_power_of_two=True,
        )
        return features.numpy().astype(np.float32, copy=False)

    def _encode(self, feats: np.ndarray) -> np.ndarray:
        inp = dict(self.cache)
        inp["x"] = feats[None, :, :].astype(np.float32)
        out = self.engine.process(self.enc_graph, inp)
        names = self.engine.get_output_names(self.enc_graph)
        for in_name, out_name in zip(self.engine.get_input_names(self.enc_graph)[1:], names[1:]):
            self.cache[in_name] = np.asarray(out[out_name])
        self.cache["processed_lens"] = np.asarray(out[names[-1]], dtype=np.int32)
        return np.asarray(out[names[0]], dtype=np.float32)[0]

    def _keyword_logits(self, enc: np.ndarray, history: Tuple[int, int]) -> np.ndarray:
        dec = self.keyword_dec_cache.get(history)
        if dec is None:
            dec = np.asarray(
                self.engine.process(self.dec_graph, {"y": np.asarray(history, np.int32)[None, :]})[
                    "decoder_out_Gemm"
                ],
                dtype=np.float32,
            )
            self.keyword_dec_cache[history] = dec
        logits = np.asarray(
            self.engine.process(
                self.join_graph,
                {"encoder_out": enc[None, :].astype(np.float32), "decoder_out": dec},
            )["logit_Gemm"][0],
            dtype=np.float32,
        )
        return logits

    def _keyword_step(self, enc: np.ndarray) -> bool:
        """Evaluate active prediction paths on TPU, then advance C++ search."""
        if self.keyword_search is None:
            return False
        histories = self.keyword_search.histories()
        logits = np.stack(
            [self._keyword_logits(enc, (int(y[0]), int(y[1]))) for y in histories], axis=0
        )
        detected = bool(self.keyword_search.step(logits))
        if detected:
            # A accepted Sherpa phrase resets all hypotheses.  Decoder cache
            # remains valid as a cache keyed by two-token history.
            self.keyword_dec_cache.clear()
        return detected

    def accept_waveform(self, samples: np.ndarray) -> List[int]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.pending = np.concatenate((self.pending, samples))
        new: List[int] = self.pending_keyword_tokens
        self.pending_keyword_tokens = []
        need = (self.CHUNK_FRAMES - 1) * self.FRAME_SHIFT + self.FRAME_LENGTH
        while len(self.pending) >= need:
            raw = self.pending[:need]
            consumed = self.CHUNK_SHIFT_FRAMES * self.FRAME_SHIFT
            if len(self.fbank_left):
                feats = self._fbank(np.concatenate((self.fbank_left, raw)))[1:self.CHUNK_FRAMES + 1]
            else:
                feats = self._fbank(raw)[:self.CHUNK_FRAMES]
            self.fbank_left = raw[consumed - self.FRAME_SHIFT:consumed].copy()
            self.pending = self.pending[consumed:]
            for enc in self._encode(feats):
                if self.keyword_tokens:
                    if self._keyword_step(enc):
                        new.extend(self.keyword_tokens)
                    continue
        self.processed_samples += len(samples)
        return new

    def accept_wave_file(self, path: str | Path) -> List[int]:
        audio, rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != self.SAMPLE_RATE:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(int(rate), self.SAMPLE_RATE)
            audio = resample_poly(audio, self.SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
        return self.accept_waveform(audio)
