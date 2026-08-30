#!/usr/bin/env python3
"""End-to-end TPU KWS regression for the bundled ``小麦小麦`` phrase.

This never opens an ONNX model.  It exercises CPU fbank, the local C++ port
of Sherpa's keyword search, and all three SAIL bmodel graphs.
"""
from pathlib import Path

import numpy as np

from sail_kws_runner import SailKwsRunner


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "kws_transducer_chunk16_bm1684x_f32.bmodel"
TOKENS = ROOT / "runtime" / "tokens.txt"
GOLDEN_AUDIO = ROOT / "golden" / "test_wav_first_chunk" / "resampled_16k_mono.npy"


def detect(samples: np.ndarray) -> list[float]:
    runner = SailKwsRunner(MODEL, TOKENS)
    original = runner._keyword_step
    encoder_frame = 0
    hit_frames: list[int] = []

    def step(enc: np.ndarray) -> bool:
        nonlocal encoder_frame
        encoder_frame += 1
        matched = original(enc)
        if matched:
            hit_frames.append(encoder_frame)
        return matched

    runner._keyword_step = step  # type: ignore[method-assign]
    runner.accept_waveform(samples)
    # The encoder emits one state per 40 ms (8 states for each 32 fbank hop).
    return [frame * 0.04 for frame in hit_frames]


def require_three(label: str, hits: list[float]) -> None:
    print(f"{label}: {len(hits)} full-phrase hits at " + ", ".join(f"{x:.2f}s" for x in hits))
    if len(hits) != 3:
        raise SystemExit(f"FAIL: expected exactly 3 小麦小麦 hits for {label}")


def main() -> None:
    require_three("golden 16 kHz audio", detect(np.load(GOLDEN_AUDIO)))
    # Deliberately use the public file API too: test.wav is 44.1 kHz.
    runner = SailKwsRunner(MODEL, TOKENS)
    original = runner._keyword_step
    frame = 0
    hits: list[float] = []

    def step(enc: np.ndarray) -> bool:
        nonlocal frame
        frame += 1
        matched = original(enc)
        if matched:
            hits.append(frame * 0.04)
        return matched

    runner._keyword_step = step  # type: ignore[method-assign]
    runner.accept_wave_file(ROOT / "test.wav")
    require_three("test.wav", hits)
    print("OK: bmodel TPU KWS matches the three-phrase regression target")


if __name__ == "__main__":
    main()
