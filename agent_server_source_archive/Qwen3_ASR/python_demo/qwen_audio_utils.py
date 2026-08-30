"""Small, model-independent helpers for Qwen3-ASR audio preparation."""
from __future__ import annotations

import numpy as np


# This bmodel advertises a 256-token text input, but the BM1684X runtime starts
# producing invalid output (and may report bm_memcpy_s2d_partial_offset errors)
# at six one-second audio invocations.  Five blocks have been verified stable.
QWEN3_ASR_SAFE_AUDIO_SECONDS = 5


def trim_to_whole_seconds_preserving_tail(
    audio,
    sample_rate: int,
    max_seconds: int = QWEN3_ASR_SAFE_AUDIO_SECONDS,
):
    """Keep complete TPU blocks while preserving the end of the utterance.

    The compiled audio graph accepts one exact second per invocation.  Growing
    a recording to the next second creates an extra device invocation and can
    overflow fixed buffers.  We instead discard only the oldest fractional
    second, which belongs to the pre-roll, and retain the command tail.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    complete_samples = (len(audio) // sample_rate) * sample_rate
    if complete_samples <= 0:
        return np.pad(audio, (0, sample_rate - len(audio)), mode="constant")
    safe_samples = min(complete_samples, sample_rate * max_seconds)
    if safe_samples == len(audio):
        return audio
    return audio[-safe_samples:]
