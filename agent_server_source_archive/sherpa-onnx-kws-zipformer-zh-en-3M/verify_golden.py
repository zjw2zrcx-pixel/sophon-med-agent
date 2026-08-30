#!/usr/bin/env python3
"""Verify the local Sherpa KWS frontend and TPU bmodel against compiler golden."""

from pathlib import Path

import numpy as np
from sophon import sail

from sail_kws_runner import SailKwsRunner


ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden" / "test_wav_first_chunk"
ATOL = 1e-4


def compare(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    delta = np.abs(actual - expected)
    maximum = float(delta.max())
    print(f"{name}: max_abs={maximum:.8g} mean_abs={float(delta.mean()):.8g}")
    if maximum > ATOL:
        raise SystemExit(f"FAIL: {name} exceeds atol={ATOL}")


def main() -> None:
    required = [
        GOLDEN / "fbank_45.npy",
        GOLDEN / "encoder_decoder_joiner_golden.npz",
        GOLDEN / "resampled_16k_mono.npy",
        ROOT / "kws_transducer_chunk16_bm1684x_f32.bmodel",
        ROOT / "runtime" / "tokens.txt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Missing golden/runtime files:\n" + "\n".join(missing))

    reference = np.load(GOLDEN / "encoder_decoder_joiner_golden.npz")
    audio = np.load(GOLDEN / "resampled_16k_mono.npy")
    expected_fbank = np.load(GOLDEN / "fbank_45.npy")

    runner = SailKwsRunner(
        ROOT / "kws_transducer_chunk16_bm1684x_f32.bmodel", ROOT / "runtime" / "tokens.txt"
    )
    actual_fbank = runner._fbank(audio[:7440])[:45][None, :, :]
    compare("fbank", actual_fbank, expected_fbank)

    engine = sail.Engine(
        str(ROOT / "kws_transducer_chunk16_bm1684x_f32.bmodel"), 0, sail.IOMode.SYSIO
    )
    encoder_inputs = {}
    for name in engine.get_input_names(runner.enc_graph):
        value = reference[f"encoder_input_{name}"]
        encoder_inputs[name] = value.astype(np.int32 if name == "processed_lens" else np.float32)
    encoder_out = engine.process(runner.enc_graph, encoder_inputs)["encoder_out_Add"]
    compare("encoder_out", np.asarray(encoder_out), reference["encoder_out"])

    decoder_out = engine.process(
        runner.dec_graph, {"y": reference["decoder_y"].astype(np.int32)}
    )["decoder_out_Gemm"]
    compare("decoder_out", np.asarray(decoder_out), reference["decoder_out"])

    joiner_out = engine.process(
        runner.join_graph,
        {
            "encoder_out": reference["joiner_encoder_out_t0"].astype(np.float32),
            "decoder_out": np.asarray(decoder_out, dtype=np.float32),
        },
    )["logit_Gemm"]
    compare("joiner_logits_t0", np.asarray(joiner_out), reference["joiner_logits_t0"])
    print("OK: frontend and all TPU graphs match compiler golden")


if __name__ == "__main__":
    main()
