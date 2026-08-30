"""Small decoder-only throughput probe for a history BModel.

This intentionally excludes tokenizer and prefill timing.  It performs one
prefill for each requested context, then times a short sequence of
forward_next calls.
"""

import argparse
import time
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--devid", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[128, 512, 1024])
    args = parser.parse_args()

    sys.path.insert(0, args.module_path)
    from pipeline import Qwen3_5

    class ModelArgs:
        devid = args.devid
        video_ratio = 0.25
        max_new_tokens = args.tokens
        model_path = args.model_path
        config_path = args.config_path

    pipeline = Qwen3_5(ModelArgs())
    model = pipeline.model
    print(
        "model_ready",
        "support_history=", pipeline.support_history,
        "SEQLEN=", model.SEQLEN,
        "MAX_INPUT_LENGTH=", model.MAX_INPUT_LENGTH,
        "PREFILL_KV_LENGTH=", model.PREFILL_KV_LENGTH,
        flush=True,
    )

    # Repeating ordinary token IDs avoids including tokenization/prompt-format
    # cost in the decoder measurement.  A real prefill is still performed.
    seed = pipeline.tokenizer.encode(
        "请保持稳定地完成当前任务。", add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer returned no seed tokens")

    try:
        for requested_context in args.contexts:
            context = min(
                int(requested_context),
                int(model.SEQLEN) - int(args.tokens) - 2,
            )
            tokens = (seed * ((context + len(seed) - 1) // len(seed)))[:context]
            token_array = np.asarray(tokens, dtype=np.int32).reshape(1, -1)

            model.clear_history()
            model.forward_embed(token_array)
            positions = np.arange(context, dtype=np.int32)
            first_token = model.forward_first(np.tile(positions, 3))

            decode_count = min(
                int(args.tokens), int(model.SEQLEN) - context - 1)
            started = time.perf_counter()
            token = int(first_token)
            for index in range(decode_count):
                position = np.asarray(
                    [context + index] * 3, dtype=np.int32)
                token = model.forward_next(position)
            elapsed = time.perf_counter() - started
            print(
                "context=%d decode_tokens=%d elapsed_ms=%.3f ms_per_token=%.3f "
                "tokens_per_sec=%.3f last_token=%d history_length=%d" % (
                    context, decode_count, elapsed * 1000,
                    elapsed * 1000 / decode_count,
                    decode_count / elapsed if elapsed else 0.0,
                    int(token), int(model.history_length)),
                flush=True,
            )
    finally:
        model.deinit()


if __name__ == "__main__":
    main()
