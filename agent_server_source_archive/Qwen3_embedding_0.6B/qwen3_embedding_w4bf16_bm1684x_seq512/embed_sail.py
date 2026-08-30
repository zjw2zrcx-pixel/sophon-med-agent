#!/usr/bin/env python3
"""Minimal single-text Qwen3-Embedding inference for the exported BM1684X bmodel."""

import argparse
import json
from pathlib import Path

import numpy as np
import sophon.sail as sail
from transformers import AutoTokenizer


class Qwen3Embedding:
    """Runs the static seq=512 bmodel and returns an MRL-prefix embedding."""

    def __init__(self, bmodel_path: Path, tokenizer_path: Path, device_id: int = 0):
        self.engine = sail.Engine(str(bmodel_path), device_id, sail.IOMode.SYSIO)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), padding_side="left")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.graphs = self.engine.get_graph_names()
        required = ["embedding"] + [f"block_{i}" for i in range(28)]
        missing = [name for name in required if name not in self.graphs]
        if missing:
            raise RuntimeError(f"bmodel is missing graphs: {missing}")

        input_name = self.engine.get_input_names("embedding")[0]
        self.seq_length = self.engine.get_input_shape("embedding", input_name)[1]
        if self.seq_length != 512:
            raise RuntimeError(f"this test script expects seq=512, got {self.seq_length}")

        # BF16 bmodels must use the explicit Tensor-map API with this SAIL
        # binding.  The numpy-dict overload attempts to materialize BF16 as a
        # normal numpy dtype and can abort in native code.
        self.embed_input = self.engine.create_input_tensors_map("embedding")
        self.embed_output = self.engine.create_output_tensors_map("embedding")
        self.block_inputs = {}
        self.block_outputs = {}
        for layer in range(28):
            graph = f"block_{layer}"
            self.block_inputs[graph] = self.engine.create_input_tensors_map(graph)
            self.block_outputs[graph] = self.engine.create_output_tensors_map(graph)

    @staticmethod
    def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
        """Encode float32 values as the uint16 BF16 bit representation."""
        return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)

    @staticmethod
    def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
        """Decode SAIL's uint16 BF16 system tensor representation."""
        return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)

    def _attention_mask(self, valid_tokens: int) -> np.ndarray:
        """Causal mask for a left-padded single sequence, 0=attend, -10000=mask."""
        seq = self.seq_length
        start = seq - valid_tokens
        mask = np.full((1, 1, seq, seq), -10000.0, dtype=np.float32)
        for query in range(start, seq):
            mask[0, 0, query, start:query + 1] = 0.0
        return mask

    def encode(self, text: str, dimensions: int = 256) -> np.ndarray:
        if not 32 <= dimensions <= 1024:
            raise ValueError("dimensions must be in [32, 1024]")
        batch = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
            return_tensors="np",
        )
        input_ids = batch["input_ids"].astype(np.int32)
        valid_tokens = int(batch["attention_mask"].sum())
        if valid_tokens == 0:
            raise ValueError("tokenizer produced no tokens")

        position_ids = np.zeros((1, self.seq_length), dtype=np.int32)
        position_ids[0, self.seq_length - valid_tokens:] = np.arange(valid_tokens, dtype=np.int32)
        mask = self._attention_mask(valid_tokens)

        self.embed_input["input_ids"].update_data(input_ids)
        self.engine.process("embedding", self.embed_input, self.embed_output)
        hidden = self._bf16_to_f32(self.embed_output["embedding"].asnumpy())

        for layer in range(28):
            graph = f"block_{layer}"
            inputs = self.block_inputs[graph]
            outputs = self.block_outputs[graph]
            inputs["input_states"].update_data(self._f32_to_bf16(hidden))
            inputs["position_ids"].update_data(position_ids)
            inputs["attention_mask"].update_data(self._f32_to_bf16(mask))
            self.engine.process(graph, inputs, outputs)
            hidden = self._bf16_to_f32(outputs["output_states"].asnumpy())

        # Left padding makes the final real token always occupy index -1.
        vector = hidden[0, -1, :dimensions].astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-12  # normalize AFTER MRL truncation
        return vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="text to embed")
    parser.add_argument("--bmodel", default="model/qwen3_embedding_w4bf16_seq512_bm1684x.bmodel")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dimensions", type=int, default=256)
    args = parser.parse_args()

    encoder = Qwen3Embedding(Path(args.bmodel), Path(args.tokenizer), args.device_id)
    vector = encoder.encode(args.text, args.dimensions)
    print(json.dumps({"dimensions": int(vector.size), "embedding": vector.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
