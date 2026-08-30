#!/usr/bin/env python3
"""Generate golden or full-corpus vectors with the full Torch model on CUDA."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def records(path: Path, text_field: str):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                yield record, str(record[text_field])


def batches(iterator, size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="full Qwen3-Embedding-0.6B model directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="bundled tokenizer directory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-field", default="text", choices=("text", "question"))
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 32 <= args.dimensions <= 1024:
        parser.error("--dimensions must be in [32, 1024]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    rows = count_rows(args.input)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), padding_side="left")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenization_rows = []
    if rows <= 1000:
        for index, (record, text) in enumerate(records(args.input, args.text_field)):
            encoded_row = tokenizer(text, truncation=True, max_length=512)["input_ids"]
            tokenization_rows.append({
                "row_index": index,
                "case_id": record.get("case_id"),
                "tokens_embedded": len(encoded_row),
                "embedded_input_ids_sha256": hashlib.sha256(
                    np.asarray(encoded_row, dtype="<i4").tobytes()
                ).hexdigest(),
            })
    model = AutoModel.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=(rows, args.dimensions)
    )
    cursor = 0
    with torch.inference_mode():
        for batch in batches(records(args.input, args.text_field), args.batch_size):
            texts = [text for _, text in batch]
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            # Left padding places the final valid token at index -1 for every row.
            vectors = hidden[:, -1, : args.dimensions].float()
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            array = vectors.cpu().numpy().astype(np.float32, copy=False)
            output[cursor : cursor + len(batch)] = array
            cursor += len(batch)
            if cursor % max(args.batch_size, 1024) == 0 or cursor == rows:
                print(json.dumps({"completed": cursor, "total": rows}))
    output.flush()
    del output

    metadata = {
        "format": "qwen3_embedding_torch_vectors_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "text_field": args.text_field,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "torch_dtype": "bfloat16",
        "attention_implementation": "eager",
        "padding_side": "left",
        "max_length": 512,
        "pooling": "last hidden-state token at index -1",
        "dimensions": args.dimensions,
        "normalization": "L2 after MRL prefix selection",
        "tokenization_rows": tokenization_rows,
        "shape": [rows, args.dimensions],
        "dtype": "float32",
        "output_sha256": sha256_file(args.output),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(torch.device(args.device)),
        },
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
