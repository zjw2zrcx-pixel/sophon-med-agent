# Qwen3-Embedding-0.6B W4BF16 remote test bundle

## Contents

- `model/qwen3_embedding_w4bf16_seq512_bm1684x.bmodel`: BM1684X, static batch 1 / sequence 512, W4BF16 symmetric weight quantization.
- `model/qwen3_embedding_bf16_seq512_bm1684x.bmodel`: matching BF16, non-quantized baseline.
- `tokenizer/`: Hugging Face tokenizer and Qwen3 embedding configuration. No original safetensors are needed at inference time.
- `embed_sail.py`: minimal SAIL inference test. It executes all 29 graphs and returns an MRL vector.
- `compare_sail.py`: comparison helper for identical inputs. On the 950M SOC, run BF16 and W4BF16 in separate processes; loading both engines in one process can exhaust native SAIL resources.

## Remote prerequisites

1. A BM1684X machine with a Sophon runtime compatible with the bmodel (bmodel reports `B.2.2+v1.0.0.dev-c3a57a2-20260428`).
2. The matching `sophon-sail` Python package from that runtime/SDK.
3. Python packages in `requirements.txt`.

The exact SAIL wheel is platform- and runtime-version-specific; install it from the remote machine's Sophon SDK rather than PyPI.

## Quick test

```bash
cd qwen3_embedding_w4bf16_bm1684x_seq512
python3 -m pip install -r requirements.txt
python3 embed_sail.py "北京是中国的首都" --dimensions 256
```

To compare BF16 and W4BF16 on identical text inputs:

```bash
python3 compare_sail.py "北京是中国的首都" "如何办理护照？" --dimensions 256
```

The output cosine similarity compares final, normalized MRL vectors. Also run with `--dimensions 1024` to distinguish INT4 quantization error from MRL truncation effects.

The bmodels expose BF16 tensors. `embed_sail.py` therefore uses explicit SAIL
Tensor maps and converts BF16 through its uint16 bit representation. The numpy
dict `Engine.process` overload is unsafe for these BF16 outputs with the
installed SAIL binding and may abort in native code.

The result is JSON with a normalized 256-dimensional vector. For documents and queries, use the model's recommended instruction format for queries, for example:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: 北京有什么著名景点？
```

## Important behavior

- The bmodel is static at 512 tokens and batch size 1. Inputs longer than 512 are truncated.
- Tokenization uses **left padding**. This places the final valid token at index `-1`.
- The model has Matryoshka Representation Learning support. The script takes the first N dimensions (default 256), then L2-normalizes; do not normalize 1024 dimensions first and then truncate.
- A 256-dimensional index reduces vector storage and brute-force similarity computation to roughly one quarter of 1024 dimensions. TPU encoding cost is unchanged.
- `attention_mask` is built as causal attention with padding excluded, consistent with the Qwen3 decoder backbone.

## Artifact integrity

`model/qwen3_embedding_w4bf16_seq512_bm1684x.bmodel`

```text
d7d26e840b46f856f5569617d54e710a8e4a0280fa90a568d9ed9b58dde78451
```
