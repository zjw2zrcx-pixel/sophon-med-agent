# Medical embedding handoff — Qwen3-Embedding-0.6B BF16

This directory is the self-contained handoff for generating medical document
vectors on a private NVIDIA 4090 server and comparing the full Torch model with
the deployed BM1684X TPU BF16 model.

No LLM rewriting is used. Every production document vector is generated from
the exact parsed encyclopedia `question` string.

## Contents

- `corpus/questions.jsonl`: 359,162 distinct questions to embed.
- `corpus/manifest.json`: source database identity, corpus ordering and hashes.
- `golden/cases.jsonl`: 13 representative comparison inputs.
- `golden/tpu_bf16_1024.npy`: TPU BF16 golden vectors, shape `(13, 1024)`.
- `golden/tpu_bf16_1024.metadata.json`: exact model, tokenizer, preprocessing,
  token counts, truncation and per-vector hashes.
- `tokenizer/`: the tokenizer assets used by the TPU golden generator.
- `scripts/generate_torch_vectors.py`: 4090 Torch generator for golden or corpus.
- `scripts/compare_vectors.py`: numerical comparison against TPU BF16 golden.
- `scripts/generate_tpu_golden.py`: reproducible TPU golden generator.
- `docs/`: source dataset notes and the RAG implementation plan snapshot.
- `SHA256SUMS`: transfer-integrity hashes.

## Authoritative model contract

The golden vectors were generated with:

- TPU artifact: `qwen3_embedding_bf16_seq512_bm1684x.bmodel`;
- TPU artifact SHA-256:
  `947c98a8a9a55295164eb53b990e48d14af474e06a69f2bdd7ae06f80f84398e`;
- dimensions: 1024;
- dtype returned to disk: float32;
- maximum sequence length: 512 tokens;
- padding: left;
- truncation: right;
- pooling: final hidden-state token at index `-1`;
- normalization: select the MRL prefix, then L2-normalize;
- document instruction: none.

The supplied tokenizer must be used for the comparison. Do not encode the JSON
record, `document_ids`, hash, or row number. Encode only the `question` value.

## Corpus row contract

Each `corpus/questions.jsonl` record contains:

```json
{
  "row_index": 0,
  "question_sha256": "...",
  "question": "曲匹地尔片的用法用量",
  "document_ids": [1]
}
```

Vector row `N` must correspond exactly to the record whose `row_index` is `N`.
`document_ids` maps the unique question vector back to one or more rows in
`med_search.sqlite`. The export is ordered by the lowest document ID represented
by each distinct question.

Do not sort, normalize, strip, rewrite, case-fold or otherwise modify question
text. JSON decoding is the only transformation before tokenization.

## 4090 environment

Create an isolated Python environment and install a CUDA build of PyTorch that
supports the 4090, then install:

```bash
python -m pip install -r requirements-torch.txt
```

Use the full, non-quantized Qwen3-Embedding-0.6B model. The generator explicitly
requests BF16 and eager attention to reduce implementation differences.

## Step 1: generate the Torch golden candidate

From this directory:

```bash
python scripts/generate_torch_vectors.py \
  --model /absolute/path/to/Qwen3-Embedding-0.6B \
  --tokenizer tokenizer \
  --input golden/cases.jsonl \
  --text-field text \
  --output golden/torch_bf16_1024.npy \
  --dimensions 1024 \
  --batch-size 13
```

The result must have shape `(13, 1024)`, float32 storage, finite values and unit
L2 norms. The model itself runs in BF16; conversion to float32 happens before
normalization and disk output, matching the TPU runtime contract.

## Step 2: compare Torch with TPU BF16

```bash
python scripts/compare_vectors.py \
  --candidate golden/torch_bf16_1024.npy \
  --minimum-cosine 0.995
```

Exact element equality is not expected across CUDA Torch and compiled BM1684X
BF16 kernels. The initial acceptance gate is:

- correct shape and row order;
- all vectors finite and normalized;
- every representative case cosine similarity at least `0.995`;
- no isolated truncation-boundary failure;
- tokenization hashes consistent with the metadata.

The `0.995` gate is deliberately conservative and should be reviewed together
with per-case L2 and maximum absolute differences. If it fails, do not generate
the full corpus until tokenizer, special tokens, truncation, attention,
pooling, model revision and normalization are reconciled.

The long synthetic case has more than 512 tokens and must report right-side
truncation to 512. It detects a common source of apparently valid but different
vectors.

## Step 3: generate all document vectors

Only after the golden comparison passes:

```bash
python scripts/generate_torch_vectors.py \
  --model /absolute/path/to/Qwen3-Embedding-0.6B \
  --tokenizer tokenizer \
  --input corpus/questions.jsonl \
  --text-field question \
  --output medical_document_vectors_1024.npy \
  --dimensions 1024 \
  --batch-size 64
```

Tune `--batch-size` for available VRAM. The output is expected to have shape
`(359162, 1024)` and occupy 1,471,127,552 payload bytes plus the small NPY
header (about 1.37 GiB).

Preserve the generated `.metadata.json` next to the vector file. Return both
files without changing row order.

If the deployed retrieval index later uses the 256-dimensional MRL prefix, it
can be derived from the 1024-dimensional normalized vectors without rerunning
the model:

```python
vectors_256 = vectors_1024[:, :256]
vectors_256 /= numpy.linalg.norm(vectors_256, axis=1, keepdims=True)
```

## Query instructions

The production document corpus has no instruction prefix. Query instruction
selection is a separate retrieval-quality decision. Golden cases contain both
plain and instructed query examples so preprocessing implementations can be
compared, but they do not change the document-vector contract.

## Integrity check after transfer

```bash
sha256sum -c SHA256SUMS
```

The full source `med_search.sqlite` and original 604 MB training JSONL are not
duplicated in this handoff. `questions.jsonl` is the complete, deterministic
vectorization input, and `manifest.json` records the source database SHA-256 and
document mapping needed to audit it.
