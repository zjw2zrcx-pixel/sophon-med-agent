# Medical Database Embedding RAG Plan

## 1. Goal

Upgrade the document-retrieval layer used by `medical_consult` with the local
Qwen3-Embedding endpoint while preserving the existing medical safety and
structured-evidence behavior.

The embedding backend is intended to improve semantic recall for colloquial,
paraphrased, and vocabulary-mismatched medical questions. It must not bypass
red-flag handling, medication restrictions, source attribution, or the graph
and structured-fact retrieval paths.

## 2. Current system

The active online tool is `medical_consult(query=...)`. The legacy Rust
`med_query` adapter remains available only for rollback diagnostics.

The current `MedicalRetriever` combines:

- canonical entity and alias matching;
- character unigram/bigram fuzzy matching;
- rule-based intent and red-flag detection;
- SQLite graph-edge retrieval;
- structured Huatuo knowledge-graph facts;
- FTS5/BM25 retrieval over Huatuo encyclopedia questions;
- conservative medication gating and clarification questions.

Current index scale:

- 62,196 entities;
- 543,673 graph edges;
- 774,602 structured facts;
- 362,420 encyclopedia QA documents;
- 359,162 distinct encyclopedia questions;
- 60,000 sampled patient dialogue queries, including 6,092 eval records.

## 3. Target architecture

```text
user medical question
  -> red-flag, medication-safety, intent and entity processing
  -> structured retrieval
       - graph edges
       - normalized facts
  -> hybrid document retrieval
       - FTS5/BM25 sparse candidates
       - Qwen3 dense-vector candidates
  -> rank fusion, deduplication and evidence limits
  -> compact sourced result for the agent
```

Dense retrieval initially applies only to encyclopedia documents. Medication
answers remain restricted to the existing explicit graph edges and structured
facts. Symptom-only medication requests, urgent red flags, and other existing
retrieval blocks remain unchanged.

## 4. Model and service contract

Initial settings:

- endpoint: OpenAI-compatible `POST /v1/embeddings`;
- model: `qwen3-embedding-0.6b`;
- dimensions: 256;
- encoding format: float;
- similarity: inner product over L2-normalized vectors, equivalent to cosine;
- document text: encyclopedia `question`, without a query instruction;
- query text:

  ```text
  Instruct: Given a medical question, retrieve relevant passages that answer the question
  Query: <original user question>
  ```

The instruction text is provisional until the model validation and retrieval
evaluation compare it against the generic web-search instruction and an
instruction-free query.

## 5. Mandatory feasibility gate

Before implementation or full index construction:

1. Start the BF16 embedding model with `/data/env310` outside the sandbox.
2. Verify `/health`, model name, 256-dimensional output and finite values.
3. Verify vector L2 norm is approximately 1.0.
4. Compare cosine similarities for:
   - identical inputs;
   - close medical paraphrases;
   - related but non-equivalent medical questions;
   - unrelated medical and non-medical questions.
5. Repeat an input to verify deterministic/stable output.
6. Measure cold/warm latency and serial multi-input behavior.
7. Verify qwen3.5 and qwen3-embedding can coexist on TPU 0. If not, stop and
   define a model scheduling strategy before integrating online retrieval.

The gate passes only if the endpoint is stable, vectors are normalized and
finite, medical paraphrases consistently score above unrelated pairs, and the
deployment can supply query embeddings during a normal agent tool round.

### 5.1 Initial W4BF16 validation result (historical, 2026-08-02)

The W4BF16 model was loaded directly on TPU 0 with `/data/env310` and tested
through `127.0.0.1:8006`.

This result is retained only as historical quantization data. The active server
contract and all Torch comparison golden vectors now use the non-quantized TPU
BF16 artifact described in section 5.2.

- `/health`: `ready`;
- reported model: `qwen3-embedding-0.6b`;
- output: 256 finite float values with L2 norm `1.0`;
- repeated identical input: cosine `1.0000001`, maximum element difference
  `0.0`;
- first request latency: `0.375 s`;
- warm single request latency: `0.339 s`;
- eight-input request: `2.684 s`, or `0.335 s/item`, confirming serial
  batch-one execution;
- loaded TPU memory reported by `bm-smi`: approximately `805 MiB`;
- invalid dimensions, unknown model and empty input all returned HTTP 400 with
  structured error bodies.

Observed cosine similarities using the provisional medical query instruction:

| Pair | Cosine |
| --- | ---: |
| late-stage syphilis treatment-duration paraphrase | 0.850339 |
| syphilis treatment duration vs follow-up timing | 0.797363 |
| syphilis query vs unrelated cold question | 0.329245 |
| syphilis query vs travel question | 0.223439 |
| abdominal-pain/diarrhea paraphrase | 0.828744 |
| abdominal query vs hypertension question | 0.398535 |
| abdominal query vs travel question | 0.279091 |

For one syphilis paraphrase pair, the no-instruction query scored `0.900255`,
the medical instruction scored `0.850339`, and the generic web-search
instruction scored `0.829140`. This single example is not sufficient to choose
the production query format. The three variants must be evaluated over the
reviewed retrieval set before freezing the contract.

At the observed `0.335 s/item`, encoding all 359,162 distinct encyclopedia
questions would take approximately 33.4 hours before retries and data-handling
overhead. The builder therefore needs resumability and progress checkpoints.

The endpoint, vector validity, deterministic output, basic semantic separation,
API validation and standalone TPU loading portions of the feasibility gate have
passed. Concurrent residency and operation with qwen3.5 remain untested and are
still required before online integration.

### 5.2 Active BF16 deployment validation (2026-08-02)

The server configuration now points to
`qwen3_embedding_bf16_seq512_bm1684x.bmodel`, and the server rejects any other
artifact filename before initializing SAIL. `/status` and embedding responses
report `variant=bf16` and the exact artifact name.

Positive BF16 validation:

- artifact bytes: `1,204,776,960`;
- artifact SHA-256:
  `947c98a8a9a55295164eb53b990e48d14af474e06a69f2bdd7ae06f80f84398e`;
- `/health`: `ready`;
- 1024 finite float values with L2 norm `1.0`;
- observed single-request latency: `0.364 s`;
- TPU memory reported by `bm-smi`: approximately `1411 MiB`.

A negative launch test deliberately passed the W4 artifact. The server entered
`error`, logged an explicit non-BF16 refusal, and returned HTTP 503 from the
embedding endpoint. This confirms the BF16 requirement cannot silently regress
through a configuration path change.

The 13-case BF16 golden set produced document/paraphrase cosine `0.888429`,
document/instructed-query cosine `0.836711`, and unrelated document cosine
`0.274677`. The handoff package and its full preprocessing metadata are under
`med_database/embedding_handoff_bf16`.

## 6. Vector index format

Keep vector artifacts separate from `med_search.sqlite`:

- `medical_document_vectors.f32.npy`: normalized float32 matrix;
- `medical_document_vector_ids.npy`: SQLite document IDs aligned with rows;
- `medical_document_vector_manifest.json`: model and corpus contract.

The manifest must include:

- format and schema versions;
- model name and embedding dimensions;
- document and query formatting contracts;
- SQLite index identity and corpus fingerprint;
- row count and ID range;
- build timestamps and completion state;
- checksums for published artifacts.

At 359,162 unique questions and 256 float32 dimensions, the raw vector matrix
is approximately 351 MiB. The first implementation should use a read-only NumPy
memory map and exact inner-product search. This avoids introducing an unverified
native ANN dependency. Add HNSW only if measured online P95 latency requires it.

## 7. Offline index builder

Create a dedicated, resumable vector-index command rather than coupling TPU
inference to the existing SQLite builder.

Requirements:

- stream documents from read-only SQLite;
- deduplicate identical questions and retain the document-ID mapping;
- call the configured embedding endpoint with bounded retries and timeouts;
- accommodate the model's static batch size of one;
- checkpoint progress without publishing incomplete artifacts;
- validate dimensions, finiteness and normalization for every response;
- atomically publish the completed vector, ID and manifest files;
- refuse silent reuse when the model, dimension, formatting or corpus changes;
- emit throughput and estimated-completion metrics.

A measured sample run must estimate the full 359,162-question build duration
before the full build is approved.

## 8. Online retrieval

Add an embedding client and dense document retriever with configurable:

- endpoint URL;
- model name;
- dimensions;
- request timeout;
- dense top-k and fusion depth;
- feature enable/disable switch.

Runtime behavior:

1. Load vector and ID artifacts read-only using `numpy.load(..., mmap_mode="r")`.
2. Validate the manifest against the active SQLite index and configured model.
3. Embed the original user query with the selected query instruction.
4. Retrieve dense top-k by normalized inner product.
5. Retrieve sparse top-k using the existing FTS5 path.
6. Fuse rankings using reciprocal-rank fusion, then deduplicate.
7. Fetch final documents by ID from SQLite and retain source attribution.
8. Preserve the current maximum evidence count and compact output limits.

If the endpoint is unavailable, times out, returns malformed vectors, or the
index contract is stale, log a bounded warning and fall back to FTS5. Dense
retrieval failure must not make `medical_consult` unavailable.

## 9. Testing and evaluation

Add tests for:

- embedding response parsing and validation;
- timeout, HTTP failure and malformed-vector fallback;
- manifest/corpus mismatch detection;
- vector-to-document ID alignment;
- exact-search ranking on a small deterministic fixture;
- sparse/dense rank fusion and deduplication;
- preservation of urgent, symptom-consultation and medication safety behavior.

Evaluate the old and hybrid retrievers using the existing dialogue eval split
plus a reviewed set of medical paraphrases. Track:

- Recall@1, Recall@5 and MRR where a target can be defined;
- no-result rate;
- irrelevant-evidence rate from manual review;
- query-embedding latency and total retrieval P50/P95;
- dense fallback/error rate;
- safety regression count.

## 10. Rollout and rollback

Roll out behind `MEDICAL_DENSE_ENABLED`, disabled by default until evaluation
passes. Keep the existing FTS5 implementation and index untouched throughout
the first deployment. Document service health checks, model load/unload steps,
index construction, corpus refresh, metrics, and rollback.

Suggested implementation sequence:

1. Complete the mandatory model/service feasibility gate.
2. Finalize the embedding text and index contracts from measured results.
3. Implement and test the resumable offline vector builder.
4. Build a small pilot index and benchmark exact search.
5. Implement the online embedding client and dense retriever.
6. Integrate sparse/dense fusion without changing safety gates.
7. Run retrieval and safety evaluations.
8. Build the full index, enable a controlled rollout, and monitor fallback and
   latency metrics.

## 10.1 Implementation and benchmark result (2026-08-03)

The initial production implementation is complete behind the default-off
`MEDICAL_DENSE_ENABLED` rollout switch:

- `dense.py` implements the strict BF16 embedding client, read-only mmap exact
  inner-product index, serialized scan concurrency, document-ID expansion and
  weighted reciprocal-rank fusion;
- `retriever.py` preserves red-flag, symptom-consultation and medication gates,
  filters self-medication document questions, and falls back to FTS on every
  dense failure;
- `medconsult.py` exposes environment configuration, validates the dense
  manifest and prewarms vector pages before serving the first query;
- exact entity matching now uses a trie instead of scanning every alias sharing
  the current first character;
- `test_dense.py` covers exact ranking and mappings, RRF deduplication, unsafe
  document filtering, and endpoint-failure fallback;
- `benchmark_hybrid_retrieval.py` provides a reproducible ten-query BF16 TPU
  benchmark and writes `med_database/hybrid_retrieval_benchmark.json`.

Final warm benchmark over 359,162 vectors / 362,420 mapped document IDs:

| Stage | P50 | P95 |
| --- | ---: | ---: |
| TPU BF16 query embedding | 328.324 ms | 333.665 ms |
| exact 256-dimension vector Top-K | 153.998 ms | 161.126 ms |
| full `MedicalRetriever.consult` | 591.709 ms | 1,058.456 ms |

Retriever construction and prewarm took 9.422 seconds in the recorded process;
it is cached by the MCP tool and is not paid per query. A prior un-prewarmed
cold mmap scan took 7.821 seconds, which is why prewarming is enabled by
default. A separate MCP factory run after cache eviction measured 20.391 seconds
for the complete cold initialization and prewarm, followed by a 621.366 ms
hybrid query. The exact scan is limited to one concurrent caller because
parallel matrix scans were previously observed to contend for memory bandwidth.

The rollout switch remains off by default pending a larger reviewed Recall@K,
MRR and safety regression set. The ten-query smoke benchmark passed basic
relevance inspection and confirmed that the high-blood-pressure query no longer
surfaces the dense Top-1 self-medication question.

### 10.2 Proxy evaluation (2026-08-03)

The dialogue eval split has no target encyclopedia document IDs, so it cannot
support honest Recall@K or MRR by itself. A reproducible, transparent proxy set
was added instead: ten topic-plus-aspect relevance cases and four safety-gate
cases. Each sparse and hybrid record includes `dense_used`.

| Metric | Sparse | Hybrid |
| --- | ---: | ---: |
| proxy Hit@1 | 0.40 | 1.00 |
| proxy Hit@2 | 0.50 | 1.00 |
| latency P50 | 106.713 ms | 587.788 ms |
| latency P95 | 566.926 ms | 1,048.048 ms |

All ten normal relevance queries used dense retrieval. All four urgent or
unsafe-medication queries passed their expected status/reason checks and used
no dense retrieval. The evaluation exposed and led to fixes for ambiguous
symptom/disease labels (for example `头痛`), dose-worded medication intent,
diet intent and follow-up/check intent prioritization.

These proxy scores must not be presented as corpus-wide Recall: substring topic
and aspect rules are weaker than document-ID gold labels and clinician relevance
judgments. In particular, the pulmonary-tuberculosis follow-up query retrieves
diagnostic/check documents that satisfy the proxy terms but are not a precise
follow-up answer. Human review and a true labeled set remain required before
enabling dense retrieval by default.

### 10.3 Four-way parallel retrieval result (2026-08-03)

After safety, entity and intent processing, graph edges, structured facts,
FTS5 and dense retrieval now run concurrently. SQLite operations use separate
read-only connections. The dense subchain remains ordered as query embedding
then exact vector scan.

The same ten-query benchmark before and after parallelization produced:

| Metric | Serial | Parallel | Change |
| --- | ---: | ---: | ---: |
| end-to-end P50 | 582.160 ms | 503.754 ms | -13.5% |
| end-to-end P95 | 1,047.231 ms | 621.693 ms | -40.6% |
| end-to-end mean | 627.860 ms | 520.022 ms | -17.2% |

Parallel branch P50/P95 timings were:

| Branch | P50 | P95 |
| --- | ---: | ---: |
| TPU BF16 embedding | 337.036 ms | 374.024 ms |
| exact vector scan | 158.395 ms | 196.150 ms |
| FTS5 | 108.992 ms | 614.008 ms |
| graph edges | 6.830 ms | 17.599 ms |
| structured facts | 4.262 ms | 15.356 ms |
| complete parallel region | 502.343 ms | 620.250 ms |

For typical queries the critical path is the ordered dense chain, approximately
495 ms at the component medians. For the syphilis treatment-duration query,
FTS5 took 614 ms and became the critical path while dense completed before the
fusion wait. Thus further median improvement primarily requires faster query
embedding or an ANN/vector-scan improvement; tail improvement also requires
optimizing the FTS token query. The parallel safety regression remained 4/4,
with `dense_used=false` for every gated query, and relevance proxy Hit@1/Hit@2
remained 1.00/1.00.

## 11. Optional DeepSeek per-document enrichment

### 11.1 Decision

DeepSeek enrichment is feasible but is an optional retrieval experiment, not a
prerequisite for the first embedding index. Each encyclopedia QA record must be
processed by one independent API request. Multiple independent requests may run
with bounded concurrency, but records must never share a prompt payload.

The generated data must remain retrieval-only metadata. It must not replace the
original question, answer, source, graph facts, medication controls, or other
evidence returned by `medical_consult`.

### 11.2 Safe output contract

DeepSeek should generate only:

- `canonical_question`;
- a narrow intent enum, including combined intents such as
  `treatment_duration_prognosis`;
- grounded medical entities with an exact source `evidence` span;
- at most three meaning-preserving alternate queries;
- at most eight grounded search keywords.

DeepSeek must not generate the final `retrieval_text`. Local deterministic code
should concatenate validated fields with the original question. This prevents a
fluent model-written answer summary from being mistaken for trusted medical
evidence.

Medication dose, route, frequency, contraindication and treatment-instruction
fields should be excluded from the retrieval card even when they occur in the
source answer. They are unnecessary for topic retrieval and increase the risk
of unsafe downstream use.

### 11.3 Initial API probe (2026-08-02)

`deepseek-v4-flash` was called through the official Chat Completions endpoint
using one encyclopedia record per request, non-thinking mode and JSON Output.
The API key was read only from `DEEPSEEK_API_KEY` and was never printed or stored
in a repository file.

The first prompt version returned valid JSON but exposed two design problems:

- it classified “三期梅毒多久能治愈吗” only as `treatment`;
- it broadened an alternate query to “三期梅毒的治疗方法” and generated a
  fluent answer-like `retrieval_text`.

The second prompt removed model-generated retrieval prose, added narrower and
combined intents, and required every alternate query to preserve all core
constraints. Three independent records then passed JSON shape, enum, list-size
and exact entity-evidence checks:

| Document | Intent | Latency | Input tokens | Output tokens | Cache hit |
| --- | --- | ---: | ---: | ---: | ---: |
| 三期梅毒多久能治愈吗 | treatment_duration_prognosis | 1.963 s | 736 | 167 | 0 |
| 曲匹地尔片的用法用量 | medication | 2.088 s | 432 | 217 | 0 |
| 肝癌术后饮食 | diet | 1.742 s | 719 | 155 | 256 |

The syphilis alternate queries all retained both duration and cure/prognosis.
The diet sample produced natural colloquial rewrites. The medication sample
correctly grounded its fields but also extracted dose and frequency, confirming
that high-risk field filtering must be enforced locally rather than trusted to
prompt wording alone. Keywords also require exact-substring or reviewed synonym
validation before publication.

At the observed average of roughly 629 input and 180 output tokens per record,
current API pricing implies an order-of-magnitude full-corpus cost around
USD 40–55, depending on prompt-cache behavior and retries. Mean observed latency
was approximately 1.93 seconds: purely sequential processing would take about
8.1 days, while idealized concurrency 16 or 32 would reduce inference wall time
to roughly 12 or 6 hours before throttling, retries and storage overhead.

### 11.4 Required pilot

Do not process the full corpus yet. First build a resumable 1,000-record pilot:

1. Stratify samples across medication, treatment, symptoms, checks, prognosis,
   diet, pregnancy, short answers, long answers and multi-question records.
2. Use one record per API request with bounded concurrency, request and cost
   caps, exponential retry, per-record hashes and prompt/model versioning.
3. Validate JSON types, enum values, list lengths, exact evidence spans,
   forbidden high-risk entity types and input-grounded keywords.
4. Preserve rejected records and reasons; never silently accept partial output.
5. Build and compare three indexes: original question, deterministic
   question-plus-source format, and DeepSeek-enriched retrieval card.
6. Measure Recall@1/5, MRR, irrelevant retrieval, medical safety regressions,
   token cost and end-to-end build time.
7. Approve full-corpus enrichment only if it materially improves retrieval over
   both non-LLM baselines without increasing safety failures.
