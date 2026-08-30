# Python medical search index

The online agent exposes only `medical_consult(query=...)`. The legacy Rust
`med_query` adapter remains in the source tree for rollback diagnostics but is
not registered with the model.

## Build or refresh the index

Run from `/data/structure`:

```bash
/data/env310/bin/python -m agents.Medical.build_index \
  --output /data/structure/med_database/med_search.sqlite
```

The builder writes a temporary database and atomically replaces the destination
only after a successful build. Runtime lookup uses `MEDICAL_INDEX_PATH` when
set, otherwise it reads `med_database/med_search.sqlite`.

## Data policy

- `med_neo4j/entity.json` and `relation.json`: primary graph entities/edges.
- `huatuo_knowledge_graph_qa/train_datasets.jsonl`: recognised structured fact
  templates only; unknown templates are skipped; validation/test are excluded.
- `huatuo_encyclopedia_qa/train_datasets.jsonl`: secondary retrieval documents;
  validation/test are excluded.
- `Chinese-medical-dialogue-data/Data/*/*.csv`: only `department`, `title`, and
  patient `ask` are sampled for language/evaluation data. The answer column is
  never imported.

The online retriever keeps entity names, aliases, and a small character n-gram
index in memory. Graph edges, facts, documents, and dialogue samples stay in
SQLite and are read on demand.

## BF16 embedding RAG

The encyclopedia path supports hybrid dense/FTS retrieval. It uses the BF16
Qwen3-Embedding endpoint, an exact cosine search over the 359,162 normalized
question vectors, weighted reciprocal-rank fusion, document-ID deduplication,
and the existing maximum evidence limits. Graph facts, red flags, symptom-only
consultation blocks, and medication restrictions still run before document
retrieval.

After the safety, entity and intent stages, graph edges, structured facts, FTS5
and the dense chain run concurrently. Each SQLite branch uses an independent
read-only connection. Dense embedding and vector search remain sequential with
respect to each other because the search requires the query vector.

Dense retrieval is disabled by default for controlled rollout. Enable it with:

```bash
export MEDICAL_DENSE_ENABLED=1
export MEDICAL_EMBEDDING_URL=http://127.0.0.1:8006
```

The default artifacts are:

- `med_database/medical_document_vectors_256.npy`;
- `med_database/medical_document_vector_offsets.npy`;
- `med_database/medical_document_vector_doc_ids.npy`;
- `med_database/medical_document_vectors_256.manifest.json`.

Runtime settings:

- `MEDICAL_DENSE_MANIFEST`: alternate manifest path;
- `MEDICAL_EMBEDDING_MODEL`: defaults to `qwen3-embedding-0.6b`;
- `MEDICAL_DENSE_DIMENSIONS`: defaults to `256`;
- `MEDICAL_DENSE_TOP_K`: defaults to `30`;
- `MEDICAL_EMBEDDING_TIMEOUT`: defaults to `5` seconds;
- `MEDICAL_DENSE_QUERY_PREFIX`: defaults to empty (original query text);
- `MEDICAL_DENSE_MAX_CONCURRENT_SCANS`: defaults to `1`;
- `MEDICAL_DENSE_PREWARM`: defaults to `1` and moves mmap page faults into
  retriever initialization.

The endpoint response must identify `model_variant=bf16`; other variants,
malformed vectors, timeouts, and endpoint failures are rejected. A query then
falls back to FTS and structured evidence rather than failing the medical tool.

Run tests and the reproducible TPU benchmark from the repository root:

```bash
/data/env310/bin/python -m unittest agents.Medical.test_dense -v
/data/env310/bin/python -m agents.Medical.benchmark_hybrid_retrieval
/data/env310/bin/python -m agents.Medical.evaluate_hybrid_retrieval
```

The latest benchmark report is
`med_database/hybrid_retrieval_benchmark.json`; the sparse/hybrid proxy
evaluation is `med_database/hybrid_retrieval_evaluation.json`. Every query row
records `dense_used`. Urgent and unsafe-medication cases are expected to record
`dense_used=false` because they are stopped before document retrieval.

## Review medical consultation traces

Create a compact, deduplicated JSONL before human or model review so repeated
benchmark payloads do not fill the review context:

```bash
/data/env310/bin/python -m agents.Medical.extract_medconsult_trajectories \
  docs/final_agent_benchmark \
  --output /tmp/medconsult_review.jsonl
```

The output retains the query, intent, normalized terms, compact associations and
evidence, plus all source occurrences. Long evidence text is bounded and exact
duplicate query/result pairs are emitted once.

Parallel comparison artifacts are
`med_database/hybrid_retrieval_parallel_benchmark.json` and
`med_database/hybrid_retrieval_parallel_evaluation.json`. Per-query timing also
includes `sparse_ms`, `edge_ms`, `fact_ms`, `dense_wait_ms` and
`parallel_total_ms`.
