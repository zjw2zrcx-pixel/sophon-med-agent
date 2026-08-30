# Decisions

## DEC-001 — Archive source references, not generated model artifacts
- Date: 2026-08-21
- Status: Accepted
- Context: The requested handoff is for compilation scripts and documentation; model weights and compiler outputs are large and non-reproducible as task memory.
- Decision: Store task-isolated Markdown manifests and Git change metadata. Keep source scripts at their repository paths and exclude weights, binaries, build output, and caches.
- Alternatives rejected: Copying complete model directories or generated deployment packages into the handoff folder.
- Consequences: The next session must use the repository checkout and available model assets to reproduce a build.
- Relevant files: `SOURCE_MANIFEST.md`, `TPU_MLIR_SNAPSHOT.md`, `CURRENT_STATE.md`
