# Current State

## Task
- Slug: `model-compilation-scripts-archive`
- Description: Preserve the locations and state of model compilation scripts and the Qwen3.5 text-only TPU-MLIR changes.

## Metadata
- Updated: 2026-08-21 22:39 +0800
- Repository: `/workspace/Project/jcs`
- Branch: `tpu-mlir: master`
- HEAD: `87edab473868fa4990c33f81b4fe602d85287180` (`docs: refine llm description`)

## Long-term goal

Keep reproducible compilation entry points and implementation-change context
available without storing model weights, caches, or generated binaries.

## Current work objective

Archive the VITS, Qwen, keyword-spotting, and embedding compilation script
locations, plus a non-binary snapshot of the Qwen3.5 text-only TPU-MLIR branch
changes.

## Recent work

- Confirmed five compilation entry points and recorded SHA-256 hashes in `SOURCE_MANIFEST.md`.
- Recorded the TPU-MLIR base commit, branch, changed-file list, and diff statistics in `TPU_MLIR_SNAPSHOT.md`.
- Added `QWEN3_COMPILATION_PATH_COMPARISON.md`, documenting the baseline full multimodal route and the changed `--text_only` route stage by stage.
- Excluded model files, weights, `.bmodel/.onnx/.npz`, build directories, and Python caches.

## Working tree

- Parent project Git resolution points to `/workspace` with no usable `HEAD`; it is not used as the source snapshot.
- `sophon_project/tpu-mlir` is on `master` at `87edab4` with 13 modified tracked files and 8 untracked source files.
- No source files were modified by this archival task.

## Verification status

- `bash -n` passed for all five archived shell entry points.
- `git diff --check` passed for the TPU-MLIR working tree.
- Full model compilation was not run.

## Current blockers

None for the requested archival scope. Exact parent-project commit provenance is unavailable because that Git checkout has no usable `HEAD`.

## Next steps

### P0 — Must do next

1. Before applying or committing TPU-MLIR changes, compare the working tree with `TPU_MLIR_SNAPSHOT.md`; acceptance: expected files and base commit still match; dependency: unchanged TPU-MLIR checkout; relevant files: `sophon_project/tpu-mlir`; user decision: No.

### P1 — Important

1. Run the relevant conversion/build validation when the required model assets and toolchain are available; acceptance: each intended target compiles successfully; dependency: TPU-MLIR and model inputs; user decision: No.

### P2 — Optional

1. Commit or export the TPU-MLIR source changes separately if a durable code snapshot is required; acceptance: a named commit or patch is supplied; user decision: Yes.

## Do not repeat

- Do not archive model weights, generated bmodels/ONNX files, calibration artifacts, build directories, or caches for this task.
- Do not treat the parent `/workspace` Git metadata as a valid project commit until a usable `HEAD` exists.

## Read first next time

- `SOURCE_MANIFEST.md`
- `TPU_MLIR_SNAPSHOT.md`
- `sophon_project/tpu-mlir/python/llm/Qwen3_5Converter.py`
- `models/Qwen3.5_4B_convert.sh`
