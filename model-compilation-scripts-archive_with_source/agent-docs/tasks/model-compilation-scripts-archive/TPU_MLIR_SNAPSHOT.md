# TPU-MLIR Working-Tree Snapshot

Updated: 2026-08-21 22:39 +0800

## Repository state

- Repository: `/workspace/Project/jcs/sophon_project/tpu-mlir`
- Branch: `master`
- Base HEAD: `87edab473868fa4990c33f81b4fe602d85287180`
- Base subject: `docs: refine llm description`
- Working tree: 13 modified tracked files; 8 untracked source files
- `git diff --check`: passed

## Change summary

The working tree adds/changes support around sequential recurrent gated delta
rule lowering/code generation and Qwen3.5 LLM conversion, including text-only
conversion plumbing. The current diff is 773 insertions and 113 deletions
across the 13 tracked files.

## Modified tracked files

```text
include/tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h
include/tpu_mlir/Dialect/Top/IR/TopOps.td
include/tpu_mlir/Dialect/Tpu/IR/TpuOps.td
include/tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp
lib/Conversion/TopToTpu/LoweringBM1684X.cpp
lib/PplBackend/CMakeLists.txt
lib/PplBackend/Dynkernel.cmake
lib/PplBackend/include/ppl_dyn_fw.h
lib/PplBackend/src/recurrent_gated_delta_rule.pl
lib/PplBackend/src_dyn/recurrent_gated_delta_rule_ctrl.c
python/llm/LlmConverter.py
python/llm/Qwen3_5Converter.py
python/tools/llm_convert.py
```

## Untracked source files

```text
lib/Conversion/TopToTpu/BM1684X/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Top/Interfaces/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/BM1684/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/BM1684X/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/CV18xx/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/Common/SequentialRecurrentGatedDeltaRule.cpp
lib/PplBackend/src/sequential_recurrent_gated_delta_rule.cpp
lib/PplBackend/src_dyn/sequential_recurrent_gated_delta_rule_ctrl.c
```

This is metadata only; no full diff, generated object, compiler cache, model
weight, or binary output is stored. Recreate the source diff with:

```bash
git -C sophon_project/tpu-mlir diff
git -C sophon_project/tpu-mlir status --short
```
