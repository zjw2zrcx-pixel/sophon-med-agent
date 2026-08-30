#!/usr/bin/env bash
# Export Qwen3-Embedding-0.6B to a BM1684X bmodel.
#
# Examples:
#   ./convert_to_bmodel.sh
#   ./convert_to_bmodel.sh --quantize bf16 --output-dir ../models/bmodel_qwen3_embedding_bf16
#   ./convert_to_bmodel.sh --seq-length 256 --quantize w8bf16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${SCRIPT_DIR}/Qwen3_Embedding_0.6B"
TPU_MLIR_ROOT="${PROJECT_ROOT}/sophon_project/tpu-mlir"
OUTPUT_DIR="${PROJECT_ROOT}/models/bmodel_qwen3_embedding"
QUANTIZE="w4bf16"
SEQ_LENGTH=512
GROUP_SIZE=64
CHIP="bm1684x"
SYMMETRIC=1
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: convert_to_bmodel.sh [options]

Options:
  -m, --model PATH          Hugging Face Qwen3-Embedding model directory
  -o, --output-dir PATH     Directory for generated bmodel (default: models/bmodel_qwen3_embedding)
  -q, --quantize MODE       w4bf16 (default), w8bf16, bf16, w4f16, w8f16, or f16
  -s, --seq-length N        Static sequence length (default: 512)
  -g, --group-size N        Weight quantization group size (default: 64)
  -c, --chip CHIP           Target chip (default: bm1684x)
      --asymmetric          Disable symmetric quantization for weight-only modes
      --dynamic             Enable dynamic prefill compilation
      --debug               Preserve generated MLIR and temporary files
  -h, --help                Show this help

The output block graph returns final hidden states. For retrieval, take the
last valid token, slice the desired MRL prefix (e.g. :256), then L2-normalize.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL_PATH="$2"; shift 2 ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -q|--quantize) QUANTIZE="$2"; shift 2 ;;
    -s|--seq-length) SEQ_LENGTH="$2"; shift 2 ;;
    -g|--group-size) GROUP_SIZE="$2"; shift 2 ;;
    -c|--chip) CHIP="$2"; shift 2 ;;
    --asymmetric) SYMMETRIC=0; shift ;;
    --dynamic) EXTRA_ARGS+=(--dynamic); shift ;;
    --debug) EXTRA_ARGS+=(--debug); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing ${MODEL_PATH}/config.json" >&2; exit 1; }
[[ -f "${TPU_MLIR_ROOT}/envsetup.sh" ]] || { echo "Missing tpu-mlir: ${TPU_MLIR_ROOT}" >&2; exit 1; }
[[ "$SEQ_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "--seq-length must be a positive integer" >&2; exit 2; }
[[ "$GROUP_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "--group-size must be a positive integer" >&2; exit 2; }

case "$QUANTIZE" in
  w4bf16|w8bf16|bf16|w4f16|w8f16|f16) ;;
  *) echo "Unsupported --quantize: ${QUANTIZE}" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
source "${TPU_MLIR_ROOT}/envsetup.sh"

CMD=(python3 "${TPU_MLIR_ROOT}/python/tools/llm_convert.py"
  --model_path "$MODEL_PATH"
  --seq_length "$SEQ_LENGTH"
  --quantize "$QUANTIZE"
  --q_group_size "$GROUP_SIZE"
  --chip "$CHIP"
  --out_dir "$OUTPUT_DIR")

if [[ "$SYMMETRIC" -eq 1 && "$QUANTIZE" == w4* || "$SYMMETRIC" -eq 1 && "$QUANTIZE" == w8* ]]; then
  CMD+=(--symmetric)
fi
CMD+=("${EXTRA_ARGS[@]}")

printf 'Export command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Generated bmodels:"
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.bmodel' -printf '  %p (%s bytes)\n'
