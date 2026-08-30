#!/usr/bin/env bash
# Compile and combine the chunk-16 sherpa-onnx KWS transducer for Sophon TPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}}"
CHIP="${1:-${CHIP:-bm1684x}}"
QUANTIZE="${2:-${QUANTIZE:-F32}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/bmodel/${CHIP}}"

if [[ $# -gt 2 ]]; then
  echo "Usage: $(basename "$0") [chip] [quantize]" >&2
  echo "Example: $(basename "$0") bm1684x F32" >&2
  exit 2
fi

[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"

ENCODER="${MODEL_DIR}/encoder-epoch-13-avg-2-chunk-16-left-64.onnx"
DECODER="${MODEL_DIR}/decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
JOINER="${MODEL_DIR}/joiner-epoch-13-avg-2-chunk-16-left-64.onnx"

for file in "$ENCODER" "$DECODER" "$JOINER"; do
  [[ -f "$file" ]] || { echo "error: required model not found: $file" >&2; exit 1; }
done

for command in model_transform.py model_deploy.py bmodel_combine.py; do
  command -v "$command" >/dev/null || {
    echo "error: $command not found; activate the TPU-MLIR environment" >&2
    exit 1
  }
done

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kws_bmodel_build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

compile() {
  local name="$1" model="$2" shapes="$3" types="$4"
  echo "==> Compiling ${name} (${CHIP}, ${QUANTIZE})"
  (
    cd "$BUILD_DIR"
    model_transform.py \
      --model_name "$name" \
      --model_def "$model" \
      --input_shapes "$shapes" \
      --input_types "$types" \
      --mlir "${name}.mlir"
    model_deploy.py \
      --mlir "${name}.mlir" \
      --quantize "$QUANTIZE" \
      --chip "$CHIP" \
      --model "${BUILD_DIR}/${name}.bmodel"
  )
}

# Static batch-1 state for chunk=16 and left-context=64. The encoder has
# 39 inputs: one 45x80 fbank chunk, six cache groups and two position states.
ENCODER_SHAPES='[[1,45,80],[64,1,128],[1,1,64,96],[64,1,48],[64,1,48],[1,128,7],[1,128,7],[32,1,128],[1,1,32,96],[32,1,48],[32,1,48],[1,128,7],[1,128,7],[16,1,128],[1,1,16,96],[16,1,48],[16,1,48],[1,128,7],[1,128,7],[8,1,256],[1,1,8,96],[8,1,96],[8,1,96],[1,128,7],[1,128,7],[16,1,128],[1,1,16,96],[16,1,48],[16,1,48],[1,128,7],[1,128,7],[32,1,128],[1,1,32,96],[32,1,48],[32,1,48],[1,128,7],[1,128,7],[1,128,3,19],[1]]'
ENCODER_TYPES='float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,float32,int64'

compile kws_encoder_chunk16 "$ENCODER" "$ENCODER_SHAPES" "$ENCODER_TYPES"
compile kws_decoder_chunk16 "$DECODER" '[[1,2]]' 'int64'
compile kws_joiner_chunk16 "$JOINER" '[[1,320],[1,320]]' 'float32,float32'

COMBINED="${OUTPUT_DIR}/kws_transducer_chunk16_${CHIP}_${QUANTIZE,,}.bmodel"
echo "==> Combining encoder, decoder and joiner"
bmodel_combine.py \
  --output "$COMBINED" \
  --inputs="${BUILD_DIR}/kws_encoder_chunk16.bmodel ${BUILD_DIR}/kws_decoder_chunk16.bmodel ${BUILD_DIR}/kws_joiner_chunk16.bmodel"

echo "Done. Combined BModel: ${COMBINED}"
