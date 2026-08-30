#!/usr/bin/env bash
# Build the 50-token / 256-frame hybrid Melo VITS package.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TTS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${SOURCE_DIR:-${TTS_DIR}/hf_vits_melo_zh_en}"
EXPORT_SCRIPT="${TTS_DIR}/native_50tk_256f/export_native_50tk_256f.py"
CHIP="${CHIP:-bm1684x}"
QUANTIZE="${QUANTIZE:-F32}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/build/${CHIP}}"
PREPARE_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--prepare-only] [--output-dir DIR]

Environment variables:
  SOURCE_DIR  Original model directory (default: ../hf_vits_melo_zh_en)
  CHIP        TPU target (default: bm1684x)
  QUANTIZE    TPU precision (default: F32)
  OUTPUT_DIR  Output package directory

--prepare-only exports the static ONNX and CPU TorchScript files without
requiring TPU-MLIR. It is useful for checking the split and CPU component.
EOF
}

while (($#)); do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "error: --output-dir needs a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -d "$SOURCE_DIR" ]] || { echo "error: source directory not found: $SOURCE_DIR" >&2; exit 1; }
SOURCE_DIR="$(cd -- "$SOURCE_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"

MODEL_ONNX="${SOURCE_DIR}/model.onnx"
ENCODER_ONNX="${SOURCE_DIR}/static_f32_50tk/encoder_b1_50tk.onnx"
DP_ONNX="${SCRIPT_DIR}/vits_dp_cpu_50.onnx"

for file in "$MODEL_ONNX" "$ENCODER_ONNX" "$DP_ONNX" "$EXPORT_SCRIPT"; do
  [[ -f "$file" ]] || { echo "error: required file not found: $file" >&2; exit 1; }
done

command -v python3 >/dev/null || { echo "error: python3 not found" >&2; exit 1; }
python3 -c 'import onnx, torch' 2>/dev/null || {
  echo "error: Python packages onnx and torch are required" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}/bmodel" "${OUTPUT_DIR}/cpu_component" "${OUTPUT_DIR}/onnx"

echo "==> Exporting the static 256-frame decoder and CPU controller"
python3 "$EXPORT_SCRIPT" --model "$MODEL_ONNX" --output-dir "${OUTPUT_DIR}/onnx"
mv "${OUTPUT_DIR}/onnx/cpu_dynamic_controller_256f.pt" \
  "${OUTPUT_DIR}/cpu_component/cpu_dynamic_controller_256f.pt"
install -m 0644 "$DP_ONNX" "${OUTPUT_DIR}/cpu_component/vits_dp_cpu_50.onnx"
install -m 0644 "$ENCODER_ONNX" "${OUTPUT_DIR}/onnx/encoder_b1_50tk.onnx"

if ((PREPARE_ONLY)); then
  echo "Done (prepare only). ONNX and CPU files are in: ${OUTPUT_DIR}"
  exit 0
fi

command -v model_transform.py >/dev/null || {
  echo "error: model_transform.py not found; activate the TPU-MLIR environment" >&2
  exit 1
}
command -v model_deploy.py >/dev/null || {
  echo "error: model_deploy.py not found; activate the TPU-MLIR environment" >&2
  exit 1
}

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vits_bmodel_build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "==> Compiling encoder (${CHIP}, ${QUANTIZE})"
(
  cd "$BUILD_DIR"
  model_transform.py \
    --model_name vits_encoder_50 \
    --model_def "${OUTPUT_DIR}/onnx/encoder_b1_50tk.onnx" \
    --input_shapes '[[1,50],[1],[1,50],[1]]' \
    --input_types 'int32,int32,int32,int32' \
    --mlir vits_encoder_50.mlir
  model_deploy.py \
    --mlir vits_encoder_50.mlir \
    --quantize "$QUANTIZE" \
    --chip "$CHIP" \
    --model "${OUTPUT_DIR}/bmodel/vits_encoder_50_${CHIP}_${QUANTIZE,,}.bmodel"
)

echo "==> Compiling flow + decoder (${CHIP}, ${QUANTIZE})"
(
  cd "$BUILD_DIR"
  model_transform.py \
    --model_name vits_flow_decoder_256 \
    --model_def "${OUTPUT_DIR}/onnx/decoder_50tk_256f.onnx" \
    --input_shapes '[[1,192,256],[1,1,256],[1,1,256,1],[1,1,256],[1,256,1]]' \
    --input_types 'float32,float32,float32,float32,float32' \
    --mlir vits_flow_decoder_256.mlir
  model_deploy.py \
    --mlir vits_flow_decoder_256.mlir \
    --quantize "$QUANTIZE" \
    --chip "$CHIP" \
    --model "${OUTPUT_DIR}/bmodel/vits_flow_decoder_256_${CHIP}_${QUANTIZE,,}.bmodel"
)

echo "Done. Hybrid deployment package: ${OUTPUT_DIR}"
