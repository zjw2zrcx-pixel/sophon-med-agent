"${TPUC_ROOT:-/root/opt/tpu-mlir-f3d218e-text-only}/python/tools/llm_convert.py" \
  -m /workspace/Project/jcs/models/Qwen3.5_4B \
  -s 8192 -c bm1684x -q w4bf16 -g 64 \
  --num_device 1 --num_core 1 \
  --use_history_kv --chunk_length 1024 \
  --text_only \
  --out_dir /workspace/Project/jcs/models/bmodel_qwen3.5_4b_history_without_vit
