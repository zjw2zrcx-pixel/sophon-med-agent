# Compilation Source Manifest

Updated: 2026-08-21 22:39 +0800

The files below remain at their original repository paths. Hashes provide a
lightweight source snapshot; no copies of model inputs or generated outputs
are stored here.

## Archived compilation entry points

| Area | Source file | SHA-256 |
|---|---|---|
| VITS text-to-speech | `text_to_speech/vits-melo-tts-zh_en/convert_to_bmodel.sh` | `1dda6aab81aee741463bd8fc619c54b7cfa67f3503cb360593054fe7d1673539` |
| Qwen3.5 4B text-only | `models/Qwen3.5_4B_convert.sh` | `640bf6bed0f641c2b5a4d7089e95d04c46cfc57ab46c5b5990378b57cdc94811` |
| Qwen3 ASR 0.6B | `models/Qwen3_ASR_0.6B_convert.sh` | `d8974dbafb532f9be2a45dcd1ad21b3c4b9f5a2c137058ac02af7cc53da98b45` |
| Keyword spotting | `keyword_spotting/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/convert_to_bmodel.sh` | `98f76120bf3180082e4dcfae32e688b665aaf67515f674894f01f157457461a4` |
| Qwen3 embedding | `embedding_model/convert_to_bmodel.sh` | `c0289a50bef39dc2f852435884f62abf3ed3439f98ae54f84c75f979b671f941` |

## Supporting documentation

- `text_to_speech/README.md`
- `text_to_speech/QUANTIZATION_PROGRESS.md`
- `keyword_spotting/README.md`
- `embedding_model/embedding_handoff_bf16/README.md`
- `embedding_model/embedding_handoff_bf16/docs/EMBEDDING_RAG_PLAN.md`
- `models/Qwen3.5_4B/README.md`

## Explicit exclusions

Model checkpoints and tokenizers, generated `.bmodel`, `.onnx`, `.npz`, and
`.pt` files, calibration tables/vectors, compiler profiles, `build/`,
`quantization_work/`, `remote_deploy/` binaries, and `__pycache__/` are not
part of this archive.
