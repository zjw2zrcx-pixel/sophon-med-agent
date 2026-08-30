# Sophon Medical Voice Agent

This repository contains the source code and model-conversion work for a hospital-oriented voice agent designed to run on Sophon BM1684X hardware.

The assistant, called **Xiaomai**, combines wake-word detection, speech recognition, a tool-using language model, local medical retrieval, robot navigation, and text-to-speech in one workflow.

## Main Features

- Chinese voice interaction with wake-word detection, ASR, and TTS
- Text and image conversations through Qwen3.5
- Local medical knowledge retrieval with safety checks and department suggestions
- Hospital navigation commands such as going to the pharmacy or emergency department
- OpenAI-compatible chat, embedding, and audio APIs
- Multi-session CLI and browser-based debugging tools
- Model conversion and validation scripts for Sophon TPU deployment

## Repository Layout

- `agent_server_source_archive/` - agent orchestration, API router, medical retrieval, voice services, navigation tools, and model runtime adapters
- `model-compilation-scripts-archive_with_source/` - TPU-MLIR conversion, compilation, and validation scripts

## Runtime Overview

```text
Microphone / Text / Image
          |
          v
Wake Word -> ASR -> Agent -> Medical Retrieval / Navigation / Utilities
                              |
                              v
                         TTS Response
```

The router exposes OpenAI-compatible endpoints on port `8000`, while the headless voice agent normally listens on port `8766`.

## Requirements

The full runtime requires:

- Linux with a compatible Sophon BM1684X runtime
- Python 3.10 and the dependencies listed in the service directories
- Compiled `.bmodel` files for the LLM, ASR, embedding, KWS, and TTS components
- The medical SQLite database and optional dense-vector index
- External robot navigation scripts for physical movement

## Archive Notice

This is a source archive. Large model weights, compiled deployment artifacts, medical databases, vector indexes, native libraries, and evaluation datasets are intentionally not included. Restore these files at the paths documented in `agent_server_source_archive/ARCHIVE_MODEL_PLACEHOLDERS.md` before attempting a complete deployment.

See `agent_server_source_archive/README.md` for detailed setup, API, and architecture documentation.
