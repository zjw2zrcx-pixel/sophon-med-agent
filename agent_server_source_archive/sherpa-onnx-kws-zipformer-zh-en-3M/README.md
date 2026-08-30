# 关键词识别（KWS）bmodel 部署

本目录的 sherpa KWS 模型是 Transducer，而不是单一网络。
`kws_transducer_chunk16_bm1684x_f32.bmodel` 是一个合并 bmodel，内部包含：

| graph | 固定输入 | 作用 |
|---|---|---|
| `kws_encoder_chunk16` | 1 路、45×80 fbank 和流式缓存 | 流式声学编码 |
| `kws_decoder_chunk16` | token 历史 `[1,2]` | RNN-T 预测网络 |
| `kws_joiner_chunk16` | `[1,320]` encoder/decoder 输出 | 输出 263 类 logits |

运行时固定为 45 帧 fbank 窗口、32 帧滑动；不要改用 `chunk-8` 模型或多路 batch。

## TPU 端运行要点

1. 在目标设备安装与 bmodel 芯片匹配的 SophonSDK/Sophon-SAIL，确认
   `python3 -c 'import sophon.sail'` 成功。
2. 使用 `sail.Engine("kws_transducer_chunk16_...bmodel", device_id, sail.IOMode.SYSIO)` 只加载一次，按 graph name 调用 encoder、
   decoder、joiner；通过 `get_graph_names()`、`get_input_names()` 核对 I/O 名称。
3. 前端仍在 CPU：16 kHz PCM → 80 维 fbank；保持每次 encoder 调用的 39 个
   输入（fbank、6 组 cache、embed state、processed lens），并将 39 个输出回写到
   下一次调用的 cache。
4. encoder 每帧输出的 `[1,320]` 送入 decoder/joiner，保留 Transducer
   beam/search 与关键词匹配逻辑。只有网络计算迁移到 TPU。

## 上板检查

```bash
model_tool --info kws_transducer_chunk16_bm1684x_f32.bmodel
```

## 服务器交付物

本目录可整体上传至目标服务器。安装匹配 BM1684X 的 SophonSDK/Sophon-SAIL 后，先执行 `model_tool --info kws_transducer_chunk16_bm1684x_f32.bmodel`，再由基于 SAIL 的程序按图名 `kws_encoder_chunk16`、`kws_decoder_chunk16`、`kws_joiner_chunk16` 调用。

`sail_kws_runner.py` 提供服务器使用的 CPU fbank + TPU encoder/decoder/joiner
封装，`sherpa_kws_server.py` 是可直接启动的自包含 HTTP 服务；`test.wav`
可用于板端烟测（会自动重采样到 16 kHz）。

## 当前生产解码与验收

`sail_kws_runner.py` 的神经网络主体始终由 `sophon.sail.Engine` 调用合并
`bmodel` 的 encoder、decoder、joiner 三图；CPU 只保留 fbank 和关键词 FSA
状态机。它使用 4 条活动路径、关键词加分 2.0、1 个 trailing blank，与原始
Sherpa KeywordSpotter 的生产参数一致，不会回退到 ONNX Runtime 全模型推理。

目标板必须在 **沙箱外** 使用已有运行时环境执行：

```bash
cd /data/structure/sherpa-onnx-kws-zipformer-zh-en-3M
/data/env310/bin/python verify_bmodel.py
/data/env310/bin/python verify_golden.py
/data/env310/bin/python verify_kws.py
/data/env310/bin/python sherpa_kws_server.py --host 127.0.0.1 --port 8004
```

`verify_golden.py` uses the self-contained `golden/test_wav_first_chunk/`
package to compare the Sherpa-compatible fbank and all three TPU graphs before
testing keyword search behavior. `verify_kws.py` then performs the full
45-frame-window/32-frame-hop streaming regression and requires exactly three
complete `小麦小麦` events on both the supplied 16 kHz golden waveform and
the bundled 44.1 kHz `test.wav`.

The CPU-side Sherpa keyword-decoder port is built as the local
`sail_keyword_search` pybind extension. Rebuild it after changing the C++
source (the deployed `.so` remains inside this directory):

```bash
cd /data/structure/sherpa-onnx-kws-zipformer-zh-en-3M/sail_keyword_search
/data/env310/bin/python setup.py build_ext --inplace --force
```

`test.wav` 是多次“小麦”录音。HTTP 生产链路应将其重采样为 16 kHz 单声道
PCM 并以 20 ms 块发送；对附带测试音频的验收要求为恰好三次“小麦小麦”事件，
静音 PCM 不得触发。部署目录自包含 `runtime/keywords.txt`、tokens 和 bmodel，
不依赖外部模型目录。
