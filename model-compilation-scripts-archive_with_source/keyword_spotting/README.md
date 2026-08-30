# sherpa-onnx 中文/英文关键词识别

本目录包含 sherpa-onnx 流式 Zipformer Transducer 的原始 ONNX、CPU 麦克风测试程序和 BM1684X BModel 转换脚本。模型由三张网络组成：

```text
16 kHz PCM → CPU fbank/流状态 → encoder → decoder → joiner → CPU 关键词搜索
                                  TPU / ONNX Runtime
```

## 目录

- `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/`：encoder、decoder、joiner 原始 ONNX、token、词典和测试音频。
- `kws_microphone.py`：使用 sherpa-onnx CPU provider 的实时麦克风测试。
- `keywords_raw.txt`、`keywords.txt`：完整关键词“小麦小麦”的原文和 token 文件。
- `keywords_single_raw.txt`、`keywords_single.txt`：单关键词“小麦”的原文和 token 文件。

当前转换契约固定为 batch 1、`chunk=16`、`left-context=64`。encoder 每次接收 `[1,45,80]` fbank 和 38 个流式状态输入；不能与 chunk-8 ONNX 混用。

## CPU 运行 ONNX

需要 Python 包 `sherpa-onnx`、`numpy`、`sounddevice`，系统还需 PortAudio；使用 PulseAudio 后端时需要 `parec`。项目现有环境名为 `jcs`。

列出麦克风并运行：

```bash
cd keyword_spotting
conda run -n jcs python kws_microphone.py --list-devices
conda run -n jcs python kws_microphone.py --device 设备ID
```

WSLg/PulseAudio 可使用：

```bash
conda run -n jcs python -u kws_microphone.py \
  --backend pulse --pulse-source RDPSource --vad-debug
```

启动后说“小麦”或“小麦小麦”。常用调节参数：

- `--threshold 0.18`：关键词触发阈值；误触发时提高，漏检时降低。
- `--score 2.0`：关键词分数。
- `--vad-threshold 0.012`：RMS 语音检测阈值；设为 `0` 可关闭 VAD。
- `--threads 2`：CPU 推理线程数。

修改 `keywords_raw.txt` 后重新生成 token：

```bash
conda run -n jcs sherpa-onnx-cli text2token \
  --tokens sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
  --tokens-type phone+ppinyin \
  --lexicon sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/en.phone \
  keywords_raw.txt keywords.txt
```

`keywords_single_raw.txt` 可用同一命令生成 `keywords_single.txt`。

## 转换为 BModel

进入已安装 TPU-MLIR 的环境，确认以下命令可用：

```bash
model_transform.py --help
model_deploy.py --help
bmodel_combine.py --help
```

然后运行：

```bash
cd keyword_spotting/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
./convert_to_bmodel.sh
```

也可以指定芯片和精度：

```bash
./convert_to_bmodel.sh bm1684x F32
```

默认输出为：

```text
bmodel/bm1684x/kws_transducer_chunk16_bm1684x_f32.bmodel
```

该文件是包含以下三张图的组合 BModel：

- `kws_encoder_chunk16`：`[1,45,80]` fbank、cache 和位置状态输入，输出 `[1,8,320]` 及新 cache。
- `kws_decoder_chunk16`：最近两个 token `[1,2]`，输出 `[1,320]`。
- `kws_joiner_chunk16`：encoder/decoder 的两个 `[1,320]` 输入，输出 263 类 logits。

可通过环境变量修改模型和输出位置：

```bash
MODEL_DIR=/path/to/original_onnx \
OUTPUT_DIR=/path/to/output \
./convert_to_bmodel.sh bm1684x F32
```

检查产物：

```bash
model_tool --info bmodel/bm1684x/kws_transducer_chunk16_bm1684x_f32.bmodel
python3 bmodel/bm1684x/verify_bmodel.py \
  bmodel/bm1684x/kws_transducer_chunk16_bm1684x_f32.bmodel
```

## CPU/TPU 分工

BModel 只包含 encoder、decoder 和 joiner。16 kHz 音频采集、80 维 fbank、38 个 cache/状态的初始化与回写、Transducer 搜索、关键词 FSA 和 VAD 仍在 CPU。

现有 `kws_microphone.py` 使用 sherpa-onnx 直接运行 ONNX，不能把路径简单替换成 `.bmodel`。TPU 部署需使用 `sophon.sail` 分别调用组合 BModel 中的三张图，并在 CPU 侧实现相同的流式状态和关键词搜索逻辑。

目录中的 `*.int8.onnx` 含 TPU-MLIR 当前转换链路不支持的动态量化算子。INT8 BModel 应从 FP32 ONNX 配合真实 16 kHz 关键词音频生成校准表；不要直接用这些 INT8 ONNX，也不要使用随机数据校准。
