# sherpa-onnx-kws-zipformer-zh-en-3M 模型的部署与推理

本文依据以下仓库文件整理：

- bmodel：`sherpa-onnx-kws-zipformer-zh-en-3M/kws_transducer_chunk16_bm1684x_f32.bmodel`
- TPU/CPU 推理封装：`sail_kws_runner.py`
- HTTP 服务：`sherpa_kws_server.py`
- token 与关键词：`runtime/tokens.txt`、`runtime/keywords.txt`
- CPU 关键词搜索扩展：`sail_keyword_search/sail_keyword_search.cc`
- 部署说明和验收：`README.md`、`verify_golden.py`、`verify_kws.py`

## 1. 部署形态

这是 BM1684X 上的流式 Zipformer Transducer 关键词识别部署。一个合并 bmodel 内部包含三个图：

| 图名 | 作用 |
|---|---|
| `kws_encoder_chunk16` | 根据当前 45×80 fbank 和流式 cache 生成声学状态 |
| `kws_decoder_chunk16` | 根据每条活动路径的最近两个 token 生成预测状态 |
| `kws_joiner_chunk16` | 合并 encoder/decoder 状态，输出 263 类 logits |

`server/config.toml` 将服务命名为 `sherpa-kws`，类型为 `kws`，监听 `127.0.0.1:8004`，并配置上述 bmodel、runtime 目录和设备号 0。Router 启动时会传入这些路径参数，但 `sherpa_kws_server.py` 为自包含部署，实际仍从自身目录解析 bmodel、`tokens.txt` 和 `keywords.txt`。

启动时服务在后台构造一次 `SailKwsRunner`，用来确认 TPU、SAIL 扩展和图都可用；成功后 `/health` 才返回 `ready`。真正创建的识别 session 会各自持有一个 `SailKwsRunner`。

## 2. SAIL 图加载和状态初始化

`SailKwsRunner.__init__()` 用：

```python
sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
```

加载 bmodel，然后检查三个固定图名。它读取 `tokens.txt` 建立 token ID 到 token 字符串的映射，再读取同目录 `keywords.txt`：文件中的 phone token 被映射为 token ID，形成生产关键词序列。当前部署包携带的目标短语是“小麦小麦”。

随后创建本地 `KeywordSearch` 扩展，参数来自 runner：

```text
keyword_score=2.0
keyword_threshold=0.18
max_active_paths=4
num_trailing_blanks=1
```

`reset()` 为 encoder 的每个输入按 SAIL shape 建立零 cache，清空 pending 音频、fbank 左上下文、decoder 状态缓存和关键词状态。

## 3. HTTP 输入和 session

服务提供：

```http
POST /v1/audio/keywords
Content-Type: application/json
```

请求体实际读取：

- `audio`：Base64 编码的 signed 16-bit little-endian PCM。
- `session_id`：可选；没有时生成 UUID。

服务把 PCM 解码为 int16，再转换为 float32 并除以 32768。代码按 16 kHz、单声道计算 block 时间：`len(pcm) / 32000.0`，因此调用方必须提供 16 kHz 单声道 16-bit PCM；服务端本身不做重采样和声道混合。仓库 README 要求外部调用方在发送前完成 16 kHz 单声道化，并建议按 20 ms 块发送。

同一 `session_id` 会复用同一个 `SailKwsRunner`，保留跨 HTTP 请求的 fbank pending、encoder cache 和关键词搜索状态。不同 session 独立维护状态。返回值包括本次输入块的起止时间和命中列表：

```json
{
  "session_id": "...",
  "hotword_hits": [],
  "block_start_s": 0.0,
  "block_end_s": 0.02
}
```

检测到完整关键词时，当前服务把结果报告为 `hotword_hits: ["小麦小麦"]`；未检测到时为空数组。

## 4. CPU fbank 前处理

`accept_waveform()` 先把新样本追加到 `pending`。当 pending 长度达到：

```text
(45 - 1) * 160 + 400 = 7440 samples
```

才处理一个 encoder 窗口。`_fbank()` 使用 `torchaudio.compliance.kaldi.fbank`，参数与 sherpa-onnx 1.13.4 的在线 fbank 契约一致：

```text
采样率       16000 Hz
mel bins     80
frame length 25 ms（400 samples）
frame shift  10 ms（160 samples）
low_freq     20 Hz
high_freq    -400 Hz
window       povey
dither       0
snip_edges   False
preemphasis  0.97
```

每个窗口生成 45 帧、每帧 80 维的 float32 fbank。首个窗口直接取前 45 帧；后续窗口会把上一窗口保留的 10 ms 左上下文拼到 raw 数据前面，再丢掉第一帧，使 chunk 边界的 fbank 帧与全局连续音频对齐。

## 5. 流式 encoder

处理一个窗口后，代码把 45×80 特征放到 encoder 输入 `x`，并把此前 encoder 输出的 cache 输入一并提交到 `kws_encoder_chunk16`。图输出的 cache 会按输入/输出名称顺序回写到 `self.cache`，`processed_lens` 强制转为 int32 保存。

encoder 每个窗口产生 8 个 40 ms 状态；代码对当前输出逐个调用关键词步骤。虽然注释提到 encoder 的 subsampled `processed_lens` 以 16 前进，但输入 fbank 的实际窗口滑动是 32 个 10 ms frame：

```text
窗口：45 fbank 帧
滑动：32 fbank 帧
重叠：13 fbank 帧
```

因此不会把 `processed_lens=16` 误解为输入每次只滑动 16 个 fbank 帧。

## 6. decoder、joiner 和 decoder cache

每个 encoder 输出状态都要针对当前关键词搜索的活动路径计算 logits。活动路径由 CPU `KeywordSearch.histories()` 返回，每条路径提供最近两个 token，组成 decoder 输入 `[1,2]`。

`_keyword_logits()` 对每个 history：

1. 如果该 history 不在 `keyword_dec_cache`，调用 `kws_decoder_chunk16` 得到 `decoder_out_Gemm`。
2. 以当前 encoder 状态 `[1,320]` 和 decoder 输出调用 `kws_joiner_chunk16`。
3. 取 `logit_Gemm` 得到该活动路径的词表 logits。

decoder 输出按二 token history 缓存；一旦关键词被接受，代码清空该 decoder cache，因为接受短语后 CPU 搜索状态被 reset，后续路径重新建立。

## 7. CPU 关键词搜索状态机

TPU 只计算神经网络 logits，Sherpa 风格的 modified beam/context graph 由本地 pybind 扩展 `sail_keyword_search` 完成。

`KeywordSearch.step(logits)` 的主要动作是：

1. 对每条活动路径和每个词表 token 计算 log probability。
2. 全局取最多 4 条候选路径。
3. 合并相同 token history，并用 log-add-exp 合并概率。
4. blank 和 `<unk>` 增加 trailing blank 计数；普通 token 推进关键词前缀状态。
5. 通过 failure table 支持关键词前缀失败后的回退，保留 Sherpa ContextGraph 的前缀匹配行为。
6. 当完整关键词已经匹配、trailing blank 条件满足、平均 token 概率达到阈值 0.18 时报告命中。

完整短语被接受后搜索状态立即 `Reset()`，所以一个连续音频里后续再次说出目标短语可以再次触发事件。代码中的 `new.extend(self.keyword_tokens)` 将命中的 token 列表返回给 Python；HTTP 层只把非空结果映射为固定字符串“小麦小麦”。

## 8. 请求到命中的完整链路

```text
Base64 PCM
  → int16 little-endian / 32768
  → session.pending
  → 达到 7440 samples
  → CPU Kaldi/Povey 80-bin fbank
  → TPU kws_encoder_chunk16
  → 每个 encoder state
  → CPU active-path histories
  → TPU kws_decoder_chunk16（按 history 缓存）
  → TPU kws_joiner_chunk16
  → CPU modified beam + ContextGraph keyword search
  → 命中后 reset，HTTP 返回 hotword_hits
```

没有达到一个完整 45 帧窗口的尾部样本会留在 session 的 `pending` 中，不会被零填充后立即送入 encoder；下一次同 session 请求会继续拼接。

## 9. 运行限制和验收

- 目标设备必须有与 bmodel 匹配的 SophonSDK/Sophon-SAIL，并且能导入 `sophon.sail`。
- `sail_keyword_search` 是部署目录内的 pybind 扩展；修改 C++ 后需要在目标 Python 环境重新 build。
- 当前服务使用 `runtime/keywords.txt` 和 `tokens.txt`，不依赖旧的外部 CPU 模型目录。
- 静音 PCM 不应触发关键词；仓库验收脚本要求供给的 16 kHz golden 波形和附带 `test.wav` 均得到指定的三次“小麦小麦”事件。
- `verify_golden.py` 先验证 fbank 与三个 TPU 图的 golden，再验证关键词搜索；`verify_kws.py` 验证完整的 45 帧窗口/32 帧滑动流式回归。

## 10. 直接调用示例

启动：

```bash
python sherpa-onnx-kws-zipformer-zh-en-3M/sherpa_kws_server.py \
  --host 127.0.0.1 --port 8004
```

发送一块 16 kHz、单声道、signed 16-bit little-endian PCM 时，先将音频字节 Base64 编码：

```bash
curl http://127.0.0.1:8004/v1/audio/keywords \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","audio":"<base64_pcm>"}'
```
