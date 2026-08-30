# VITS Melo BM1684X 部署诊断（待编译端修复）

日期：2026-07-27  
目标设备：BM1684X-SOC（TPU 0，`bm-smi` 可见；Sophon Driver/Lib 0.5.1）

## 结论

当前 **hybrid TPU VITS 不能作为线上 TTS 后端**。TPU bmodel 可以被
`sophon.sail` 成功加载并运行，输出 WAV 也非空，但试听仅能分辨零散音素，
无法理解为输入文本。该问题不是设备、WAV 容器、采样率或纯静音问题，而是
**文本前端输出与编译 bmodel 的输入契约不一致，或 hybrid 拆图的中间张量/随机
控制契约不一致**。

线上应继续使用 Ekho，直到本报告中的“编译端交付物”齐备并通过一致性测试。

## 当前设备与可用性

沙箱内访问 TPU 会失败；在沙箱外执行时正常：

```text
bm-smi: TPU 0 = 1684X-SOC, active, memory approximately 75 / 950 MB
```

下列 bmodel 已使用 SAIL 成功打开：

```text
vits_encoder_50_bm1684x_f32.bmodel
vits_flow_decoder_512_bm1684x_f32.bmodel
```

因此“不正确语音”不是 `sail.Engine`、设备号或 bmodel 加载失败造成的。

## 已部署 hybrid 图契约

### A. Encoder（TPU）

`vits_encoder_50_bm1684x_f32.bmodel` / graph `vits_encoder_50_full_conditions`

| 输入 | shape | dtype |
|---|---:|---|
| `x` | `[1, 50]` | int32（runtime 实际传入） |
| `x_lengths` | `[1]` | int32 |
| `tones` | `[1, 50]` | int32 |
| `sid` | `[1]` | int32 |

输出：

```text
hidden     /enc_p/encoder/Mul_3_output_0_Mul        [1,192,50]
m_p        /enc_p/Split_output_0_Split              [1,192,50]
logs_p     /enc_p/Split_output_1_Split              [1,192,50]
x_mask     /enc_p/Unsqueeze_2_output_0_Unsqueeze    [1,1,50]
speaker    /Unsqueeze_6_output_0_Unsqueeze          [1,256,1]
condition  /enc_p/encoder/Transpose_output_0_Transpose [1,1,256]
```

### B. Duration / dynamic controller（CPU，必要保留）

```text
vits_sdp_cpu_50.onnx
cpu_dynamic_controller.pt
```

CPU 冒烟测试可运行，固定随机输入下约 198–218 ms；测试用例经
`length_scale=0.5` 后得到 391 帧，处在 512 帧桶内。

### C. Flow + decoder（TPU）

`vits_flow_decoder_512_bm1684x_f32.bmodel` / graph `vits_flow_decoder_512`

```text
z       [1,192,512] float32
y_mask  [1,1,512]   float32
y_mask4 [1,1,512,1] float32
condition [1,1,256] float32
speaker   [1,256,1] float32
output y_Tanh [1,1,262144] float32
```

运行时将 CPU 输出按真实帧数填入 512 帧桶，并裁剪到
`真实帧数 × 512` 样本，采样率为 44,100 Hz。

## 可复现失败

文本：`您好，我正在为您服务。`

使用公开 Melo 前端生成 `phones` 与 `tones`，将 phone 符号按当前
`preprocess_assets/tokens.txt` 映射为 ID，调用 hybrid runtime 后：

```text
WAV: 44.1 kHz mono
时长: 1.533 s
RMS: 0.0822
```

音频非静音，但人工试听为不可理解的零散音素。此前的“小麦小麦”样本也表现
为约 0.85 s 的错误语音；该词只是诊断输入，不是 TTS 固定文本或热词逻辑的一部分。

已确认的错误尝试：

1. 不含 `_` padding、全零 tone：有波形，但语义错误。
2. 手工把声母 tone 设 0、韵母设 3/4：输出近似静音。
3. 使用公开 Melo 前端的 `_ x iao m ai ... _` 和其输出 tones：仍无法保证与
   此 bmodel 的训练/导出前端相同。

## 高概率根因（按优先级）

1. **前端版本/符号表不一致**：当前 `tokens.txt` 只有 112 个符号；公开 Melo
   的 `symbols.py` 使用多语言合并 symbol 表且 token ID 排序不同。即使将 phone
   字符串重新映射到 112-token 表，tone、语言 ID、特殊符号和清洗规则仍可能不同。
2. **tone ID 语义不一致**：公开 Melo 的中文 tone 输出为原始 0–5；导出图可能期望
   已偏移 tone ID、带语言偏移的 tone ID，或与 token 对齐方式不同。
3. **hybrid 拆图不等价**：`vits_sdp_cpu_50.onnx`、`cpu_dynamic_controller.pt` 与两个
   bmodel 必须来自同一次导出；任意一个版本不匹配都会使 latent `z` 无语义。
4. **随机输入契约不一致**：VITS 的 duration noise/latent noise、`noise_scale`、
   `noise_scale_w`、`length_scale` 以及随机数分布和种子必须和原始 ONNX 路径一致。

## 编译端必须交付的最小内容

请将以下内容与 bmodel 一起导出，版本必须锁定：

1. `frontend.py`：唯一权威的 `text -> phones/token_ids, tone_ids, sid` 实现；需支持中文、英文与中英混读。
2. （可选）`golden_vectors/`，每条包含：
   - `text.txt`
   - `x.npy`（真实长度，不含或明确说明 padding）
   - `tones.npy`
   - `sid.npy`
   - `encoder_outputs.npz`（上述六个中间输出）
   - `logw.npy`、`z.npy`、`y_lengths.npy`
   - `wav_reference.wav`（原始官方运行时产物）。这些向量属于验收材料，已从生产目录移除。
3. 明确的导出版本：原始模型 commit、Melo/sherpa-onnx 版本、torch/onnxruntime、TPU-MLIR、SophonSDK。
4. 固定随机策略：所有随机张量的 shape、distribution、seed，以及三个 noise scale 的语义。
5. bmodel 图 I/O 名称、dtype、静态 shape 的导出清单；特别确认 token/tone 是 `int32` 还是 `int64`。

建议至少提供以下 golden 文本：

```text
你好。
您好，我正在为您服务。
Hello world.
您好，please follow me.
```

## 编译端验收标准

在 x64 编译端和 ARM/BM1684X 端均执行同一向量：

1. 前端生成的 `x`、`tones` 逐元素一致。
2. Encoder 六个输出与 ONNX 对照（FP32 合理容差，如 max abs error <= 1e-4）。
3. CPU `logw`、`z`、`y_lengths` 一致；若随机则固定 seed 后比对。
4. flow/decoder 输出与 ONNX 对照，裁剪前后长度一致。
5. 最终 WAV 时长相同，且主观试听可辨；建议附加 mel/STFT 相似度阈值。

只有通过以上步骤，才能将 VITS 设置为 Agent 的主 TTS；否则必须保持 Ekho 回退。
