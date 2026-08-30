# KWS 首块 golden：`test.wav`

此目录用于将部署端链路逐层对齐到编译端参考：

`test.wav` → fbank → encoder → decoder/joiner → 关键词 FSA。

它对应同级目录中的 `test.wav` 与编译端导出的首块参考数值；目标部署不需要、也不包含参考模型文件或重新编译工具。

## 文件

| 文件 | 内容 |
| --- | --- |
| `fbank_45.npy` | 第一次 encoder 调用的 fbank，float32，shape `[1, 45, 80]`。 |
| `encoder_decoder_joiner_golden.npz` | 首块参考结果和全部 encoder 输入状态。关键键为 `encoder_out` `[1,8,320]`、`decoder_y`、`decoder_out`、`joiner_encoder_out_t0`、`joiner_logits_t0` `[1,263]`。 |
| `resampled_16k_mono.npy` | 可选的 16 kHz float32 单声道波形，可优先用于隔离重采样差异。 |
| `feature_config.json` | 完整、机器可读的音频和 fbank 配置。 |

`encoder_input_*` 键是第一次调用的零初始化缓存。参考中的 `processed_lens` 为 `int64 [0]`；bmodel 图的同名输入为 `int32 [0]`，部署端应按 bmodel 的 `int32` 签名传入。

## 固定前端与关键词参数

* sherpa-onnx：`1.13.4`
* 重采样：`sherpa_onnx::LinearResample`，44.1 kHz → 16 kHz，cutoff=7920 Hz，`num_zeros=6`。
* fbank：80 bins、25 ms 窗、10 ms 帧移、Povey window、20 Hz 至 `-400` Hz、dither=0、`snip_edges=false`、DC 去除、preemphasis=0.97、FFT round-to-power-of-two，输入为归一化 float（不乘 32768）。
* 关键词搜索：`max_active_paths=4`、`score=2.0`、`threshold=0.18`、`num_trailing_blanks=1`。

## 对齐顺序

1. 先比较部署端 fbank 的前 45 帧与 `fbank_45.npy`（建议最大绝对误差不大于 `1e-4`）。
2. 将 npz 内所有 `encoder_input_*` 喂给首个 TPU encoder；比较 `encoder_out`。FP32 bmodel 可先以 `1e-4` 为参考阈值。
3. 以 `decoder_y` 运行 TPU decoder，并以 `joiner_encoder_out_t0` 与所得 decoder 输出运行 joiner；比较 `decoder_out` 与 `joiner_logits_t0`。
4. 上述数值一致后，若仍有唤醒词差异，问题位于关键词 FSA 的 token 历史、分数累积或流式时序，而非 bmodel 编译。
