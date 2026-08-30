# VITS Melo：BM1684X 混合部署包

本目录可整体复制到服务器。部署链路如下：

```text
文本前处理/G2P → TPU encoder → CPU duration predictor + 动态帧控制 → TPU flow + decoder → WAV
```

原始 VITS ONNX 的随机和动态长度子图不能直接由 TPU-MLIR 编译。本包将它们拆到 CPU；编码器、flow、decoder 仍在 TPU，故不是完整 CPU 推理。

## 内容

- `vits_encoder_50_bm1684x_f32.bmodel`：固定 50 token 的 TPU 编码器。
- `vits_dp_cpu_50.onnx`：CPU 确定性时长预测器，输入来自编码器；与原 ONNX 的实际时长路径逐元素一致。
- `cpu_dynamic_controller.pt`：TorchScript，按预测时长展开先验并采样 latent。
- `vits_flow_decoder_512_bm1684x_f32.bmodel`：最多 512 声学帧的 TPU flow + decoder。
- `hybrid_vits_runtime.py`：SAIL、ONNX Runtime 和 Torch 的推理封装。
- `inspect_bmodel.py`、`test_cpu_stages.py`：上板检查与 CPU 冒烟测试。
- `compare_encoder_golden.py`、`compare_full_golden.py`：诊断用对比脚本；生产目录不包含 golden 向量，验收时需临时提供验证包。

## 服务器依赖与检查

需要 BM1684X Runtime 的 `sophon.sail`，以及 `numpy`、CPU `onnxruntime`、`torch`。不需要 TPU-MLIR、原始 ONNX 或转换中间产物。文本前处理资源已放入 `preprocess_assets/`。

```bash
cd bmodel/bm1684x
python3 inspect_bmodel.py
python3 compare_encoder_golden.py
python3 compare_full_golden.py
python3 test_cpu_stages.py
```

encoder 应显示 4 个 int32 输入、6 个 float32 输出；flow/decoder 应显示 5 个 float32 输入和 `[1,1,262144]` 输出。

## 调用

调用方应使用 `preprocess_assets/` 中的文本规范化、G2P、`tokens.txt` 和词典，生成 token ID 与 tone ID。`x` 和 `tones` 是有效 token（而非手工补齐后的矩阵），长度为 1 到 50：

```python
import numpy as np
from hybrid_vits_runtime import HybridVitsRuntime
runtime = HybridVitsRuntime('.', device_id=0)
wav = runtime.synthesize_tokens(
    x=np.asarray([12, 34, 56], dtype=np.int64),
    tones=np.asarray([0, 0, 0], dtype=np.int64), sid=1)
# wav 为 44,100 Hz float32 单声道波形
```

重要：原始 ONNX 元数据的 `speaker_id=1`，而 sherpa-onnx 对单说话人 Melo 会强制使用该值；`HybridVitsRuntime` 因此默认 `sid=1`。不要传入旧示例中的 `sid=0`。模型元数据 `add_blank=1`，前端必须在 phone/tone 序列首尾及每个元素之间插入 token/tone 0。
脚本会补齐到 50 token，但 `x_lengths` 仍为真实长度。CPU 计算真实声学帧数，再补齐至 512 帧的 TPU-B 桶，最后把波形裁剪到 `真实帧数 × 512`。

## 限制和性能

- 超过 50 token 必须分句；50 是 token 数，不是语音帧数。
- 若预测声学帧超过 512，脚本会报错，需要拆句或额外编译更大帧桶。
- CPU↔TPU 最大传输张量为 `[1,192,512]` FP32，约 384 KiB，不会形成带宽瓶颈。
- 已回传服务器数据：CPU 时长预测约 12.7 ms、动态控制约 8.8 ms；完整 VITS CPU 约 4.1 s，因此完整 CPU 路径不可用。
- VITS 原本带随机采样，重复调用不保证波形逐样本一致；可设置 `torch.manual_seed()` 固定 CPU 随机源。

验收时请分别覆盖中文、英文和中英混读，检查时长、截断与试听效果。
