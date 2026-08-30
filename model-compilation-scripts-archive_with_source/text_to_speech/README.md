# Melo VITS 中文/英文 TTS

本目录保存 Melo VITS 的原始 ONNX、静态拆分脚本、BModel 转换入口和混合运行时。当前可复现的部署方式不是“全模型上 TPU”，而是：

```text
文本前处理 → BModel encoder → CPU duration predictor + TorchScript controller
           → BModel flow/decoder → WAV
```

## 目录

- `hf_vits_melo_zh_en/`：原始 Hugging Face 模型与前端资源；主权重为 `model.onnx`。
- `native_50tk_256f/`：从原始 ONNX 导出 256-frame decoder 和 CPU controller 的脚本。
- `vits-melo-tts-zh_en/`：转换入口、CPU 测试工具及旧的 512-frame 运行时。
- `remote_deploy/vits_melo_50tk_256f/`：已有的 BM1684X FP32 部署参考包。

当前转换契约固定为 batch 1、最多 50 token、最多 256 声学帧。超过限制的文本应先切句。

## 转换为 BModel

先进入已安装 TPU-MLIR 的环境，并确保以下命令可用：

```bash
model_transform.py --help
model_deploy.py --help
```

拆分 ONNX 还需要 Python 包 `onnx` 和 `torch`。在 `text_to_speech` 根目录运行：

```bash
cd vits-melo-tts-zh_en
./convert_to_bmodel.sh
```

默认目标是 `bm1684x`、精度为 `F32`，输出到：

```text
vits-melo-tts-zh_en/build/bm1684x/
├── bmodel/
│   ├── vits_encoder_50_bm1684x_f32.bmodel
│   └── vits_flow_decoder_256_bm1684x_f32.bmodel
├── cpu_component/
│   ├── vits_dp_cpu_50.onnx
│   └── cpu_dynamic_controller_256f.pt
└── onnx/
    ├── encoder_b1_50tk.onnx
    └── decoder_50tk_256f.onnx
```

可以通过环境变量指定芯片、精度、原始权重目录和输出目录：

```bash
CHIP=bm1684x QUANTIZE=F32 \
SOURCE_DIR=../hf_vits_melo_zh_en \
OUTPUT_DIR=./build/bm1684x \
./convert_to_bmodel.sh
```

只拆分模型并生成 CPU 文件、不调用 TPU-MLIR：

```bash
./convert_to_bmodel.sh --prepare-only
```

说明：encoder 使用原始权重目录中已验证的静态拆分文件 `static_f32_50tk/encoder_b1_50tk.onnx`；decoder 和 controller 每次从 `model.onnx` 重新导出。CPU duration predictor 使用与该拆分契约配套的 `vits_dp_cpu_50.onnx`。

## 在 CPU 上运行 `.pt` 部分

`.pt` 文件只是动态时长展开和 latent 生成控制器，不会单独生成语音。它需要 encoder 与 duration predictor 的中间结果；完整合成仍需两个 BModel。

先做独立 CPU 冒烟测试：

```bash
python3 vits-melo-tts-zh_en/run_cpu_controller.py \
  vits-melo-tts-zh_en/build/bm1684x/cpu_component/cpu_dynamic_controller_256f.pt
```

期望输出包含 `z_shape: [1, 192, 256]`、`frame_mask_shape: [1, 1, 256]`，并且 `frames <= 256`。实际调用接口为：

```python
controller = torch.jit.load("cpu_dynamic_controller_256f.pt", map_location="cpu").eval()
with torch.inference_mode():
    z, frame_mask, y_lengths, duration = controller(
        m_p, logs_p, x_mask, logw,
        noise_scale, length_scale, latent_noise,
    )
```

其中 `latent_noise` 必须是 FP32 `[1, 192, 256]`；`m_p`、`logs_p`、`x_mask` 来自 encoder，`logw` 来自 `vits_dp_cpu_50.onnx`。若 `y_lengths[0] > 256`，必须切句，不能直接截断。

## 运行依赖

- 转换机：TPU-MLIR、Python 3、`onnx`、`torch`。
- BM1684X 设备：`sophon.sail`、`numpy`、CPU `onnxruntime`、`torch`。
- 音频采样率：44.1 kHz。

旧的 `vits-melo-tts-zh_en/hybrid_vits_runtime.py` 使用 512-frame BModel 和旧 controller，仅用于旧部署包；不要与本 README 的 256-frame 产物混用。
