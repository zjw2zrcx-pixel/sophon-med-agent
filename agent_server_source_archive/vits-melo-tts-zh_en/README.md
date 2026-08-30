# VITS Melo：50-token / 256-frame BM1684X 部署包

本目录现在是 256-frame 版本的独立运行目录，已替换原 512-frame decoder。

```text
文本前处理/G2P → TPU encoder → CPU duration predictor + 256f controller
                 → TPU flow + decoder → 44.1 kHz float32 WAV
```

## 文件

- `vits_encoder_50_bm1684x_f32.bmodel`：50-token FP32 encoder。
- `vits_dp_cpu_50.onnx`：FP32 duration predictor。
- `cpu_dynamic_controller_256f.pt`：原生 256-frame TorchScript controller，七输入、四输出。
- `vits_flow_decoder_256_bm1684x_f32.bmodel`：256-frame FP32 flow + decoder。
- `preprocess_assets/`：与模型配套的前端资源。
- `hybrid_vits_runtime.py`：运行入口。

## 固定契约

- token bucket：`50`
- frame bucket：`256`
- `sid=1`、`noise_scale=0.667`、`length_scale=1.0`
- controller 的 `latent_noise`：`float32 [1,192,256]`
- decoder 输入：`z [1,192,256]`、`y_mask [1,1,256]`、`y_mask4 [1,1,256,1]`、`condition [1,1,256]`、`speaker [1,256,1]`
- decoder 输出：`y_Tanh [1,1,131072]`

当 controller 预测帧数大于 256 时，运行时会报错；调用方应先切句或回退到独立的 512-frame 部署包。禁止将 512-frame latent 切片后伪装成 256-frame controller 输出。

## 调用

```python
import numpy as np
from hybrid_vits_runtime import HybridVitsRuntime

runtime = HybridVitsRuntime(".", device_id=0)
wav = runtime.synthesize_tokens(
    x=np.asarray([12, 34, 56], dtype=np.int32),
    tones=np.asarray([0, 0, 0], dtype=np.int32),
    sid=1,
)
```

需要 BM1684X Runtime 的 `sophon.sail`、CPU `onnxruntime` 和 `torch`。模型加载和推理必须在可见 BM1684X 设备的环境中执行。

## 检查

```bash
python3 inspect_bmodel.py
python3 test_cpu_stages.py
```
