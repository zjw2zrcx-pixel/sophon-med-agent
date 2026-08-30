# VITS Melo BM1684X：50-token / 256-frame 部署契约

当前目录已切换为原生 256-frame 链路。encoder、duration predictor、controller 和 decoder 来自同一份 `vits_melo_50tk_256f` 部署包。

## 模型与版本契约

```text
vits_encoder_50_bm1684x_f32.bmodel
vits_dp_cpu_50.onnx
cpu_dynamic_controller_256f.pt
vits_flow_decoder_256_bm1684x_f32.bmodel
```

controller 是七输入、四输出 TorchScript 图，`max_frames=256`，最后一个输入必须是显式的 `latent_noise [1,192,256] float32`。它输出 `z`、`frame_mask`、`y_lengths` 和 `duration`；`z` 与 `frame_mask` 原样传给 256-frame decoder，不做 512→256 截断。

## Encoder

输入：

```text
x          [1,50]       int32
x_lengths [1]           int32
tones      [1,50]       int32
sid        [1]          int32
```

输出：

```text
hidden     [1,192,50]
m_p        [1,192,50]
logs_p     [1,192,50]
x_mask     [1,1,50]
speaker    [1,256,1]
condition  [1,1,256]
```

## Controller

```text
m_p, logs_p, x_mask, logw
noise_scale [1]
length_scale [1]
latent_noise [1,192,256]
```

输出：

```text
z          [1,192,256]
frame_mask [1,1,256]
y_lengths  [1] int64
duration   [1,1,50] int64
```

## Decoder

五路输入：

```text
/Add_2_output_0                  [1,192,256]
/Cast_4_output_0                 [1,1,256]
/Unsqueeze_10_output_0           [1,1,256,1]
/enc_p/encoder/Transpose_output_0 [1,1,256]
/Unsqueeze_6_output_0            [1,256,1]
```

输出：

```text
y_Tanh [1,1,131072]
```

最终波形裁剪为 `y_lengths * 512` 个 float32 sample，采样率为 44,100 Hz。

## 限制与验收

当前 `calibration_tk50` 的 253 条样本预测帧数范围为 99–215，均在 256-frame bucket 内。生产调用仍必须检查 `y_lengths <= 256`；超过上限时切句或回退到另一个独立的 512-frame 部署包。

`hybrid_vits_runtime.py` 已使用 `cpu_dynamic_controller_256f.pt` 的显式 latent_noise 契约；不要恢复旧的六输入 controller 调用方式。
