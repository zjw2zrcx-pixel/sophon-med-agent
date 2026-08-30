# VITS 50tk / 512-frame 量化进度

最后更新：2026-08-02 00:23 CST。

## 256-frame FP32 对照桶（2026-08-02）

部署端基础测试表明，512-frame 桶对常见短句可能偏大，因此新增独立的
`quantized_variants/50tk_256f/f32/convert_to_bmodel.sh`。它以相同的静态 decoder
ONNX 用 TPU-MLIR 的 256-frame 输入形状重新导入并编译；Encoder、CPU duration
predictor 与 controller 保持同一套 50-token 资产。部署包为
`vits-melo-tts-zh_en_256f/`，其运行时在预测帧数超过 256 时明确报错，须在上游
切句或回退至 512-frame 桶。

已于 2026-08-02 在 TPU-MLIR `v1.0.0.dev-c3a57a2-20260428` 成功编译为
`vits_flow_decoder_256_bm1684x_f32.bmodel`。`model_tool --info` 已确认五个输入均为
256-frame 静态 shape、输出为 `[1,1,131072]`。编译器估算设备内存为 219,116,800 bytes，
相对现有 512-frame FP32 的 299,808,512 bytes 降低约 26.9%；该数值只用于候选筛选，
仍须以上板实测延迟、内存和音频验收为准。

该桶是 FP32 性能/内存对照，不替代当前 512-frame 的 INT8 混精工作；测试时应记录
短句覆盖率、切句次数、TPU 延迟和内存，并针对不超过 256 frame 的相同输入与
512-frame FP32 输出做数值和试听对照。

## 当前任务

为 BM1684X 的静态 FP32 VITS Decoder（batch=1、50 token、512 frame）寻找可用的 INT8 混合精度 BModel。目标是降低 Decoder 延迟/占用，同时保持端到端音频可用。

## 已确认的结论

- FP32 B1 Encoder / Decoder 已能正确编译并上板运行。
- B2/B4 动态 batch 可运行，但端到端每条仅约 1--2% 改善，TPU 内存显著增加；生产应优先保留 B1。
- 全 F16 未通过音频验收：Encoder 的 `m_p` / `logs_p` 已超过误差阈值，时长取整会变帧，端到端波形严重偏离。
- 先前全 INT8 PTQ 同样未通过；Decoder 是主要精度敏感瓶颈。
- Hugging Face 的 `model.int8.onnx` 为 ONNX Runtime 动态量化图，包含 `DynamicQuantizeLinear`、`ConvInteger`、`MatMulInteger`，当前 TPU-MLIR 不支持直接导入。
- 当前可行路径是：FP32 ONNX 静态切分 + TPU-MLIR 原生 calibration + 自动/手工混精 qtable。

## 资产位置

- 原始/静态模型与 BModel：`hf_vits_melo_zh_en/static_f32_50tk/`
- Golden vectors 归档：`golden_vectors.tar.gz`
- 已解压的量化工作区：`hf_vits_melo_zh_en/static_f32_50tk/quantization_work/golden_vectors/`
- 基础 MSE calibration table：`quantization_work/decoder_mse_32.cali`（837 条目）
- 自动混精 qtable：`quantization_work/decoder_int8mix_32.qtable`
- 自动混精 calibration table：`quantization_work/decoder_int8mix_32.cali`

## Golden vectors

归档含 160 条真实 FP32 链路样本：128 条 calibration、32 条 hold-out。全部固定 batch=1、50 token、512 frame、`sid=1`、`noise_scale=0.667`、`length_scale=1.0`、seed=20260727。每条含 Decoder 五输入、Decoder FP32 输出、波形及 CPU controller 中间结果。

本轮小试使用前 32 条 calibration 样本；hold-out 未参与校准。

## 自动混精搜索状态

正在执行：

```text
run_calibration.py decoder_b1.mlir
  --data_list decoder_calibration_32.txt
  --input_num 32 --inference_num 8
  --cali_method mse --search search_qtable
  --part_quantize N_mode
  --expected_cos 0.995 --min_layer_cos 0.99
  --max_float_layers 64 --chip bm1684x
  --fp_type F32 --mix_mode wi8ai8_fp
```

截至更新时已运行约 1 小时 47 分，工作进程约 376% CPU、RSS 约 15.6 GiB；此前 Swap 峰值约 15 GiB，当前约 12 GiB。最终混精 cali 已在 00:09 写出，但进程尚未退出。

当前 qtable 的 499 个 F32 条目是 `N_mode` 生成的初始 pattern 保护层，不是最终精简结论：401 个在 `/flow`、91 个在 `/dec`。源代码表明最终 qtable 会在敏感层逐项搜索、混精模型输出 cosine 验证、必要时回退层数调整之后才覆盖写出。

## 下一步与决策门槛

1. 等待搜索输出 `success search qtable` 并退出；确认 qtable 修改时间更新。
2. 统计最终回退层数和 mode 分布。若仍接近数百层，不编译交付候选：预期性能收益不足。
3. 若回退规模合理，使用 `decoder_int8mix_32.cali` 与最终 qtable 编译 B1 INT8 混精 BModel。
4. 用 32 条 hold-out 做验收：frame 数必须完全一致，Decoder cosine >= 0.995，波形长度一致，并测试实际 TPU 延迟/内存。
5. 若自动搜索长期无完成结果或最终回退过多，停止自动搜索，转为手工混精：仅量化稳定 Conv/MatMul，保护 flow 的 `Exp`/`Log`/归一化/关键 `Div` 和最终波形头；最多再试一轮。
6. 若手工混精仍无法通过音频验收，停止 PTQ，转向 Decoder QAT。

## 重要限制

- 不要将 B2/B4 作为当前性能优化方向。
- 不要以工具默认 cosine 容差作为音频可用标准；最终必须使用 Golden hold-out 的 frame 数、波形误差和试听验收。
- 所有新中间 MLIR、NPZ、cali/qtable 必须保留在 `quantization_work/`；部署包只放最终 BModel、CPU 资产、运行程序和验收记录。
