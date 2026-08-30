# Agent Server 源码归档说明

本归档保存 Agent、Router、模型服务适配器、推理源码、配置、tokenizer/词表和必要的小型运行时资源。
为控制体积，模型权重、设备编译产物、医疗数据库和大型评测数据均未包含。

## 需要外部恢复的模型文件

将对应文件放回原路径后，`server/config.toml` 和各服务的默认参数才能在匹配的 BM1684X/Sophon 环境中加载本地模型：

- `Qwen3.5/*.bmodel`
- `qwen3.5history/*.bmodel`
- `Qwen3_ASR/qwen3_asr.bmodel`
- `Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/model/*.bmodel`
- `sherpa-onnx-kws-zipformer-zh-en-3M/*.bmodel`
- `vits-melo-tts-zh_en/vits_encoder_50_bm1684x_f32.bmodel`
- `vits-melo-tts-zh_en/vits_flow_decoder_256_bm1684x_f32.bmodel`
- `vits-melo-tts-zh_en/vits_dp_cpu_50.onnx`
- `vits-melo-tts-zh_en/cpu_dynamic_controller_256f.pt`

压缩包中保留了模型目录、配置和 tokenizer；权重目录没有人为填入伪造的二进制文件。

## 未纳入的其他大文件

- `med_database/med_search.sqlite`、医疗向量 `.npy` 和原始医疗数据集
- `teacher_trajectories/`、大规模 benchmark 输出及运行日志
- `.so`、`.a`、`.o`、`build/` 和 Python 缓存
- 测试音频、图片、视频、抓包日志和内部 Git 元数据

## 已知路径注意事项

当前源码中的 `server/config.toml` 仍有 `../Qwen3_5/...` 路径，而仓库实际目录为 `Qwen3.5/`；本归档保留原始状态，未擅自修改运行逻辑。

完整医疗检索需要在目标环境恢复 `med_database/med_search.sqlite`，或按 README 中的命令重新构建索引。
