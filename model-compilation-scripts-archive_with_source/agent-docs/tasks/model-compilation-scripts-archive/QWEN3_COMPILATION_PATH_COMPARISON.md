# Qwen3.5 编译通路对照

更新时间：2026-08-21

本文对照 `sophon_project/tpu-mlir` 基线提交
`87edab473868fa4990c33f81b4fe602d85287180` 与当前工作区改动，重点记录
`models/Qwen3.5_4B_convert.sh` 使用的 Qwen3.5 4B text-only 编译路径。
本文只记录脚本、源码调用关系和产物类别，不保存模型权重或编译产物。

## 一、共同入口与参数

入口脚本：

```text
models/Qwen3.5_4B_convert.sh
  -> ${TPUC_ROOT}/python/tools/llm_convert.py
```

当前脚本参数：

```text
-m /workspace/Project/jcs/models/Qwen3.5_4B
-s 8192
-c bm1684x
-q w4bf16
-g 64
--num_device 1 --num_core 1
--use_history_kv --chunk_length 1024
--text_only
--out_dir /workspace/Project/jcs/models/bmodel_qwen3.5_4b_history_without_vit
```

两条通路都经过以下公共阶段：

1. `llm_convert.py` 解析模型路径、序列长度、芯片、量化方式、设备数、
   history-KV 和输出目录。
2. 通过 `config.model_type == qwen3_5` dispatch 到
   `llm.Qwen3_5Converter.Qwen3_5Converter`。
3. 加载 `config.json` 和 safetensors 模型句柄。
4. `Qwen3_5Converter.__init__()` 设置 Qwen3.5 的 MRoPE、线性注意力、
   recurrent state、VIT 配置和 `dynamic=True`。
5. `LlmConverter.run()` 创建输出目录，先生成 MLIR，再调用 bmodel 编译，
   最后合并 bmodel 并清理中间 `.npz`（`--only_mlir` 除外）。
6. `execute_tasks()` 将 deploy 命令写入 `task.txt`，通过 GNU parallel
   并行调用 `model_deploy.py`，之后 `combine()` 生成最终组合模型。

## 二、原本的 Qwen3.5 编译通路（基线）

基线中 `Qwen3_5Converter.__init__()` 固定：

```text
self.do_vit = True
self.dynamic = True
```

因此即使用户只需要文本模型，也会走完整多模态路径。

### 2.1 基线 MLIR 生成顺序

`LlmConverter.gen_all_mlir()` 的正常模式顺序为：

```text
1. gen_vit_mlir()
   -> vit.mlir / ViT 权重文件

2. gen_embedding_lmhead_mlir()
   -> embedding.mlir
   -> embedding_cache.mlir
   -> lm_head.mlir

3. gen_sample_head_mlir()
   -> 仅在 do_sample 或 do_lora 导致 lmhead_with_topk=False 时生成

4. for layer_id in range(num_layers): gen_block_mlir(layer_id)
   -> block_<id>.mlir
   -> block_cache_<id>.mlir
   -> 若 use_history_kv：block_kv_<id>.mlir
```

Qwen3.5 的 `gen_block_mlir()` 根据 `layer_types` 分派：

```text
full_attention      -> gen_block_full_attn_mlir()
linear_attention    -> 生成 Qwen3.5 gated-delta-rule 线性注意力块
```

基线线性注意力块使用：

```text
Top.ChunkGatedDeltaRuleOp
  + triu_mask / strict_triu_mask / tril_mask / eye
  + recurrent state / convolution state
  -> block_<id>.mlir 或 block_cache_<id>.mlir
```

### 2.2 基线 bmodel 编译顺序

`LlmConverter.compile_all()` 将上述 MLIR 转换为 bmodel：

```text
1. compile_vit()
2. compile_common("embedding", with_size=True)
3. compile_common("embedding_cache")
4. 可选：compile_common("lm_head_lora")
5. 可选：compile_common("embedding_lora")
6. 可选：compile_common("embedding_cache_lora")
7. compile_lm_head()
8. 若无 top-k：compile_greedy_head() + compile_sample_head()
9. 对每个 layer_id：
   9.1 compile_block(layer_id)
   9.2 compile_block_cache(layer_id)
   9.3 若 use_history_kv：compile_block_kv(layer_id)
10. 若 chunk_length > 0：对每层生成并编译 block_cache_<id>_<stage>
11. execute_tasks()
12. 删除非 config 目录下的 `.npz`
13. combine() 合并全部 bmodel
```

基线的核心特点是：ViT、embedding、lm_head 和全部 transformer block 都
参与完整构建；`--use_history_kv` 只是在每层额外增加 history-KV 阶段，
并不会移除 ViT 或 embedding/lm_head。

## 三、改动后的 Qwen3.5 编译通路

当前工作区对入口和转换器增加了 `--text_only`。在本次脚本中该参数为真，
因此：

```text
llm_convert.py --text_only
  -> Qwen3_5Converter.__init__()
  -> self.do_vit = not args.text_only
  -> self.do_vit = False
  -> 不生成、不编译 ViT
```

同时，`qwen3_5` dispatch 配置仍强制 `dynamic=True`，并保留
`--use_history_kv` 的约束：动态编译、`chunk_length=1024`、
`max_input_length=1024`（由 history-KV 规则推导）。

### 3.1 改动后的正常 MLIR 生成顺序

本次脚本不使用 `--only_mlir`，所以当前顺序为：

```text
1. 跳过 gen_vit_mlir()

2. gen_embedding_lmhead_mlir()
   -> embedding.mlir
   -> embedding_cache.mlir
   -> lm_head.mlir

3. 若 lmhead_with_topk=False：gen_sample_head_mlir()

4. for layer_id in range(num_layers): gen_block_mlir(layer_id)
   -> block_<id>.mlir
   -> block_cache_<id>.mlir
   -> 若 use_history_kv 且该模型支持：block_kv_<id>.mlir
```

虽然 `gen_vit_mlir()` 仍在 `Qwen3_5Converter` 中实现，
`if self.do_vit` 使它在 text-only 模式不进入生成队列。

### 3.2 改动后的正常 bmodel 编译顺序

```text
1. 跳过 compile_vit()
2. compile_common("embedding", with_size=True)
3. compile_common("embedding_cache")
4. 可选：compile_common("lm_head_lora")
5. 可选：compile_common("embedding_lora")
6. 可选：compile_common("embedding_cache_lora")
7. compile_lm_head()
8. 若无 top-k：compile_greedy_head() + compile_sample_head()
9. 对每个 layer_id：
   9.1 compile_block(layer_id)
   9.2 compile_block_cache(layer_id)
   9.3 若 use_history_kv：compile_block_kv(layer_id)
10. 对 decode_chunk_list 编译 block_cache_<id>_<stage>
11. execute_tasks()
12. 删除非 config 目录下的 `.npz`
13. combine() 合并文本侧 bmodel
```

### 3.3 改动后的线性注意力实现路径

普通 prompt/cache 阶段仍使用原有 chunk kernel：

```text
Qwen3_5Converter.gen_block_by_length(strict_verify=False)
  -> Top.ChunkGatedDeltaRuleOp
  -> Top/Tpu lowering
  -> BM1684X codegen
  -> model_deploy.py
```

当前改动新增了严格验证/transactional 场景的另一条路径：

```text
gen_block_verify_strict()
  -> gen_block_by_length(strict_verify=True)
  -> Top.SequentialRecurrentGatedDeltaRuleOp
  -> recurrent_steps + conv_steps 输出
  -> SequentialRecurrentGatedDeltaRule lowering/interface
  -> BM1684X sequential kernel/codegen
  -> block_verify_strict_<id>.bmodel
```

对应新增源码包括：

```text
lib/Conversion/TopToTpu/BM1684X/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Top/Interfaces/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/BM1684/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/BM1684X/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/CV18xx/SequentialRecurrentGatedDeltaRule.cpp
lib/Dialect/Tpu/Interfaces/Common/SequentialRecurrentGatedDeltaRule.cpp
lib/PplBackend/src/sequential_recurrent_gated_delta_rule.cpp
lib/PplBackend/src_dyn/sequential_recurrent_gated_delta_rule_ctrl.c
```

本次 `Qwen3.5_4B_convert.sh` 没有打开 `--speculative_mode`，所以不会进入
`block_verify_strict_*`、`verify_head_*` 或 transactional 编译分支；这些是
同一改动分支提供的扩展通路。

## 四、两条通路对照

| 阶段 | 原本基线 | 改动后本次 text-only 脚本 |
|---|---|---|
| 模型入口 | `llm_convert.py` | 同一入口 |
| converter | `Qwen3_5Converter` | 同一 converter |
| dynamic | Qwen3.5 强制 dynamic | 保持强制 dynamic |
| ViT MLIR/bmodel | 始终生成、编译 | `--text_only` 后跳过 |
| embedding | 生成并编译 | 保留 |
| lm_head | 生成并编译 | 保留 |
| transformer blocks | 全部生成并编译 | 保留 |
| history-KV | 额外生成 `block_kv_<id>` | 保留，且由 `chunk_length=1024` 驱动 |
| decode chunk | 按 `decode_chunk_list` 编译 | 保留，受 gate-only 环境变量控制 |
| linear attention 普通阶段 | `ChunkGatedDeltaRuleOp` | 保持不变 |
| sequential/transactional 阶段 | 无 | 新增，可生成 `block_verify_strict_<id>` |
| 输出目录 | 脚本指定的目标目录 | `bmodel_qwen3.5_4b_history_without_vit` |

## 五、验证与边界

- 对照依据：基线 `git show HEAD:...`、当前工作区源码和
  `models/Qwen3.5_4B_convert.sh`。
- 已验证入口 shell 脚本通过 `bash -n`。
- 已验证 TPU-MLIR 当前工作区通过 `git diff --check`。
- 未执行实际模型编译；因此本文描述的是源码控制流和计划产物，
  不是一次成功编译的运行日志。
- 完整 diff 未复制到归档；如需恢复精确差异，使用
  `git -C sophon_project/tpu-mlir diff`。
