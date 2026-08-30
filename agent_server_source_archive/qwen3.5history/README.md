# Qwen3.5-4B BM1684X history-KV 模型包

本目录包含使用 TPU-MLIR `87edab4` 编译的 Qwen3.5-4B BModel、tokenizer/config，以及与该模型匹配的最新版 LLM-TPU Qwen3.5 推理 demo。

## 模型信息

- BModel：`qwen3_5_4b_agent_merged_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260804_125518.bmodel`
- 芯片：BM1684X
- 量化：W4BF16，group size 64
- `seq_length`：8192
- `chunk_length` / `MAX_INPUT_LENGTH`：1024
- 设备和 core：1 device / 1 core
- 编译选项：`--use_history_kv`
- BModel 版本：`B.2.2+v1.0.0.dev-87edab4-20260803`
- SHA-256：`16393f06b78ac8b7bf930419c5f144611222455838f15951eebb5190ec01eeba`

编译命令等价于：

```bash
llm_convert.py \
  -m /workspace/Project/jcs/models/qwen3_5_4b_agent_merged \
  -s 8192 -c bm1684x -q w4bf16 -g 64 \
  --num_device 1 --num_core 1 \
  --use_history_kv --chunk_length 1024 \
  --out_dir /workspace/Project/jcs/models/qwen3.5history
```

`model.log` 是 `model_tool --info` 的完整输出。本模型共有 76 个网络，其中包含：

- `block_0 ... block_31`：prefill 网络；
- `block_cache_0 ... block_cache_31`：逐 token decode 网络；
- `block_kv_3/7/11/15/19/23/27/31`：已有历史时，Full Attention 层使用的 history-prefill 网络；
- `embedding`、`embedding_cache`、`lm_head` 和 `vit`。

## 随包 demo

这些文件来自本地 LLM-TPU 仓库提交：

```text
8c55988e712d8d68ec90493bb21bdc42af8c0f47
2026-07-30 Merge pull request #171 from sophgo/xin.zhang
```

目录用途：

| 目录 | 用途 | 是否适配当前 BModel |
| --- | --- | --- |
| `demo/cpp_demo` | 普通 C++ 多轮对话，上一轮状态继续追加 | 是，推荐用于 agent 多轮上下文 |
| `demo/cpp_demo_share_prompt` | 保存固定前缀快照，每个问题前回滚到同一前缀 | 是，最接近 prefix cache |
| `demo/python_demo` | Python pipeline + C++/pybind 推理模块 | 是 |
| `demo/cpp_demo_pp` | 多卡 Pipeline Parallel 参考实现 | 否；需要 `--distribute_strategy pp` 生成的拆分 BModel |
| `demo/UPSTREAM_QWEN3_5_README.md` | 上游原始 Qwen3.5 文档 | 参考 |
| `demo/run_demo.upstream.sh` | 上游固定下载 2B 模型的脚本 | 仅参考，不要用于本 4B 模型 |

demo 内已经包含 CMakeLists、`chat.cpp/.hpp`、pipeline、头文件、PCIE/SoC tokenizer 静态库及测试媒体。

## HTTP 精确前缀缓存

History 模型由 `server/qwen3_5_history_server.py` 提供服务，使用
`suha.v1` 的 `system | user | history | attempt` 四槽请求。前端每次发送
完整槽位，后端使用实际 ChatML token ID 做精确前缀比较，并保留四个不可变
设备快照：

| 变化范围 | 恢复点 | 重新 prefill |
| --- | --- | --- |
| 完全相同 | A | 无 |
| 仅 attempt 变化 | H | attempt |
| history 变化 | U | history、attempt |
| user 变化 | S | user、history、attempt |
| system 变化 | 无 | 全部 |

缓存支持多个 Session，默认最多 4 个。超过数量时完整释放“距离上一次推理
完成最久”的 Session，再创建新 Session。所有 Session 的设备快照总预算是
`1073741824` bytes（1 GiB）；预算不足时同样优先淘汰其他最不活跃 Session，
单个快照仍无法放入时跳过该检查点而不影响本次推理。

当前前缀缓存路径只支持文本。真实图片请求会返回 400；图片的文字描述可以
由 Agent 放入 `history` 或 `attempt`。

## 编译和运行 C++ 多轮 demo

以下命令假设当前目录就是本模型目录：

```bash
cd demo/cpp_demo
cmake -S . -B build -DTARGET_ARCH=pcie
cmake --build build -j

./build/pipeline \
  -m ../../qwen3_5_4b_agent_merged_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260804_125518.bmodel \
  -c ../../config \
  -d 0
```

SoC 上在设备本机编译即可；CMake 会根据 `aarch64` 选择 `lib_soc`。如果系统没有普通 OpenCV，可按 demo 自带 README 启用 `/opt/sophon/sophon-opencv-latest`。

普通 demo 会在同一个 `Qwen3_5` 实例上持续追加 user、assistant 及后续 tool-return 对应的 token。交互中输入 `/clear`、`/new` 或 `/c` 会调用 `clear_history()`。

## 运行共享前缀 demo

`cpp_demo_share_prompt` 是当前最接近 prefix cache 的官方实现：先把 `--prompt` 做一次 prefill 并在 TPU 设备内存中保存快照；之后每个问题开始前恢复该快照，因此各问题共享 system/固定文档前缀，但彼此不累积。

```bash
cd demo/cpp_demo_share_prompt
cmake -S . -B build -DTARGET_ARCH=pcie
cmake --build build -j

./build/pipeline \
  -m ../../qwen3_5_4b_agent_merged_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260804_125518.bmodel \
  -c ../../config \
  -d 0 \
  --prompt "@story.txt"
```

`--prompt` 只用于建立共享缓存，不生成回答。`@file.txt`/`@file.md` 会内联文件内容；其他 `@path` 会被当作图片或视频。

需要注意：该 demo 是“固定前缀 + 多个相互独立问题”，并不是“前缀 + 不断增长的多轮 history”。若 agent 需要 `[system] + [user/history 持续追加]`，应使用普通 `cpp_demo` 的持续 session 模式；若大量请求共享不变 system/工具说明，再各自独立推理，则使用 `cpp_demo_share_prompt`。

## Python demo

上游 CMake 当前按 Python 3.10 查找 pybind11：

```bash
cd demo/python_demo
cmake -S . -B build
cmake --build build -j
cp build/*cpython*.so .

python3 pipeline.py \
  -m ../../qwen3_5_4b_agent_merged_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260804_125518.bmodel \
  -c ../../config \
  -d 0
```

Python pipeline 会输出 `Total Tokens`，也可以读取 `model.history_length`。输入 `/clear`、`/new` 或 `/c` 清空历史。

## KV-cache 是如何分配的

这里的“KV-cache”实际包含两类状态，因为 Qwen3.5 是 Full Attention 与线性/循环层混合结构：

1. Full Attention 层是第 3、7、11、15、19、23、27、31 层，保存随序列长度增长的 key/value；
2. 其他层不保存传统的逐 token KV，而是把 `past_key/past_value` 复用为 convolution state 和 recurrent state。

`Qwen3_5::init()` 加载 BModel 后，缓存直接绑定到 BMRuntime 网络 stage 的设备内存：

- Full Attention：`block_cache_i` 的 `input_mems[3]` 和 `input_mems[4]`；
- 线性层：`block_cache_i` 的 `input_mems[1]` 和 `input_mems[2]`；
- history 模式另行申请 `SEQLEN * HIDDEN_SIZE * sizeof(uint16_t)` 的长输入临时 buffer。

因此缓存主体位于 TPU device memory，不是 Python/C++ 的普通 host vector。BModel 中的 shape 决定其最大容量；本模型上限是 8192 token，不能在运行时把它无损扩展到 16K。

## Prefill 和 decode 如何使用缓存

### 首次或追加 prefill

history 模式下 `forward_first()` 转入 `forward_first_with_kv()`。输入按最多 1024 token 切块：

- 没有旧 KV 时运行普通 `block_i`；
- 已有旧 KV 时，Full Attention 层运行 `block_kv_i`，并把旧 key/value 作为额外输入；
- 新生成的 Full Attention KV 按 `old_kvlen * KV_BYTES` 偏移追加到长期缓存；
- 线性层读取旧 conv/recurrent state，完成本 chunk 后覆盖为新状态；
- `history_length` 随输入及生成 token 增长。

这正是下一轮 user prompt 或 tool return 能直接在上一轮缓存后继续推理的基础。调用方必须同时保证 chat template、position ids 和消息边界与原序列完全一致。

### 逐 token decode

`forward_next()` 先调用 `embedding_cache`，随后逐层运行 `block_cache_i`：

- Full Attention 层按 `(history_length - 1) * KV_BYTES` 写入当前 token 的 key/value；
- 线性层原地更新 conv/recurrent state；
- 最后运行 `lm_head` 并递增 `history_length`。

## 清除、快照和手动注入

### 清除

可以清除。`clear_history()` 会把所有层的 key/value 或线性状态清零，并把 `history_length` 置 0。普通和 Python demo 都把 `/clear`、`/new`、`/c` 映射到此调用。

### 保存/恢复 prefix

`cpp_demo_share_prompt` 新增了：

- `save_share_prompt()`：在 TPU 上申请备份内存并执行 device-to-device copy；
- `restore_share_prompt()`：把备份复制回工作缓存并恢复 `history_length`；
- 每轮问题前自动 restore，因此不需要重算共享前缀。

Full Attention 层只备份 `share_length * KV_BYTES` 的已用部分；线性层必须备份完整的 conv/recurrent state。

当前实现适合“启动后只保存一次、长度固定的共享前缀”。备份 buffer 第一次保存时按当时前缀长度分配；不要在同一实例上再次保存一个更长前缀，除非先修改代码以重新分配 backup。

### 手动加入或导入 KV

没有稳定的公开 KV 序列化/反序列化接口。技术上可以仿照 shared-prompt demo 对设备内存执行 D2D/H2D copy，但必须同时满足：

- 模型、量化、层数、shape 和 BMRuntime 内存布局完全一致；
- tokenizer 输出、chat template 和 position ids 完全一致；
- Full Attention KV 与所有线性层状态必须成套恢复；
- 同步恢复准确的 `history_length` 和位置状态。

只复制传统 K/V、遗漏线性层状态，或把另一段文本的 cache 接到当前 token 序列上，都会得到静默错误结果。远程服务中建议封装 `save/restore/clear`，不要向外暴露裸设备地址。

也不建议从当前 BModel 中手工删除 `block_kv_*`。若不需要 history，应重新编译且不传 `--use_history_kv`；否则网络计数、运行时检测和 prefill 路径可能不一致。

## 能否查看 KV-cache 命中

当前 LLM-TPU 实现没有 vLLM 风格的 prefix hash、block table、LRU，也没有 `cache_hit/cache_miss` 计数器。因此不存在自动比较新 prompt 前缀并报告“命中”的动作。

可以观测的是：

- 启动打印 `History Support: True`：BModel 含 `block_kv_*`；
- `history_length > 0`：本次 prefill 会基于现有状态追加；
- Python 的 `model.history_length` 或 demo 打印的 `Total Tokens`：当前已占用长度；
- shared-prompt demo 打印 `Shared prompt saved...`：快照已建立；其后每轮固定执行 restore，可视为由程序逻辑保证的 100% 共享前缀复用，而不是哈希命中。

若需要正式命中指标，可以在服务层为 prefix 计算 token-id hash，并在调用 `restore_share_prompt()` 时增加 hit 计数；hash 必须基于最终 token ids，而不能只基于原始字符串。

## 用于远程 agent 服务的建议

### 持续多轮 session

可以保留上一轮缓存并在其上直接处理新 user prompt 或 tool return，条件是：

1. `Qwen3_5` runtime 实例不能销毁；
2. 请求必须路由回同一个实例、设备及缓存；
3. agent 只 tokenization 新追加的消息，不要把整个历史再次 prefill；
4. position ids 和 chat-template 边界持续递增；
5. 接近 8192 token 时主动压缩上下文或 `clear_history()`。

当前普通 demo 在容量检查中预留 128 token，并在历史过满时自动 clear。生产服务最好由 agent 显式决定“总结后重建”或“新会话”，不要依赖静默清除。

### 固定 system prefix

可以用 shared-prompt demo 的 snapshot/restore 作为起点：

```text
启动实例
  -> prefill(system + 固定工具描述)
  -> save snapshot
请求到达
  -> restore snapshot
  -> prefill(user/history/tool return)
  -> decode
```

但当前 demo 每次都会回滚到固定前缀，不会保存每个用户各自不断增长的分支。多用户生产服务需要实现 session→snapshot/runtime 的映射，并评估每份 snapshot 的 TPU 显存成本。

### 生命周期限制

缓存只存在于当前进程和 TPU device memory 中：

- 进程退出、runtime `deinit()`、设备复位或模型重新加载后缓存消失；
- 当前 demo 不支持跨机器、跨进程持久化；
- 单个 `Qwen3_5` 实例是可变状态，不应让多个请求无锁并发访问；
- 若做实例池，必须启用 session affinity，或在每次请求前恢复对应的完整快照。

## 源码入口

- 普通 history 实现：`demo/cpp_demo/chat.cpp`
- 普通多轮编排：`demo/cpp_demo/pipeline.cpp`
- prefix 快照实现：`demo/cpp_demo_share_prompt/chat.cpp`
- 每轮回滚编排：`demo/cpp_demo_share_prompt/pipeline.cpp`
- Python binding：`demo/python_demo/chat.cpp`
- Python 对话层：`demo/python_demo/pipeline.py`
