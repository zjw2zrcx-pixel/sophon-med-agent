# Qwen3-Embedding-0.6B 模型的部署与推理

本文依据以下实际文件整理：

- 推理实现：`Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/embed_sail.py`
- 模型说明与约束：同目录 `README.md`
- HTTP 服务：`server/qwen3_embedding_server.py`
- 服务编排：`server/router.py`、`server/config.toml`
- tokenizer：`Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/tokenizer/`

## 1. 部署配置与实际加载的模型

统一服务配置中的名称、地址和路径是：

```text
名称：qwen3-embedding
API 类型：embedding
地址：127.0.0.1:8006
设备：devid=0
tokenizer：../Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/tokenizer
```

配置目录名中同时放有 W4BF16 和 BF16 两个 bmodel，但 `server/qwen3_embedding_server.py` 的 `EXPECTED_MODEL_ARTIFACT` 固定为：

```text
qwen3_embedding_bf16_seq512_bm1684x.bmodel
```

服务启动时会检查传入路径的文件名和文件是否存在；文件名不是这个 BF16 文件时直接拒绝加载。因此当前 HTTP 服务实际使用的是非量化 BF16 artifact，而不是同目录中的 `qwen3_embedding_w4bf16_seq512_bm1684x.bmodel`。`embed_sail.py` 本身可以通过参数测试两种模型，但服务端做了明确的 BF16 选择。

如果由 `server/router.py` 启动，Router 根据 `config.toml` 启动 `qwen3_embedding_server.py`，然后轮询其 `/health`，等待状态从 `loading` 变为 `ready`。服务的模型加载在 FastAPI lifespan 中创建，构造 `Qwen3Embedding` 放入单线程执行器。

## 2. SAIL 引擎和 bmodel 网络

`Qwen3Embedding.__init__()` 使用：

```python
sail.Engine(str(bmodel_path), device_id, sail.IOMode.SYSIO)
```

加载 bmodel，并检查模型图包含：

- `embedding`
- `block_0` 到 `block_27`

因此一次完整推理固定执行 29 个图：先执行 token embedding，再依次执行 28 个 Transformer block。代码读取 `embedding` 输入 shape 的第二维作为序列长度，并强制要求为 512；该 bundle 是静态 batch=1、sequence=512，不能按请求动态缩短设备图的 shape，也不能并行处理同一个 bmodel 的多个样本。

代码为每个图预先创建 SAIL 的 input/output tensor map。由于实际 bmodel 张量是 BF16，代码没有使用可能触发 native 类型物化问题的 numpy-dict overload，而是通过 Tensor-map API 更新数据。

## 3. 服务入口和请求校验

服务暴露 OpenAI 风格接口：

```http
POST /v1/embeddings
Content-Type: application/json
```

请求 JSON 的实际处理规则：

- `model` 可以是 `qwen3-embedding-0.6b` 或兼容别名 `qwen3-embedding`。
- `input` 必须是非空字符串，或只包含字符串的非空列表。
- `dimensions` 默认 256，允许范围为 32 到 1024。
- `encoding_format` 默认 `float`，也可指定 `base64`。

模型未 ready 时返回 503，JSON 无法解析或参数不合法时返回 400，推理异常时返回 500。通过 Router 访问时，Router 的 embedding 路由会把请求转发到 8006 子服务；设备推理发生在该子进程内。

## 4. 单条文本的预处理

`encode(text, dimensions)` 使用 tokenizer 目录加载的 Hugging Face `AutoTokenizer`。初始化时明确设置 `padding_side="left"`，如果没有 pad token，则把 eos token 作为 pad token。

对一条文本，tokenizer 执行：

```text
padding="max_length"
truncation=True
max_length=512
return_tensors="np"
```

结果是 batch=1 的 `[1, 512]` `input_ids`。超过 512 个 token 的输入会被截断；短输入左侧补齐。`attention_mask.sum()` 得到有效 token 数 `valid_tokens`，全空输入会报错。

左 padding 的直接作用是：最后一个真实 token 总是在序列下标 `-1`，所以最终向量从 `hidden[0, -1, :]` 取，而不是根据有效长度再计算位置。

position IDs 是 `[1, 512]` 的 int32 数组：左侧 padding 区域为 0，最后 `valid_tokens` 个位置填入 `0..valid_tokens-1`。attention mask 初始为 `[1, 1, 512, 512]` 的 `-10000`，只对真实 token 的因果可见区域置 0：每个真实 query 只能关注从第一个真实 token 到自己为止的位置，左侧 padding 不参与注意力。

## 5. BF16 数据如何在设备上流转

SAIL system tensor 对 BF16 使用 uint16 的 bit representation。代码提供两个转换函数：

- `_f32_to_bf16()`：float32 视图转 uint32，右移 16 位，再转 uint16。
- `_bf16_to_f32()`：uint16 转 uint32，左移 16 位，再视图恢复 float32。

完整推理流程中的张量流转是：

1. `input_ids` 以 int32 更新到 `embedding` 图。
2. `embedding` 图输出的 BF16 hidden state 通过 `asnumpy()` 取出，再转成 float32。
3. 每个 block 输入前，将 float32 hidden 转成 uint16 BF16；position IDs 保持 int32；attention mask 也转成 BF16 bit representation。
4. block 输出再转回 float32，作为下一层 block 的输入。

这套转换是在 CPU 可见的 SYSIO tensor 上完成，随后由 `engine.process(graph, inputs, outputs)` 将对应图提交到 BM1684X。代码没有在两个 bmodel 之间做同时驻留；README 也说明比较 BF16 和 W4BF16 时应使用独立进程，避免占满原生 SAIL 资源。

## 6. 28 层设备推理和向量生成

`encode()` 的设备执行顺序为：

```text
tokenizer
  → 构造 input_ids / position_ids / causal attention_mask
  → embedding 图
  → block_0
  → block_1
  → ...
  → block_27
  → 取最后一个真实 token 的 hidden state
  → MRL 截断
  → L2 归一化
```

最后一步具体为：

```python
vector = hidden[0, -1, :dimensions].astype(np.float32)
vector /= np.linalg.norm(vector) + 1e-12
```

这里的 `dimensions` 是 Matryoshka Representation Learning（MRL）前缀维度。默认取前 256 维，也可以取 32 到 1024 之间的任意值。重要顺序是先截断到目标维度，再归一化；不能先归一化完整 1024 维后再截断，否则结果不是当前实现产生的向量。

TPU 编码计算仍按固定 512 序列图执行，降低到 256 维只降低结果存储和后续相似度计算成本，不改变这段 bmodel 的设备计算长度。

## 7. 批量请求的真实行为

服务端 `_encode_many()` 对输入列表逐条执行：

1. 先用 tokenizer 统计每条文本截断后的有效 token 数，累加为 `prompt_tokens`。
2. 调用 `encoder.encode(text, dimensions)` 生成向量。
3. 将每条向量转成 Python list。

这些文本是在单线程 executor 中串行处理的，因为导出的 bmodel 是静态 batch=1。请求之间不会并行占用同一个 `encoder`；多个 HTTP 请求也会排队进入同一个 executor。

## 8. HTTP 返回格式

`encoding_format="float"` 时，每条数据包含 float 数组；`encoding_format="base64"` 时，服务先把向量转为 float32 bytes，再 Base64 编码。返回结构包括：

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
  ],
  "model": "qwen3-embedding-0.6b",
  "model_variant": "bf16",
  "model_artifact": "qwen3_embedding_bf16_seq512_bm1684x.bmodel",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

实际数组长度等于请求的 `dimensions`，示例中的二维数组仅用于说明结构。

## 9. 医疗检索中使用时的输入约定

模型 README 给出的查询指令格式是：

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: 北京有什么著名景点？
```

本篇只说明模型服务本身；医疗数据库如何调用 embedding、如何与稀疏检索合并，见 `医疗混合搜索系统.md`。

## 10. 直接调用示例

启动子服务的参数由 Router 配置提供，也可以直接运行：

```bash
python server/qwen3_embedding_server.py \
  --host 127.0.0.1 --port 8006 \
  --model-path Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/model/qwen3_embedding_bf16_seq512_bm1684x.bmodel \
  --config-path Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512/tokenizer \
  --module-path Qwen3_embedding_0.6B/qwen3_embedding_w4bf16_bm1684x_seq512 \
  --devid 0
```

调用：

```bash
curl http://127.0.0.1:8006/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b","input":"北京是中国的首都","dimensions":256}'
```
