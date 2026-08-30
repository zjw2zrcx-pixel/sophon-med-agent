# BM1684 OpenAI-Compatible Model Server

将 BM1684 上运行的 Qwen3.5-VL（多模态对话）与 Qwen3-ASR（语音识别）模型封装为 OpenAI 兼容的 HTTP API 服务。

## 架构概览

```
                    ┌──────────────────────┐
                    │   Router (8000)       │
                    │   /v1/chat/completions│
                    │   /v1/audio/transcrip.│
                    │   /v1/models          │
                    └──────────┬───────────┘
                               │ HTTP 反向代理
              ┌────────────────┼────────────────┐
              │                 │                │
     ┌────────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐
     │ Qwen3_5 Server │ │ Qwen3_ASR    │ │  未来模型...  │
     │   (8001)        │ │  (8002)       │ │  (800x)       │
     │  独立进程       │ │  独立进程      │ │  独立进程     │
     └────────────────┘ └──────────────┘ └──────────────┘
```

每个模型服务器运行在独立进程中，拥有各自的 `chat.so` 动态链接库和显存空间。Router 根据 `model` 字段将请求转发到对应的服务器。

## 目录结构

```
structure/
├── Qwen3_5/
│   ├── python_demo/
│   │   ├── pipeline.py          # 原始 demo + 新增 generate() 方法
│   │   ├── chat.cpython-*.so   # Qwen3_5 专属动态链接库
│   │   └── ...
│   ├── config/                   # tokenizer / processor 配置
│   ├── qwen3.5.bmodel           # 模型权重
│   └── run.sh
│
├── Qwen3_ASR/
│   ├── python_demo/
│   │   ├── pipeline.py          # 原始 demo + 新增 transcribe() 方法
│   │   ├── chat.cpython-*.so   # Qwen3_ASR 专属动态链接库
│   │   └── ...
│   ├── config/                   # tokenizer / processor 配置
│   ├── qwen3-asr.bmodel        # 模型权重
│   └── run.sh
│
├── server/
│   ├── config.toml               # 本地与在线模型配置
│   ├── router.py                 # 路由服务器
│   ├── qwen3_5_server.py         # Qwen3.5-VL 模型服务器
│   ├── qwen3_asr_server.py       # Qwen3-ASR 模型服务器
│   ├── manage.py                 # 启停管理脚本
│   ├── test_all.py               # 全链路测试脚本
│   └── README.md                  # 本文档
│
├── openai.md                     # OpenAI 接口参考文档
└── server/bird.jpg, test.mp3     # 测试资源
```

## 前置依赖

```bash
# Python 依赖（模型服务器所在机器需要）
pip install fastapi uvicorn httpx pyyaml

# 模型运行依赖（每台模型机器按各自 README 安装）
# Qwen3_5:  pip install torchvision transformers qwen_vl_utils
# Qwen3_ASR: pip install torch==2.4.1 transformers==4.57.6 qwen_asr librosa
```

## 快速开始

### 1. 启动所有服务

```bash
cd server
python manage.py start
```

该命令会：
1. 读取 `config.toml`；本地服务按需启动进程，在线模型由 Router 直接代理
2. 启动路由服务器
3. 将 PID 保存到 `.pids/pids.json`
4. 输出各服务的日志路径和状态

### 2. 查看服务状态

```bash
python manage.py status
```

输出示例：
```
Server Status:
------------------------------------------------------------
  qwen3.5         (qwen3-vl        ) [chat ] ONLINE
    URL: http://127.0.0.1:8001
  qwen3-asr       (qwen3-asr       ) [audio] ONLINE
    URL: http://127.0.0.1:8002
------------------------------------------------------------
  Router                            ONLINE
    URL: http://0.0.0.0:8000/v1
------------------------------------------------------------
```

### 3. 停止所有服务

```bash
python manage.py stop
```

该命令会：
1. 向每个模型服务器发送 `POST /shutdown`
2. 服务端收到信号后调用 `model.deinit()`（与 demo 退出流程一致），然后 `sys.exit(0)`
3. 等待进程结束后清理 PID 文件

### 4. 重启所有服务

```bash
python manage.py restart
```

## 配置文件说明 (`config.toml`)

```toml
[router]
host = "0.0.0.0"
port = 8000

[[servers]]
name = "qwen3-5-history"
display_name = "qwen3.5-4b-history"
type = "chat"
backend = "local"
host = "127.0.0.1"
port = 8007
model_path = "../qwen3.5history/model.bmodel"
config_path = "../qwen3.5history/config"
module_path = "../qwen3.5history/demo/python_demo"
server_script = "qwen3_5_history_server.py"
startup = false

[[servers]]
name = "deepseek-v4-flash"
display_name = "deepseek-v4-flash"
type = "chat"
backend = "openai"
provider = "deepseek"
base_url = "https://api.deepseek.com"
upstream_model = "deepseek-v4-flash"
api_key_env = "DEEPSEEK_API_KEY"
thinking = "enabled"
reasoning_effort = "medium"
```

### 添加新模型

添加新的 `[[servers]]` 条目；`backend="local"` 还需编写对应的
`xxx_server.py`。关键要求：
- 服务器必须提供 `/health` 和 `/shutdown` 端点
- Chat 类型必须提供 `POST /v1/chat/completions`
- Audio 类型必须提供 `POST /v1/audio/transcriptions`
- 关停时必须调用 `model.deinit()` 后 `sys.exit(0)`

## API 接口文档

### Router 端点 (默认 8000 端口)

#### `GET /frontend`
只读 Agent 内部对话面板。每个顶层用户输入独立成页，实时展示模型输出、
工具调用、工具返回和最终输出；页面不提供录音、图片或文字输入功能。

#### `GET /v1/agent/traces`
返回最近的 Agent 执行轮次快照，供面板刷新后恢复显示。

#### `POST /v1/agent/events`
Agent 进程使用的内部事件上报接口。事件按 `trace_id` 聚合，Router 仅保留
最近 100 个轮次的有界数据。

#### `WS /v1/agent/events/ws`
面板使用的只读实时事件流。连接后先发送快照，随后推送新事件。

#### `GET /health`
健康检查。

#### `GET /v1/models`
返回所有配置的服务器及其在线状态。

响应：
```json
{
  "object": "list",
  "data": [
    {"id": "qwen3.5", "object": "model", "owned_by": "local", "status": "online", "type": "chat"},
    {"id": "qwen3-asr", "object": "model", "owned_by": "local", "status": "online", "type": "audio"}
  ]
}
```

非流式响应会返回本次推理的精确 token 统计：

```json
{
  "choices": [{"message": {"role": "assistant", "content": "你好"}}],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 12,
    "total_tokens": 140
  }
}
```

其中 `prompt_tokens` 是完整逻辑输入的 token 数；即使命中 prefix cache，也不是只计算本次重新 prefill 的 token。Agent 固定发送 `"stream": false`，并将最近一次结果保存到会话的 `last_usage`。

#### `POST /v1/chat/completions`
对话接口（路由转发到 chat 类型服务器）。

请求体与 OpenAI 标准一致：

```json
{
  "model": "qwen3.5",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}
```

支持多模态（图片）：
```json
{
  "model": "qwen3.5",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64_data>"}}
      ]
    }
  ]
}
```

服务器仍兼容流式响应（SSE）：设置 `"stream": true`。Agent 主流程不使用该模式。

会话续接（可选请求头）：
- `x-session-id: <string>` — 同一 session_id 的请求会复用 KV cache（不清理历史）
- `x-clear-history: true` — 强制清理历史

#### `POST /v1/audio/transcriptions`
语音转文字（路由转发到 audio 类型服务器）。

请求格式：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 音频文件（mp3, wav, m4a 等） |
| `model` | string | 否 | 模型名（默认 qwen3-asr） |
| `language` | string | 否 | 强制指定语言代码 |

响应：
```json
{"text": "今天天气不错"}
```

#### `POST /v1/shutdown/{server_name}`
远程关闭指定模型服务器（内部管理用，一般通过 `manage.py stop` 调用）。

### 模型服务器端点 (8001/8002)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/shutdown` | POST | 调用 deinit() 后退出 |
| `/v1/chat/completions` | POST | 仅 chat 类型服务器 |
| `/v1/audio/transcriptions` | POST | 仅 audio 类型服务器 |

## 会话管理

### 无状态模式（默认）

不传 `x-session-id` 时，每次请求都会清理历史（`clear_history`），等效于全新对话。

### 有状态模式

通过 `x-session-id` 头传递会话标识：

```bash
# 第一次请求（自动创建会话）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: my-session-123" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"请记住数字42"}]}'

# 第二次请求（同一 session，历史续接）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: my-session-123" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"我刚才让你记住什么数字？"}]}'

# 强制清理历史
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: my-session-123" \
  -H "x-clear-history: true" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"新对话"}]}'
```

## 关停机制

关停流程与 demo 中 Python 退出时 `__del__` → `deinit()` 一致：

1. `manage.py stop` 发送 `POST /shutdown` 到每个模型服务器
2. 服务端收到信号后执行 `pipeline.model.deinit()`（显存释放）
3. 然后调用 `sys.exit(0)` 优雅退出进程

**重要**：不要手动 kill -9 进程，这会跳过 `deinit()`，可能导致显存泄漏。

## 客户端使用示例

### Python (openai 库)

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-string-is-ok",
    base_url="http://localhost:8000/v1"
)

# 文本对话
response = client.chat.completions.create(
    model="qwen3.5",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)

# 流式对话
stream = client.chat.completions.create(
    model="qwen3.5",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 图片识别
import base64
with open("bird.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen3.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
)

# 语音转文字
with open("test.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr",
        file=f,
    )
    print(transcript.text)
```

### curl

```bash
# 文本对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"你好"}]}'

# 流式对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 语音转文字
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test.mp3" -F "model=qwen3-asr"
```

## pipeline.py 修改说明

### Qwen3_5/pipeline.py

新增 `generate()` 方法（类成员），复用 `chat()` 的全部推理逻辑：

```python
def generate(self, messages, media_type, clear_history_flag=True):
    """generator：每次 yield 一个词（word-level delta）"""
    # clear_history_flag=True 时清理 KV cache
    # 严格复用: forward_embed → vit_process → forward_prefill → forward_next 循环
    ...
    yield word  # 每个 word 就是一个增量文本片段
```

原始 `chat()` 方法完全保留不变。

### Qwen3_ASR/pipeline.py

新增 `transcribe()` 方法（类成员），复用 `asr()` 的全部推理逻辑：

```python
def transcribe(self, context_str, audio_path, language=None, clear_history_flag=True):
    """generator：yield 转录文本（跳过 <asr_text> 之前的内容）"""
    # 遇到 <asr_text> token 时开始输出
    # 每个 word yield 一次
    ...
    yield word  # 仅 yield 转录内容部分
```

原始 `asr()` 方法完全保留不变。

## 日志查看

各服务日志存储在 `server/.pids/` 目录：

```bash
# 查看模型服务器日志
cat .pids/qwen3-vl.log
cat .pids/qwen3-asr.log
cat .pids/router.log

# 实时查看
tail -f .pids/qwen3-vl.log
```

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 模型服务器启动失败 | .so 文件路径不正确 | 检查 `module_path` 配置 |
| 502 Bad Gateway | 模型服务器未启动 | `python manage.py status` 检查状态 |
| 显存不足 | TPU 内存被旧进程占用 | `python manage.py stop` 后重新 start |
| 推理超时 | 请求过长 | 减少输入文本长度 |
| Base64 图片解析失败 | 格式不正确 | 确保 `data:image/jpeg;base64,` 前缀 |
| ASR 无输出 | 音频格式不支持 | 使用 ffmpeg 转换为 wav/mp3 |
