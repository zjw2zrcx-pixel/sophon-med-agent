# BM1684 本地多模态 Agent 系统

这是一个运行在 Sophgo BM1684（或兼容 BM1684 运行时环境）上的本地 AI Agent 项目。项目把本地模型服务、Agent 编排、语音交互、工具调用、技能加载和医疗知识检索组合成一条完整链路：

```text
浏览器麦克风/文本/图片
        │
        ▼
Voice Agent（8766）
        │  ASR + Agent + TTS
        ▼
Router（8000，OpenAI 兼容）
        ├── Qwen3.5-VL（8001，文本/图片对话）
        ├── sherpa KWS（8004，原始 PCM 唤醒词检测）
        ├── VITS Melo TTS（8005，Agent 播报主后端）
        ├── Dolphin Streaming ASR（8002，保留作兼容/诊断）
        └── Qwen3-ASR（8003，最终语音识别）
```

项目主体在两个目录：

- `agents/`：Agent 核心。负责意图识别、模式切换、上下文、工具/技能执行、医疗问答和语音会话。
- `server/`：模型服务编排层。负责启动、监控和代理本地 Qwen3.5-VL、Dolphin ASR、Qwen3-ASR，并提供 OpenAI 风格接口。

其余目录主要是模型文件、运行时 Demo、语音资源、医疗索引和历史/实验代码。

## 功能概览

- 本地 Qwen3.5-VL 多模态对话：支持文本和 Base64 图片输入。
- OpenAI 兼容 HTTP API：支持 `/v1/chat/completions`、`/v1/audio/transcriptions` 和流式接口。
- Agent 模式路由：按规则优先、模型兜底，将请求分发到 `Act`、`QA`、`Med_QA`、`Wait` 等模式。
- 容错命令解析：兼容小模型常见的大小写、引号、括号和拼写错误。
- MCP 工具调用：语音播报、系统信息、时间、医疗咨询、导航等工具由 Agent 统一注册和执行。
- Skill 技能：从 `SKILL.md` 加载可组合的流程型技能，当前包含房间探索和目标定位示例。
- 医疗本地检索：SQLite 医疗索引配合实体别名、事实和文档检索，线上只向模型暴露高层的 `medical_consult`。
- 无头会话：可以通过 CLI 创建多个相互隔离的会话，提交文本/图片、查看历史和导出结果。
- 浏览器语音交互：唤醒词默认为“**小麦**”。原始 PCM 先由 sherpa KWS 判定，命中后 Qwen3-ASR 做最终识别，再交给 Agent 处理；Dolphin 不在唤醒词关键路径中。
- Agent 调试面板：`http://localhost:8000/frontend` 展示最近的 Agent 轮次、模型输出、工具调用和工具返回。

## 运行环境

项目当前启动脚本默认使用：

```text
/data/env310/bin/python
```

也可以通过 `PYTHON_BIN` 环境变量替换。模型服务依赖 BM1684 运行时及各模型 Demo 的 Python 扩展；没有 TPU 或对应运行库的机器不能直接加载这些 `.bmodel` 文件。

至少需要以下 Python 包（实际模型 Demo 可能还需要各自 README 中列出的依赖）：

```bash
python -m pip install fastapi uvicorn httpx pyyaml websockets
```

语音浏览器链路还需要现代浏览器、麦克风权限和 `tmux`（使用一键启动脚本时）。正式 Agent 的 TTS 只使用 Router 中已加载的 VITS Melo 服务；服务异常会明确返回错误，不会降级为本地语音引擎。

## 快速启动

### 一键启动模型服务和语音 Agent

```bash
cd /data/structure
PYTHON_BIN=/data/env310/bin/python ./start_all.sh
```

脚本会创建两个 tmux 会话：

- `server`：Router 及其管理的三个模型服务。
- `agent`：`agents.Headless.voice_agent` 语音 Agent。

启动后可访问：

```text
调试面板： http://127.0.0.1:8000/frontend
Router：   http://127.0.0.1:8000
Agent：    http://127.0.0.1:8766
WebSocket：ws://127.0.0.1:8766/ws
```

查看日志：

```bash
tmux attach -t server
tmux attach -t agent
```

从 tmux 返回而不停止进程：按 `Ctrl+B`，再按 `D`。

停止全部服务：

```bash
./stop_all.sh
```

### 分步启动

只启动 Router 和模型服务：

```bash
cd server
/data/env310/bin/python manage.py start
```

查看服务状态：

```bash
/data/env310/bin/python manage.py status
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/v1/models
```

模型加载需要时间。只有对应服务的 `/health` 返回 `{"status":"ready"}` 后，Router 才会转发请求。

模型服务就绪后，再启动 Agent：

```bash
./start_agent_direct.sh       # 当前终端运行
# 或
./start_agent.sh              # 放入 tmux 会话运行
```

停止模型服务：

```bash
cd server
/data/env310/bin/python manage.py stop
```

## Agent 的使用方式

### Headless CLI

CLI 不依赖浏览器，可用于调试 Agent、批处理和脚本集成。模型服务必须先在 `http://127.0.0.1:8000` 就绪。

```bash
# 创建 QA 会话
python -m agents.Headless.cli session create --mode QA

# 使用返回的会话 ID 提问
python -m agents.Headless.cli prompt <SESSION_ID> --text "什么是 Python？"

# 机器可读输出
python -m agents.Headless.cli prompt <SESSION_ID> --text "介绍一下这张图" \
  --image <BASE64_IMAGE> --format json

# 查看会话和历史
python -m agents.Headless.cli session list
python -m agents.Headless.cli history <SESSION_ID> --format md --tail 10

# 交互式 REPL
python -m agents.Headless.cli
```

支持的会话模式：

| 模式 | 用途 |
| --- | --- |
| `Wait` | 等待或低交互状态 |
| `QA` | 普通知识问答 |
| `Med_QA` | 医疗相关问答，带安全提示和免责声明 |
| `Act` | 行动/导航/操作决策 |
| `Voice` | 语音场景使用的对话模式 |

完整 CLI 命令说明见 [agents/Headless/README.md](agents/Headless/README.md)。

### 浏览器语音

语音 Agent 的默认工作流如下：

1. 浏览器通过 WebSocket 发送 16 kHz 单声道 PCM 音频。
2. Agent 将音频转发给 Router 的 `/v1/audio/transcriptions/stream`。
3. sherpa KWS 直接在原始 PCM 上返回唤醒词命中结果，不依赖 ASR 增量文本。
4. 命中“**小麦**”后开始捕获用户指令，并播放确认音。
5. 静音或超时后，将音频交给 Qwen3-ASR 做最终识别。
6. 识别文本进入 Agent；Agent 请求服务端 VITS Melo 生成 TTS 音频。VITS 不可用时该轮播报明确失败，便于排障。

默认限制包括：单次语音捕获最多约 10 秒，Qwen3-ASR 最终识别保留安全的 5 秒音频窗口。若需修改唤醒词，浏览器可以通过协议消息更新 `hotwords`；默认连接建立后为 `小麦`。

## Agent 内部工作流

```text
输入
  ▼
ModeRoute：规则匹配（高置信度直接返回）/ LLM 分类兜底
  ▼
高置信度任务：Harness 附加确定性 TaskPlan；其他任务由模型直接选择业务工具
  ▼
Harness Controller：维护 ExecutionState、FactStore、world_epoch 和预算
  ▼
API：必要时将最新 append-only ExecutionProjection 交给模型请求一个业务动作
  ▼
CallRoute + ActionValidator：解析并检查步骤、重复、重试和成功状态
  ▼
MCP/Skill 执行 → observation → Harness 更新权威状态
  ▼
成功条件满足后输出最终文本
```

模型面向小模型设计，每轮直接调用一个 XML 包裹的严格 JSON 业务工具，例如：

```xml
<tool>{"tool_call":"navigate","param":{"action":"start","target":"药房"}}</tool>
<tool>{"tool_call":"speak","param":{"text":"已经处理完成"}}</tool>
```

TaskPlan 与 ExecutionState 相互独立：计划不保存 `completed`，步骤进度只能由经过
验证的工具 observation 推进。工具 metadata 声明 READ/WRITE、产生事实、失效事实和
重试上限；WRITE 会增加 `world_epoch`，因此同参数查询只有在世界状态发生变化后才能
作为验证再次执行。模型看到的是追加的 `<execution_state_event>` 快照，不能修改内部
状态。`CallRoute` 仍兼容旧式宽松语法，但新轨迹统一记录 XML 严格 JSON。
`plan`/`act` 不再暴露给模型；业务工具在执行前仍经过 Harness 的当前步骤、重复调用、
重试预算、事实和安全策略校验，实际状态只由 observation 更新。医疗高置信度链路由
Harness 直接执行 `medical_consult → query/speak`，安全播报只取受控工具事实。

## Server API

Router 默认监听 `8000`。

### 健康与管理

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | Router 健康检查 |
| `GET` | `/status` | Router 和各模型服务状态 |
| `GET` | `/v1/models` | 列出模型及 online/loading/offline 状态 |
| `POST` | `/v1/load/{name}` | 动态加载配置中的服务 |
| `POST` | `/v1/unload/{name}` | 动态卸载服务 |
| `POST` | `/shutdown` | 停止 Router 管理的所有服务 |

当前配置的模型名和端口：

| 对外模型名 | 类型 | 端口 | 用途 |
| --- | --- | ---: | --- |
| `qwen3.5` | `chat` | 8001 | Qwen3.5-VL 对话/图像理解 |
| `dolphin-asr` | `audio` | 8002 | 流式 ASR 和普通转写 |
| `qwen3-asr` | `audio` | 8003 | 最终语音转写 |

### 文本或多模态对话

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
    "stream": false
  }'
```

图片使用 OpenAI 风格的 `content` 数组和 `data:` URL：

```json
{
  "model": "qwen3.5",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
    ]
  }],
  "stream": true
}
```

### 音频转写

普通转写使用 `multipart/form-data`：

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@server/test.mp3 \
  -F model=qwen3-asr
```

浏览器语音使用：

```text
POST /v1/audio/transcriptions/stream
```

该接口由 `agents/Headless/voice_agent.py` 按 JSON 音频块协议调用，普通 API 使用者通常不需要直接调用。

### Agent 观测接口

```text
GET  /frontend
GET  /v1/agent/traces
POST /v1/agent/events
WS   /v1/agent/events/ws
```

Router 最多保留最近 100 个 Agent 轮次；事件上报失败不会阻断 Agent 主流程。

## 配置

模型服务配置位于 [server/config.toml](server/config.toml)。每个 `[[servers]]` 条目至少包含：

- `name` / `display_name`：内部名称和 API 中的模型名。
- `type`：`chat` 或 `audio`。
- `host` / `port`：子服务监听地址。
- `model_path`：`.bmodel` 路径，相对于 `server/` 解析。
- `config_path`：tokenizer/processor 配置目录。
- `module_path`：Python Demo 和动态库目录。
- `server_script`：对应的 FastAPI 模型服务脚本。
- `devid`：BM1684 设备编号。
- `startup`：Router 启动时是否自动加载。

Agent 默认配置写在 `agents/agent.py` 的 `AgentConfig` 中：

```python
AgentConfig(
    server_url="http://127.0.0.1:8000",
    model_name="qwen3.5-4b-history",
    max_context_tokens=4096,
    camera_backend="auto",
)
```

聊天模型既可使用本地 `qwen3.5-4b-history`，也可使用配置中的
`deepseek-v4-flash`。在线模型只从 Router 进程环境读取
`DEEPSEEK_API_KEY`，配置文件不保存密钥。Voice Agent 和 Headless CLI
支持 `--model`，未指定且处于交互终端时会显示模型菜单；非交互默认本地模型。

每个外部对话轮次默认原子写入 `trajectories/<session>/<turn>_<task>.json`。
可用 `--trajectory-dir` 修改目录或用 `--no-trajectory` 关闭；轨迹不包含图片、
Authorization、API Key 或 DeepSeek `reasoning_content`。

Agent 使用 `suha.v3` 五槽协议。固定提示、已完成的自然语言对话、当前任务、成功工具历史和当前失败分别
置于 `<system>`、`<conversation>`、`<user>`、`<history>`、`<attempt>`；服务端继续兼容 `suha.v1/v2`。工具调用采用
`<tool>{"tool_call":"name","param":{...}}</tool>`。模型每轮直接选择一个业务工具；
高置信度任务的确定性计划由 Harness 内部附加，不消耗单独的模型规划轮。后续成功和失败
只更新 history/attempt 与执行投影，以保持本地前缀缓存命中。

需要使用文件图片或 V4L2 摄像头时，可设置 `camera_backend`、`camera_device` 或 `camera_image_path`。无摄像头时 Agent 会自动使用 dummy backend，文本功能不受影响。

## 医疗索引

仓库中已包含 `med_database/med_search.sqlite`。如需重新构建或切换索引：

```bash
python -m agents.Medical.build_index \
  --output /data/structure/med_database/med_search.sqlite
```

运行时可以使用 `MEDICAL_INDEX_PATH` 指定索引文件。Agent 启动后默认在后台预热索引，
可用 `MEDICAL_INDEX_PREWARM=0` 关闭。医疗输出仅供信息参考，不能替代医生诊断、处方或急救；
线上工具故意只注册 `medical_consult`，旧的 `MedQueryTool` 不会暴露给模型。

## 目录说明

```text
.
├── agents/                  # Agent 主体
│   ├── agent.py             # 总编排入口
│   ├── API/                 # LLM API 封装和 Session
│   ├── ModeRoute/           # 意图分类和模式路由
│   ├── Modes/               # QA/医疗/行动/等待/语音模式
│   ├── CallRoute/           # 模型命令解析、安全和分发
│   ├── MCP/                 # 工具基类、注册器和工具实现
│   ├── Skill/               # SKILL.md 加载和技能管理
│   ├── Medical/             # 医疗 SQLite 索引与检索
│   ├── Headless/            # CLI、会话管理、正式语音 Agent
│   ├── Camera/              # 摄像头抽象和图片采集
│   └── Voice/               # 旧的轻量语音状态机
├── server/                  # Router 和三个模型服务
├── Qwen3_5/                 # Qwen3.5-VL bmodel、配置和 Demo
├── Qwen3_ASR/               # Qwen3-ASR bmodel、配置和 Demo
├── Dolphin_CN_Streaming/    # Dolphin 流式 ASR 及浏览器前端
├── med_database/            # 医疗数据和 SQLite 索引
├── sherpa-onnx-*/            # 唤醒词模型资源
├── vits-melo-tts-zh_en/      # TTS 运行时和模型资源
├── tests/                   # Agent、医疗、导航和语音测试
├── start_all.sh             # 一键启动
├── start_agent*.sh          # 单独启动语音 Agent
└── stop_all.sh              # 停止全部服务
```

## 已接入与暂未接入的代码

以下状态按当前源码中的 import、注册表和启动脚本判断，不代表这些文件没有价值。

### 当前主链路已使用

- `agents/agent.py`、`agents/API/`、`agents/Modes/`、`agents/ModeRoute/`、`agents/CallRoute/`。
- `agents/MCP/tools/__init__.py` 中 `ALL_TOOLS` 注册的工具：`SpeakTool`、系统工具、`MedicalConsultTool`、`NavigateTool`。
- `agents/Skill/` 和 `agents/Skill/examples/` 中能通过校验的技能。
- `agents/Headless/manager.py`、`agents/Headless/cli.py`、`agents/Headless/voice_agent.py`。
- `server/router.py`、`server/qwen3_5_server.py`、`server/dolphin_asr_server.py`、`server/qwen3_asr_server.py`。
- `Dolphin_CN_Streaming/frontend.html`：语音 Agent 使用的浏览器前端资源。

### 当前默认链路未使用或仅作兼容/实验保留

- `agents/Voice/voice_server.py`、`agents/Voice/hotword.py`：旧的轻量语音 WebSocket/文本分段状态机；正式启动脚本使用 `agents.Headless.voice_agent`。
- `agents/CallRoute/framer.py` 中的 `StreamFramer`：已导出但当前 Agent 主要使用完整响应路由；属于未来流式命令场景的预留组件。
- `agents/MCP/tools/medquery.py` 的 `MedQueryTool`：旧医疗查询适配器，仅供回滚和诊断，不在 `ALL_TOOLS` 注册，不应直接作为线上模型工具。
- `agents/Modes/qa.md`、各模型目录下的 `run.sh`、原始 Demo 和 `verify_bmodel.py`：说明、单模型实验或模型验证用途，不是项目总入口。
- `sherpa-onnx-kws-zipformer-zh-en-3M/`：唤醒词模型资源；当前 `Headless` 链路的热词判断由 ASR 流式结果完成，不能据此认为该目录会自动参与启动。
- `server/sniff.py`、`server/sniff_logs/` 等诊断资源：用于抓包、调试或测试，不属于生产请求链路。

修改工具、模式或模型时，应优先检查注册表和启动脚本，而不是只新增文件。例如新增 MCP 工具后，需要将其加入 `agents/MCP/tools/__init__.py` 的 `ALL_TOOLS`，否则模型不会看到也不会调用它。

## 测试与故障排查

### sherpa KWS 与 VITS 验收

`sherpa-kws` 和 `vits-melo-tts` 已作为 Router 管理的独立服务（8004/8005），并在
`Voice Agent` 中分别处于唤醒词和播报的主路径。二者的每次调用都会写出
`PERF: sherpa_kws=...`、`PERF: vits_tts=...` / `PERF: vits_total_ms=...`；浏览器
播放事件还会记录端到端播放延迟。可据此分离模型推理、Agent、WebSocket 与客户端播放延迟。

当前提交的 sherpa 目录**只有 bmodel 和 test.wav**。Transducer KWS 还必须有与该
263 类输出匹配的 `tokens.txt`（或 `.k2sym`）和基于 SAIL 的 39 状态流式 runner；缺少
它们时服务会显式处于 `error`，不会悄悄退回 Dolphin 文本匹配。将这两个原始 sherpa
资源放到 `sherpa-onnx-kws-zipformer-zh-en-3M/` 后，可用下列方式验证：

```bash
cd /data/structure
/data/env310/bin/python sherpa-onnx-kws-zipformer-zh-en-3M/verify_bmodel.py
curl http://127.0.0.1:8000/status
# 完整 runner 安装后，以 test.wav 分块送入 /v1/audio/keywords，检查 hotword_hits 与首命中时间。
```

VITS 的模型服务会严格接收 Melo 前端输出的 `phonemes`、`tones`，避免用未经验证的
中英 G2P 产生错误声音。仓库当前不含该前端依赖；若服务返回前端不可用，Agent 会明确报告
VITS 错误。上板可先运行 CPU 阶段冒烟测试：

```bash
cd /data/structure
/data/env310/bin/python vits-melo-tts-zh_en/test_cpu_stages.py
```

补齐 Melo zh/en G2P 前端后，向 `POST /v1/audio/speech` 提交不超过 50 个 token 的
`phonemes`/`tones`，响应中的 `audio` 是 44.1 kHz WAV Base64，保存后即可试听中英混读。

运行不依赖真实模型输出的单元测试：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

服务启动后运行模型服务全链路测试：

```bash
cd server
python test_all.py
```

常见问题：

1. `Router` 在线但模型是 `loading/offline`：查看 `tmux attach -t server`，确认 `.bmodel`、配置目录、动态库目录和 `devid` 正确。
2. Agent 报 `LLM server unreachable`：先检查 `curl http://127.0.0.1:8000/health` 和 `curl http://127.0.0.1:8000/v1/models`。
3. 语音无响应：确认浏览器连接 `ws://<host>:8766/ws`、麦克风权限、唤醒词“小麦”以及 Router 的 `sherpa-kws`/`qwen3-asr` 状态。
4. 图片无法识别：确认使用 `qwen3.5`，图片以合法的 `data:image/...;base64,...` 形式放入 `content` 数组。
5. 医疗查询没有结果：确认 `med_database/med_search.sqlite` 存在，或设置了正确的 `MEDICAL_INDEX_PATH`。
6. 停止后仍有进程：先运行 `cd server && python manage.py stop`，再检查 `python manage.py status` 和 tmux 会话。

## 相关文档

- [agents/agent.md](agents/agent.md)：Agent 架构和命令容错设计。
- [agents/Headless/README.md](agents/Headless/README.md)：Headless CLI、会话管理和编程接口。
- [agents/Medical/README.md](agents/Medical/README.md)：医疗索引构建和数据策略。
- [server.md](server.md)：模型服务器和 OpenAI 兼容接口的补充说明。

## 安全与部署提示

这是面向本地设备和研发调试的系统。Router 默认绑定 `0.0.0.0`，并且 CORS 允许任意来源；若暴露到局域网或公网，应在反向代理、防火墙和鉴权层限制访问。医疗回答不构成医疗建议，导航、移动和其它执行型工具在接入真实硬件前必须进行独立的安全审查。
