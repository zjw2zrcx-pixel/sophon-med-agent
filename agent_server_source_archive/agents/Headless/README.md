# Headless Voice Agent

Headless 接口为统一的 Voice Agent 提供多会话管理。所有会话使用同一份 Voice 提示词，但分别保存对话历史。

## 浏览器调试台

启动 Voice Agent 后访问 `http://<host>:8766/`（`/debug` 也是同一页面）。调试台提供：

- 真正经过 `HeadlessManager` 的 Agent 文本会话，可查看历史、工具结果与本次指标；
- Router 的模型状态、直接 chat completion、文件/麦克风 ASR 和 TTS；
- Router 已上报的最近 Agent 执行轨迹。

调试台的“远程语音 Agent 全链路”会作为一个 `remote-audio.v1` 浏览器设备接入
`/ws`，显示热词命中、Qwen ASR 文本、Agent 最终输出、TTS 播放文本与协议日志。浏览器
录音需在 HTTPS 或 `localhost` 安全上下文中授权麦克风；以 HTTP 从局域网 IP 打开时，可
先使用“回放测试音频”上传包含热词的 WAV/MP3，走相同的 KWS → ASR → Agent → TTS 链路。

页面通过 Voice Agent 的只读/调试桥接端点访问 Router，因此即使 Router 仅绑定在
`127.0.0.1:8000`，也可从另一台机器的浏览器使用。它不会提供模型加载、卸载或关停操作；
这些操作仍请通过 `server/manage.py` 完成。Agent 调试接口为：

- `GET/POST /v1/agent/sessions`、`DELETE /v1/agent/sessions/{id}`
- `GET /v1/agent/sessions/{id}/history`
- `POST /v1/agent/prompt`

## CLI

```bash
# 创建会话
python -m agents.Headless.cli session create --id my-session --tag demo

# 提交文本
python -m agents.Headless.cli prompt my-session --text "带我去药房"

# 查看历史
python -m agents.Headless.cli history my-session --format md
python -m agents.Headless.cli config set history_visible_entries 8

# 启动时选择聊天模型并指定轨迹目录
python -m agents.Headless.cli --model deepseek-v4-flash \
  --trajectory-dir /data/structure/trajectories repl

# 查看状态
python -m agents.Headless.cli status

# 删除会话
python -m agents.Headless.cli session delete my-session
```

`session create` 支持：

- `--id`：指定会话 ID。
- `--tag`：添加标签，可重复使用。

系统不提供模式选择或模式切换命令。

未传 `--model` 且 stdin 为交互终端时会显示 Router 返回的聊天模型菜单；
非交互运行默认使用并按需加载 `qwen3.5-4b-history`。每个外部轮次默认保存一份
脱敏 JSON；使用 `--no-trajectory` 可关闭。

## Python API

```python
import asyncio

from agents.agent import Agent
from agents.Headless.manager import HeadlessManager


async def main():
    agent = Agent()
    agent.initialize()
    manager = HeadlessManager(agent)

    session_id = manager.create_session(tags=["demo"])
    result = await manager.submit(session_id, "现在几点？")
    print(result.text)

    await agent.shutdown()


asyncio.run(main())
```

## 会话隔离

`HeadlessManager` 为每个客户端保存独立的 `Session`。提交请求时，它临时把目标 Session 注入共享的 `VoiceMode`，调用 `Agent.handle_input_in_session()`，完成后恢复原 Session。共享 VoiceMode 的访问由异步锁串行化，避免不同客户端互相覆盖上下文。
