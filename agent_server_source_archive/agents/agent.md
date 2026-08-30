# Voice Agent 架构

当前线上 Agent 只有一个统一的 `Voice` 运行时。系统不做模式分类，也不支持模型控制的模式切换。导航、医疗咨询、实时查询、混合工具调用和普通问答都由 Voice 提示词与工具循环统一处理。

## 数据流

```text
用户语音/文本
    │
    ▼
VoiceMode.loop()
    │
    ├─ 构建 Voice system prompt
    ├─ 追加用户消息与运行时执行策略
    ├─ 调用模型
    ├─ 解析 tool_call / skill_call
    ├─ 执行工具并追加结果
    └─ 继续循环，直到自然语言回答或 speak 成功
```

## 提示词与缓存槽位

Agent 默认使用 `suha.v3` 五槽协议（服务端继续兼容 `suha.v1/v2`），后端固定按以下顺序序列化，不能依赖 JSON 键顺序：

1. `system`：完全固定的身份、规则、工具说明和可用 Skill 摘要。
2. `conversation`：进入本轮前已经完成的自然语言用户/助手轮次，最多三轮；它不属于工具历史。
3. `user`：仅包含本轮用户输入、传感器/运行时约束，并在规划后追加冻结 plan。
4. `history`：先前任务中已经真实执行成功的工具或 Skill，以及当前任务按 index 追加的 `<execution_state_event>`。当前成功调用仍完整写入 ledger，但本任务内通过状态事件提供给模型，下一任务才进入历史基线；以最后一条状态事件为当前权威投影。
5. `attempt`：最近一次成功提交后的工具失败、策略拒绝或命令语法错误。

快照仍以精确 token 前缀为唯一恢复依据。检查点位于每个槽内容末尾；若新值是在旧值末尾追加，允许恢复旧检查点并只 prefill 新增 token。因而 plan 追加可部分命中 `U`，执行状态追加可部分命中 `H`，新用户进入且 conversation 继续增长时可部分命中 `C`。若旧 token 不是新 prompt 的逐 token 前缀则自动退回更浅检查点，绝不复用不相容 KV。

成功工作追加到 `history` 时会清空 `attempt`。新用户任务保留已归档的 `history`，但会清空上一任务的 `attempt`。查询类工具返回空结果作为可恢复失败写入 `attempt`。`history_visible_entries` 只限制送给模型的尾部条数，不删除会话内的完整归档。

每个新任务第一轮只允许调用 `plan`。规划提示作为临时 `<system kind="planning">`
放在 history 块，不改变固定 system；规范化 plan 从第二轮开始注入 user，且不会被当作
成功工具事实写入 history。

## Harness 状态机

Harness 将三个对象严格分离：`TaskPlan` 是冻结的语义计划，`ExecutionState`
是仅由 Harness 修改的真实进度，模型每轮只能提出下一动作。内部状态可以原地更新，
但送给模型的状态只会追加新的 `<execution_state_event>`，旧事件不会修改或删除。

步骤生命周期为 `BLOCKED / READY / ACTIVE / COMPLETED / SKIPPED / FAILED`；
顶层执行状态包括 `RUNNING / RECOVERING / BLOCKED / GOAL_SATISFIED / FINISHED`
及失败、取消、预算耗尽终态。条件采用 `TRUE / FALSE / UNKNOWN` 三值逻辑，缺失或
已失效的事实不能让条件通过。

工具的 `x-harness` 元数据声明 READ/WRITE、产生和失效的事实以及重试预算。WRITE
成功后才增加 `world_epoch` 并使指定事实失效；只有新的工具 observation 可以重新建立
有效事实。相同工具和规范化参数在同一 epoch 内不会再次执行，而 WRITE 后的验证查询
会因 epoch 已变化而被允许。满足 `success_conditions` 后，ActionValidator 会禁止任何
新工具调用。

默认预算为 12 个 Agent 动作、10 次工具调用、2 次 replan 预留额度和每步骤最多
2 次尝试。当前 MVP 执行协议为 `PLAN once / ACT / ... / ACT(FINISH)`，每轮强制一个动作；
计划修订和并行工具仍保留为后续阶段能力。

执行阶段禁止裸业务工具调用和裸最终文本。模型必须输出：

```xml
<tool>{"tool_call":"act","param":{"step_id":"s2","action_type":"CALL_TOOL","tool":"navigate","arguments":{"action":"start","target":"药房"}}}</tool>
<tool>{"tool_call":"act","param":{"step_id":"","action_type":"FINISH","response":"最终答复"}}</tool>
```

Harness 将 `act` 解包成真实工具调用；`step_id` 必须与最新状态投影一致。最新投影包含
`goal/current_step/completed_steps/known_facts/remaining_steps/success_condition`，例如：

```json
{"goal":"ensure_service_running","current_step":2,"current_step_id":"s2","completed_steps":[{"id":1,"step_id":"s1","result":"service_status=stopped"}],"known_facts":{"service_status":"stopped"},"remaining_steps":["start_service","verify_service_status"],"success_condition":"service_status == \"running\""}
```

Agent 可在启动时选择 `qwen3.5-4b-history`、无缓存本地模型或 Router 配置的
`deepseek-v4-flash`。本地聊天模型属于同一个路由互斥组，不能同时加载到同一设备。

Agent 的模型请求统一为非流式调用（`stream: false`）。后端响应的 `usage.prompt_tokens`、`usage.completion_tokens` 和 `usage.total_tokens` 会保存到 `Session.last_usage`；Headless 会话状态也会返回该字段。

## 可用命令

```xml
<tool>{"tool_call":"plan","param":{"goal":"目标","steps":[...]}}</tool>
<tool>{"tool_call":"act","param":{"step_id":"s1","action_type":"CALL_TOOL","tool":"工具名","arguments":{}}}</tool>
```

系统不接受 `mode_switch`。

## 当前工具

- `plan`：每个新任务第一轮生成执行计划；计划随后注入 user 槽，不作为成功工具历史。
- `speak`：语音播报。
- `get_system_stats`：获取系统状态。
- `get_time`：获取时间和日期。
- `medical_consult`：查询本地医疗知识库并进行风险检查。
- `navigate`：启动或停止医院内导航。

## 上下文管理

`Session.history` 保存完整诊断轨迹；发送给模型的内容只来自 `Session.prompt_slots`。完整诊断历史不会被临时重排后重新注入模型。
