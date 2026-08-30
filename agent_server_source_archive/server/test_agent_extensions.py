from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from agents.API.session import ConversationTurn, Session  # noqa: E402
from agents.API.trajectory import TrajectoryWriter  # noqa: E402
from agents.CallRoute.router import CallRouter  # noqa: E402
from agents.CallRoute.parser.parser import Command  # noqa: E402
from agents.MCP.manager import MCPManager  # noqa: E402
from agents.MCP.base import Tool, ToolContext, ToolResult  # noqa: E402
from agents.MCP.tools.plan import PlanTool  # noqa: E402
from agents.MCP.tools.voice import QueryTool, SpeakTool  # noqa: E402
from agents.Modes.base import ModeBase  # noqa: E402
from config_loader import load_config  # noqa: E402
import router as server_router  # noqa: E402
from agents import model_selection  # noqa: E402
from agents.API.api import _visible_message_content  # noqa: E402


class AgentExtensionTests(unittest.TestCase):
    def test_query_tool_adds_recording_instruction_and_returns_mock_followup(self):
        questions = []
        tool = QueryTool()
        result = asyncio.run(tool.call(
            {"question": "症状持续多久了？"},
            ToolContext(extra={
                "query_followup_provider": lambda question: (
                    questions.append(question) or "已经三天了"
                )
            }),
        ))
        self.assertTrue(result.success)
        self.assertEqual(questions, ["症状持续多久了？"])
        self.assertEqual(result.transient["followup_text"], "已经三天了")
        self.assertEqual(
            QueryTool.playback_text("症状持续多久了？"),
            "症状持续多久了。在滴的一声后开始回答",
        )

    def test_query_ends_turn_and_exposes_followup_without_ending_session(self):
        class MedicalStub(Tool):
            name = "medical_consult"
            description = "medical"
            param_schema = {"query": "query"}
            modes = ["Test"]
            harness_metadata = {
                "effect": "READ", "produces": ["medical.consultation"],
                "retry": {"max_attempts": 1},
            }
            async def call(self, params, context):
                return ToolResult(success=True, data="need_more_info", facts={
                    "medical.consultation": {"status": "need_more_info"},
                    "dialogue.followup_required": True,
                    "dialogue.followup_questions": ["持续多久了？"],
                })

        class FakeAPI:
            def __init__(self):
                self.outputs = [
                    '<tool>{"tool_call":"plan","param":{"goal":"补充问诊",'
                    '"success_conditions":[{"fact":"dialogue.question",'
                    '"operator":"exists"}],"steps":[{"step_id":"s1",'
                    '"goal":"查询","preferred_tool":"medical_consult"},'
                    '{"step_id":"s2","goal":"追问","preferred_tool":"query",'
                    '"depends_on":["s1"]}]}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"CALL_TOOL","tool":"medical_consult",'
                    '"arguments":{"query":"我头晕"}}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s2",'
                    '"action_type":"CALL_TOOL","tool":"query",'
                    '"arguments":{"question":"持续多久了？"}}}</tool>',
                ]
            async def chat_complete(self, session): return self.outputs.pop(0)
            async def emit_agent_event(self, *args, **kwargs): return None

        class FakeSkills:
            def get_skills_description(self, mode): return ""
            def get_active_skill_prompt(self): return ""
            def deactivate_all(self): return None

        class TestMode(ModeBase):
            name = "Test"
            def get_system_prompt(self): return "fixed"

        mcp = MCPManager()
        query_tool = QueryTool()
        query_tool.modes = ["Test"]
        plan_tool = PlanTool()
        plan_tool.modes = ["Test"]
        mcp.register_many([plan_tool, MedicalStub(), query_tool])
        mode = TestMode(FakeAPI(), mcp, FakeSkills(), CallRouter(mcp=mcp))
        result = asyncio.run(mode.loop(
            "我头晕",
            tool_context_extra={"query_followup_provider": lambda _: "三天了"},
        ))
        self.assertEqual(result.turn_end_reason, "query")
        self.assertFalse(result.session_ended)
        self.assertEqual(result.continuation_audio["followup_text"], "三天了")
        self.assertEqual(mode.session.conversation_turns[-1].status, "awaiting_input")
        mode.session.execution_state.set_fact(
            "dialogue.followup_required", True, source="test", step_id="s2"
        )
        speak_rejection = mode.get_command_rejection(
            "我头晕",
            Command(type="tool_call", name="speak", params={"text": "请休息"}),
            successful_tools={"medical_consult"},
        )
        self.assertIn("MEDICAL_FOLLOWUP_REQUIRED", speak_rejection)
        mode.session.conversation_turns.append(ConversationTurn(
            index=2, task_id="prior", user="三天了", assistant="还有其他症状吗",
            status="awaiting_input", end_reason="query",
        ))
        mode.session.execution_state.set_fact(
            "dialogue.followup_required", True, source="test", step_id="s2"
        )
        rejection = mode.get_command_rejection(
            "站起来更晕",
            Command(type="tool_call", name="query", params={"question": "还有吗"}),
            successful_tools=set(),
        )
        self.assertIn("SESSION_TURN_LIMIT", rejection)

    def test_medical_speak_cannot_invent_department(self):
        class FakeSkills:
            def get_skills_description(self, mode): return ""
            def get_active_skill_prompt(self): return ""
            def deactivate_all(self): return None

        class TestMode(ModeBase):
            name = "Test"
            def get_system_prompt(self): return "fixed"

        mode = TestMode(mock.Mock(), MCPManager(), FakeSkills(), mock.Mock())
        mode.session = Session(mode="Test")
        plan = {
            "goal": "answer", "steps": [{"step_id": "s1", "goal": "answer"}]
        }
        from agents.Harness import TaskPlan, ExecutionState
        mode.session.execution_state = ExecutionState.create(TaskPlan.from_payload(plan))
        mode.session.execution_state.set_fact(
            "medical.consultation",
            {"status": "ok", "departments": [], "recommended_destination": ""},
            source="test", step_id="s1",
        )
        rejection = mode.get_command_rejection(
            "咨询疾病",
            Command(type="tool_call", name="speak", params={"text": "建议去骨科"}),
            successful_tools={"medical_consult"},
        )
        self.assertIn("MEDICAL_DEPARTMENT_UNSUPPORTED", rejection)
        mode.session.execution_state.set_fact(
            "navigation.target", "骨科", source="test", step_id="s1",
        )
        allowed = mode.get_command_rejection(
            "带我去骨科",
            Command(type="tool_call", name="speak", params={"text": "正在导航至骨科"}),
            successful_tools={"medical_consult", "navigate"},
        )
        self.assertEqual(allowed, "")

    def test_query_may_end_turn_before_session_success_conditions(self):
        from agents.Harness import ActionValidator, TaskPlan, ExecutionState, ToolMetadata
        state = ExecutionState.create(TaskPlan.from_payload({
            "goal": "consult and answer",
            "success_conditions": [
                {"fact": "speech.last_text", "operator": "exists"}
            ],
            "steps": [{"step_id": "s1", "goal": "ask follow-up"}],
        }))
        validation = ActionValidator().validate_tool(
            state=state, tool="query", arguments={"question": "持续多久？"},
            metadata=ToolMetadata.from_dict({
                "terminal": True, "turn_terminal": True,
                "session_terminal": False, "produces": ["dialogue.question"],
            }),
            available_tools=["query"],
        )
        self.assertTrue(validation.allowed, validation.message)

    def test_speak_is_blocked_before_last_unfinished_step(self):
        from agents.Harness import ActionValidator, TaskPlan, ExecutionState, ToolMetadata
        state = ExecutionState.create(TaskPlan.from_payload({
            "goal": "navigate and answer",
            "steps": [
                {"step_id": "s1", "goal": "navigate"},
                {"step_id": "s2", "goal": "answer", "depends_on": ["s1"]},
            ],
        }))
        validation = ActionValidator().validate_tool(
            state=state, tool="speak", arguments={"text": "提前结束"},
            metadata=ToolMetadata.from_dict({
                "terminal": True, "turn_terminal": True,
                "session_terminal": True, "produces": ["speech.last_text"],
            }),
            available_tools=["speak"],
        )
        self.assertFalse(validation.allowed)
        self.assertEqual(validation.error_code, "SESSION_TERMINAL_BEFORE_FINAL_STEP")

    def test_projection_makes_dynamic_medical_policy_explicit(self):
        from agents.Harness import TaskPlan, ExecutionState
        state = ExecutionState.create(TaskPlan.from_payload({
            "goal": "consult", "steps": [{"step_id": "s1", "goal": "answer"}],
        }))
        state.set_fact(
            "medical.consultation", {"status": "need_more_info", "departments": []},
            source="test", step_id="s1",
        )
        state.set_fact(
            "dialogue.followup_required", True, source="test", step_id="s1",
        )
        projection = state.projection()
        self.assertEqual(
            projection["dialogue_policy"]["required_next_tool"],
            "query_unless_session_turn_limit",
        )
        self.assertEqual(
            projection["medical_response_policy"]["department_rule"],
            "do_not_name_or_recommend_any_department",
        )

    def test_native_function_call_is_preserved_by_xml_adapter(self):
        content = _visible_message_content({
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"format":"time"}'},
            }],
            "reasoning_content": "must not be exposed",
        })
        self.assertEqual(
            content,
            '<tool>{"param":{"format":"time"},"tool_call":"get_time"}</tool>',
        )
        self.assertNotIn("reasoning", content)

    def test_toml_contains_local_and_deepseek(self):
        config = load_config()
        models = {item["display_name"]: item for item in config["servers"]}
        self.assertEqual(models["qwen3.5-4b-history"]["backend"], "local")
        deepseek = models["deepseek-v4-flash"]
        self.assertEqual(deepseek["backend"], "openai")
        self.assertEqual(deepseek["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertNotIn("api_key", deepseek)

    def test_xml_parser_preserves_plan_array(self):
        manager = MCPManager()
        manager.register(PlanTool())
        router = CallRouter(mcp=manager)
        parsed = router.parse_response(
            '<tool>{"tool_call":"plan","param":{"goal":"回答",'
            '"steps":["查询","播报"],"done_when":"完成"}}</tool>'
        )
        self.assertEqual(parsed.text, "")
        self.assertEqual(len(parsed.commands), 1)
        self.assertEqual(parsed.commands[0].params["steps"], ["查询", "播报"])
        result = asyncio.run(manager.execute(
            "plan", parsed.commands[0].params, ToolContext()
        ))
        self.assertTrue(result.success)

        local_style = asyncio.run(manager.execute("plan", {
            "goal": "介绍彩虹",
            "steps": [{
                "name": "speak", "params": {"text": "说明"},
                "condition": "none",
            }],
        }, ToolContext()))
        self.assertTrue(local_style.success)
        normalized = json.loads(local_style.data)
        self.assertIn("done_when", normalized)
        self.assertEqual(normalized["steps"][0]["preferred_tool"], "speak")
        self.assertEqual(normalized["steps"][0]["step_id"], "s1")

        legacy = router.parse_response(
            '{tool_call:"plan" param{goal:"回答",steps:["查询","播报"],done_when:"完成"}}'
        )
        self.assertEqual(legacy.commands[0].params["steps"], ["查询", "播报"])

        # Qwen 4B commonly omits exactly the outermost closing brace while
        # keeping the XML wrapper and inner JSON otherwise valid.
        repaired = router.parse_response(
            '<tool>{"tool_call":"plan","param":{"goal":"回答",'
            '"steps":[{"step_id":"s1","goal":"直接回答"}]}</tool>'
        )
        self.assertEqual(len(repaired.commands), 1)
        self.assertEqual(repaired.commands[0].params["steps"][0]["step_id"], "s1")

        misplaced = router.parse_response(
            '<tool>{"tool_call":"plan","param":{"goal":"回答",'
            '"steps":[{"step_id":"s1","goal":"直接回答",'
            '"condition":null,"verification":false}}]}</tool>'
        )
        self.assertEqual(len(misplaced.commands), 1)
        self.assertEqual(misplaced.commands[0].name, "plan")

    def test_trajectory_redacts_reasoning_images_and_secrets(self):
        session = Session(id="test-session")
        session.prompt_slots.start_task(system="fixed", user_input="hello")
        session.prompt_slots.set_plan({
            "goal": "answer", "steps": ["reply"], "done_when": "done"
        })
        session.begin_external_turn("hello", {"source": "benchmark_text", "asr_text": "hello"})
        session.finish_external_turn("final", end_reason="speak")
        session.prompt_slots.user_image = "data:image/jpeg;base64,SECRET_IMAGE"
        session.model_call_records.append({
            "task_id": session.prompt_slots.task_id,
            "output": "final",
            "reasoning_content": "hidden chain",
            "image": "data:image/jpeg;base64,SECRET_IMAGE",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = TrajectoryWriter(directory).write_turn(
                session=session,
                trace_id="trace",
                mode="Voice",
                status="completed",
                final_text="final",
                metrics={},
                events=[],
                model_config={"api_key": "SECRET_KEY", "model": "test"},
            )
            raw = path.read_text("utf-8")
            self.assertNotIn("hidden chain", raw)
            self.assertNotIn("SECRET_IMAGE", raw)
            self.assertNotIn("SECRET_KEY", raw)
            session_payload = json.loads((path.parent / "session.json").read_text("utf-8"))
            self.assertEqual(session_payload["schema_version"], "trajectory_session.v1")
            self.assertEqual(session_payload["status"], "completed")
            self.assertEqual(session_payload["turns"][0]["input_metadata"]["asr_text"], "hello")
        self.assertTrue(json.loads(raw)["has_image"])

    def test_mode_forces_plan_then_injects_it_into_user(self):
        class FakeAPI:
            def __init__(self):
                self.outputs = [
                    '<tool>{"tool_call":"plan","param":{"goal":"回答",'
                    '"steps":["直接回答"],"done_when":"已回答"}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"FINISH","response":"完成"}}</tool>',
                ]
                self.slots = []

            async def chat_complete(self, session):
                self.slots.append(session.prompt_slots.to_request_dict())
                return self.outputs.pop(0)

            async def emit_agent_event(self, *args, **kwargs):
                return None

        class FakeSkills:
            def get_skills_description(self, mode):
                return ""
            def get_active_skill_prompt(self):
                return ""
            def deactivate_all(self):
                return None

        class TestMode(ModeBase):
            name = "Test"
            def get_system_prompt(self):
                return "fixed"

        mcp = MCPManager()
        mcp.register(PlanTool())
        call_router = CallRouter(mcp=mcp)
        api = FakeAPI()
        mode = TestMode(api, mcp, FakeSkills(), call_router)
        result = asyncio.run(mode.loop("你好"))
        self.assertEqual(result.text, "完成")
        self.assertIn('kind="planning"', api.slots[0]["history"])
        self.assertNotIn('kind="planning"', api.slots[1]["history"])
        self.assertIn("<plan>", api.slots[1]["user"])
        self.assertEqual(mode.session.prompt_slots.history.entries, [])

    def test_repeated_failed_signature_is_blocked_before_reexecution(self):
        class FailingTool(Tool):
            name = "lookup"
            description = "always fails"
            param_schema = {"query": "text"}
            modes = ["Test"]
            def __init__(self):
                self.calls = 0
            async def call(self, params, context):
                self.calls += 1
                return ToolResult(success=False, error="no result", retryable=True)

        class FakeAPI:
            def __init__(self):
                self.outputs = [
                    '<tool>{"tool_call":"plan","param":{"goal":"查找",'
                    '"steps":["查询"],"done_when":"返回结果"}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"CALL_TOOL","tool":"lookup",'
                    '"arguments":{"query":"same"}}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"CALL_TOOL","tool":"lookup",'
                    '"arguments":{"query":"same"}}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"CALL_TOOL","tool":"lookup",'
                    '"arguments":{"query":"same"}}}</tool>',
                ]
            async def chat_complete(self, session):
                return self.outputs.pop(0)
            async def emit_agent_event(self, *args, **kwargs):
                return None

        class FakeSkills:
            def get_skills_description(self, mode): return ""
            def get_active_skill_prompt(self): return ""
            def deactivate_all(self): return None

        class TestMode(ModeBase):
            name = "Test"
            def get_system_prompt(self): return "fixed"

        failing = FailingTool()
        mcp = MCPManager()
        mcp.register(PlanTool())
        mcp.register(failing)
        mode = TestMode(FakeAPI(), mcp, FakeSkills(), CallRouter(mcp=mcp))
        result = asyncio.run(mode.loop("查找内容"))
        self.assertEqual(failing.calls, 1)
        self.assertIn("停止", result.text)
        categories = [item["category"] for item in mode.session.prompt_slots.attempt]
        self.assertIn("duplicate_no_world_change", categories)

    def test_execution_requires_act_wrapper_and_binds_current_step(self):
        class CountingTool(Tool):
            name = "lookup"
            description = "lookup"
            param_schema = {"query": "text"}
            modes = ["Test"]
            def __init__(self): self.calls = 0
            async def call(self, params, context):
                self.calls += 1
                return ToolResult(success=True, data="found")

        class TestSpeak(SpeakTool):
            modes = ["Test"]

        class FakeAPI:
            def __init__(self):
                self.outputs = [
                    '<tool>{"tool_call":"plan","param":{"goal":"查找",'
                    '"steps":[{"step_id":"s1","goal":"查询",'
                    '"preferred_tool":"lookup"},{"step_id":"s_final",'
                    '"goal":"播报结果","preferred_tool":"speak",'
                    '"depends_on":["s1"]}]}}</tool>',
                    '<tool>{"tool_call":"lookup","param":{"query":"x"}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s1",'
                    '"action_type":"CALL_TOOL","tool":"lookup",'
                    '"arguments":{"query":"x"}}}</tool>',
                    '<tool>{"tool_call":"act","param":{"step_id":"s_final",'
                    '"action_type":"CALL_TOOL","tool":"speak",'
                    '"arguments":{"text":"已完成"}}}</tool>',
                ]
            async def chat_complete(self, session): return self.outputs.pop(0)
            async def emit_agent_event(self, *args, **kwargs): return None

        class FakeSkills:
            def get_skills_description(self, mode): return ""
            def get_active_skill_prompt(self): return ""
            def deactivate_all(self): return None

        class TestMode(ModeBase):
            name = "Test"
            def get_system_prompt(self): return "fixed"

        tool = CountingTool()
        mcp = MCPManager(); mcp.register(PlanTool()); mcp.register(tool); mcp.register(TestSpeak())
        mode = TestMode(FakeAPI(), mcp, FakeSkills(), CallRouter(mcp=mcp))
        result = asyncio.run(mode.loop("查找"))
        self.assertEqual(result.text, "已完成")
        self.assertEqual(tool.calls, 1)
        self.assertEqual(mode.session.execution_state.status, "FINISHED")
        event_types = [x["type"] for x in mode.session.prompt_slots.execution_events]
        self.assertIn("ACT_REJECTED", event_types)
        self.assertIn("ACTION_APPLIED", event_types)

    def test_noninteractive_default_loads_local_model(self):
        request = httpx.Request("GET", "http://router/v1/models")
        initial = httpx.Response(200, request=request, json={"data": [{
            "id": "qwen3.5-4b-history", "type": "chat",
            "backend": "local", "status": "offline",
        }]})
        ready = httpx.Response(200, request=request, json={"data": [{
            "id": "qwen3.5-4b-history", "type": "chat",
            "backend": "local", "status": "online",
        }]})
        load = httpx.Response(
            200, request=httpx.Request("POST", "http://router/v1/load/model"),
            json={"status": "starting"},
        )
        with mock.patch.object(model_selection.httpx, "get", side_effect=[initial, ready]), \
             mock.patch.object(model_selection.httpx, "post", return_value=load) as post, \
             mock.patch.object(model_selection.time, "sleep"), \
             mock.patch.object(model_selection.sys.stdin, "isatty", return_value=False):
            selected = model_selection.select_model(
                "http://router", default="qwen3.5-4b-history"
            )
        self.assertEqual(selected, "qwen3.5-4b-history")
        post.assert_called_once()


class RemoteRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_key = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "test-secret"
        self.old_client = server_router._http_client

    async def asyncTearDown(self):
        if server_router._http_client is not None:
            await server_router._http_client.aclose()
        server_router._http_client = self.old_client
        if self.old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self.old_key

    async def test_prompt_slots_are_translated_with_thinking_enabled(self):
        observed = {}

        async def upstream(request):
            observed["authorization"] = request.headers.get("authorization")
            observed["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant", "content": "ok",
                    "reasoning_content": "not persisted by Agent",
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                          "total_tokens": 12, "prompt_cache_hit_tokens": 4,
                          "prompt_cache_miss_tokens": 6},
            })

        server_router._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        transport = httpx.ASGITransport(app=server_router.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://router") as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4-flash",
                "prompt_slots": {
                    "version": "suha.v2", "system": "sys", "user": "task",
                    "history": "[]", "attempt": "[]",
                },
                "stream": False,
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed["authorization"], "Bearer test-secret")
        body = observed["body"]
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertIn("<system>", body["messages"][0]["content"])
        self.assertIn("<history>", body["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
