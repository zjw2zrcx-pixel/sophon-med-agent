from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agents.CallRoute.parser.parser import Command
from agents.CallRoute.router import CallRouter
from agents.MCP.base import Tool, ToolContext, ToolResult
from agents.MCP.manager import MCPManager
from agents.MCP.tools.medconsult import MedicalConsultTool
from agents.MCP.tools.voice import QueryTool, SpeakTool
from agents.Modes.voice import VoiceMode
from agents.medical_policy import grounded_medical_speech


class _Skills:
    def get_skills_description(self, mode):
        return ""

    def get_active_skill_prompt(self):
        return ""

    def deactivate_all(self):
        return None


class _MedicalStub(Tool):
    name = "medical_consult"
    description = "medical"
    param_schema = {"query": "query"}
    modes = ["Voice"]
    harness_metadata = {
        "effect": "READ", "idempotent": True,
        "produces": ["medical.consultation"],
        "retry": {"max_attempts": 1},
    }

    async def call(self, params, context):
        consultation = {
            "status": "ok",
            "intent": "medication",
            "questions": [],
            "departments": [],
            "recommended_destination": "",
            "medication_allowed": True,
            "medication_notice": (
                "药物关系仅是资料关联，不代表适合当前用户；"
                "抗菌药、抗病毒药及处方药须由医生评估后使用。"
            ),
            "evidence": [{"text": "头孢拉定胶囊；阿莫西林颗粒"}],
            "associations": [],
        }
        return ToolResult(
            success=True,
            data=json.dumps(consultation, ensure_ascii=False),
            facts={
                "medical.consultation": consultation,
                "dialogue.followup_required": False,
                "dialogue.followup_questions": [],
            },
            diagnostics={"schema_version": "medical-consult-timing.v1"},
        )


class DirectHarnessPerformanceTests(unittest.TestCase):
    def test_medical_workflow_consults_then_speaks_without_model_turn(self):
        class API:
            calls = 0

            async def chat_complete(self, session):
                self.calls += 1
                raise AssertionError("compact medical workflow must not call LLM")

            async def emit_agent_event(self, *args, **kwargs):
                return None

        mcp = MCPManager()
        mcp.register_many([_MedicalStub(), QueryTool(), SpeakTool()])
        api = API()
        mode = VoiceMode(api, mcp, _Skills(), CallRouter(mcp=mcp))
        result = asyncio.run(mode.loop("我感冒了，应该吃什么药？"))
        self.assertEqual([item[0].name for item in result.commands], [
            "medical_consult", "speak",
        ])
        self.assertEqual(api.calls, 0)
        self.assertNotIn("头孢", result.text)
        self.assertIn("已有医嘱", result.text)
        self.assertEqual(mode.session.execution_state.status, "FINISHED")

    def test_general_model_turn_calls_business_tool_directly(self):
        class API:
            calls = 0
            saw_planning = None

            async def chat_complete(self, session):
                self.calls += 1
                slots = session.prompt_slots.to_request_dict()
                self.saw_planning = 'kind="planning"' in slots["history"]
                return '<tool>{"tool_call":"speak","param":{"text":"好的。"}}</tool>'

            async def emit_agent_event(self, *args, **kwargs):
                return None

        mcp = MCPManager()
        mcp.register(SpeakTool())
        api = API()
        mode = VoiceMode(api, mcp, _Skills(), CallRouter(mcp=mcp))
        result = asyncio.run(mode.loop("讲个简短笑话"))
        self.assertEqual(result.text, "好的。")
        self.assertEqual(api.calls, 1)
        self.assertFalse(api.saw_planning)
        self.assertEqual(result.commands[0][0].name, "speak")
        self.assertNotIn("### act", mcp.get_tool_description("Voice"))
        self.assertNotIn("### plan", mcp.get_tool_description("Voice"))

    def test_medical_speech_never_turns_drug_evidence_into_instruction(self):
        text = grounded_medical_speech({
            "status": "ok",
            "intent": "medication",
            "medication_allowed": True,
            "medication_notice": "处方药须由医生评估后使用。",
            "evidence": [{"text": "头孢拉定胶囊；阿莫西林颗粒"}],
        })
        self.assertNotIn("头孢拉定", text)
        self.assertNotIn("阿莫西林", text)
        self.assertIn("已有医嘱", text)

    def test_medical_tool_exposes_structured_stage_timings(self):
        class Retriever:
            def consult(self, query):
                return {
                    "status": "ok", "query": query, "intent": "causes",
                    "questions": [], "evidence": [{"text": "受控证据"}],
                    "retrieval": {
                        "mode": "hybrid", "dense_used": True,
                        "parallel_total_ms": 4.5,
                    },
                }

        tool = MedicalConsultTool(dense_enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.sqlite"
            index.touch()
            with mock.patch.object(tool, "_index_path", return_value=index), \
                    mock.patch.object(tool, "_retriever", return_value=Retriever()):
                result = asyncio.run(tool.call(
                    {"query": "原因是什么"}, ToolContext()
                ))
        self.assertTrue(result.success)
        self.assertEqual(
            result.diagnostics["schema_version"],
            "medical-consult-timing.v1",
        )
        self.assertIn("retriever_prepare_ms", result.diagnostics["stages"])
        self.assertIn("consult_ms", result.diagnostics["stages"])
        self.assertEqual(result.diagnostics["retrieval"]["parallel_total_ms"], 4.5)

    def test_call_router_preserves_tool_diagnostics(self):
        mcp = MCPManager()
        mcp.register(_MedicalStub())
        router = CallRouter(mcp=mcp)
        result = asyncio.run(router.execute_command(Command(
            type="tool_call", name="medical_consult", params={"query": "感冒"},
            confidence=1.0,
        )))
        self.assertEqual(
            result.diagnostics["schema_version"],
            "medical-consult-timing.v1",
        )


if __name__ == "__main__":
    unittest.main()
