from __future__ import annotations

import asyncio
import json
import unittest

from agents.MCP.base import ToolContext
from agents.MCP.tools.medconsult import MedicalConsultTool


class MedicalConsultDialogueTests(unittest.TestCase):
    def test_explicitly_missing_subject_requests_clarification_without_retrieval(self) -> None:
        tool = MedicalConsultTool(dense_enabled=False)
        result = asyncio.run(tool.call(
            {"query": "我想咨询一种疾病的推荐药，但还没说具体名称。"},
            ToolContext(),
        ))
        self.assertTrue(result.success)
        payload = json.loads(result.data)
        self.assertEqual(payload["status"], "need_more_info")
        self.assertEqual(payload["retrieval"]["mode"], "not_run")
        self.assertTrue(result.facts["dialogue.followup_required"])


if __name__ == "__main__":
    unittest.main()
