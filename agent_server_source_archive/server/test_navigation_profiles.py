from __future__ import annotations

import unittest

from agents.MCP.base import ToolContext
from agents.MCP.tools.navigate import NavigateTool


class NavigationProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_hospital_profile_accepts_registration_and_departments(self):
        tool = NavigateTool(execution_mode="mock", location_profile="hospital")
        for spoken, expected in [
            ("带我去挂号窗口", "挂号处"),
            ("领我去呼吸科", "呼吸内科"),
            ("请前往心内科", "心血管内科"),
            ("带我到抽血处", "检验科"),
        ]:
            request = tool.match_request(spoken)
            self.assertEqual(request, ("start", expected))
            result = await tool.call(
                {"action": "start", "target": expected}, ToolContext()
            )
            self.assertTrue(result.success)
            self.assertEqual(result.facts["navigation.target"], expected)

    def test_basic_profile_does_not_enable_unmapped_real_destination(self):
        tool = NavigateTool(execution_mode="real", location_profile="basic")
        self.assertIsNone(tool.match_request("带我去呼吸科"))

