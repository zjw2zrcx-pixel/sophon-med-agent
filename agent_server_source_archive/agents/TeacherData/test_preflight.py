from __future__ import annotations

import unittest

from agents.TeacherData.preflight import preflight_cases


class TeacherPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonmedical_case_does_not_call_medical_tool(self) -> None:
        result = await preflight_cases([{
            "id": "general-1", "category": "general", "prompt": "一公斤是多少克",
            "turns": ["一公斤是多少克"], "expected": {},
        }], workers=1)
        self.assertEqual(result["summary"], {"total": 1, "eligible": 1, "reject": 0})
        self.assertEqual(result["cases"][0]["checks"], [])


if __name__ == "__main__":
    unittest.main()
