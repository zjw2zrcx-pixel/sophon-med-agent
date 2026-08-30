from __future__ import annotations

import unittest

from agents.TeacherData.correct_navigation import correct_run


class CorrectNavigationTests(unittest.TestCase):
    def test_repairs_actions_and_state_without_rewriting_user_negation(self) -> None:
        run = {
            "id": "n1", "final": "正在前往收费处",
            "expected": {"navigation_target": "药房"},
            "turns": [{
                "input": "不是收费处，带我去拿药的地方",
                "commands": [{"name": "navigate", "params": {"target": "收费处"}}],
            }],
            "state": {"navigation.target": "收费处"},
        }
        repaired, old = correct_run(
            run, expected_target="药房", reviewer="tester", source="source.json"
        )
        self.assertEqual(old, "收费处")
        self.assertEqual(repaired["turns"][0]["input"], "不是收费处，带我去拿药的地方")
        self.assertEqual(repaired["turns"][0]["commands"][0]["params"]["target"], "药房")
        self.assertEqual(repaired["state"]["navigation.target"], "药房")
        self.assertEqual(repaired["final"], "正在前往药房")


if __name__ == "__main__":
    unittest.main()
