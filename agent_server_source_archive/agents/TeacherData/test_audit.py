from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agents.TeacherData.audit import audit_runs


class TeacherAuditTests(unittest.TestCase):
    def _audit(self, run: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "trajectories" / run["id"]
            target.mkdir(parents=True)
            (target / "000001_task.json").write_text(
                json.dumps({"errors": []}), encoding="utf-8"
            )
            return audit_runs(root, {"runs": [run]})["cases"][0]

    @staticmethod
    def _command(name: str, **extra) -> dict:
        return {"name": name, "success": True, "params": {}, **extra}

    def test_clean_navigation_is_accepted(self) -> None:
        commands = [
            self._command("plan"),
            self._command("navigate", params={"action": "start", "target": "药房"}),
            self._command("speak", params={"text": "已开始导航至药房"}),
        ]
        result = self._audit({
            "id": "nav-1", "category": "navigation", "status": "completed",
            "session_end_reason": "speak", "final": "已开始导航至药房",
            "expected": {"required_tools": ["navigate"], "forbidden_tools": [],
                         "navigation_target": "药房"},
            "turns": [{"end_reason": "speak", "commands": commands}],
        })
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["reasons"], [])

    def test_navigation_to_clinical_department_is_not_medical_hallucination(self) -> None:
        commands = [
            self._command("plan"),
            self._command("navigate", params={"action": "start", "target": "骨科"}),
            self._command("speak", params={"text": "已开始导航至骨科"}),
        ]
        result = self._audit({
            "id": "nav-department", "category": "navigation", "status": "completed",
            "session_end_reason": "speak", "final": "已开始导航至骨科",
            "expected": {"required_tools": ["navigate"], "forbidden_tools": [],
                         "navigation_target": "骨科"},
            "turns": [{"end_reason": "speak", "commands": commands}],
        })
        self.assertEqual(result["decision"], "accept")

    def test_ignored_medical_followup_is_rejected(self) -> None:
        commands = [
            self._command("plan"),
            self._command("medical_consult", medical={
                "status": "need_more_info", "questions": ["持续多久？"],
                "departments": [], "recommended_destination": "",
            }),
            self._command("speak", params={"text": "建议去骨科"}),
        ]
        result = self._audit({
            "id": "med-1", "category": "medical", "status": "completed",
            "session_end_reason": "speak", "final": "建议去骨科",
            "expected": {"required_tools": ["medical_consult"], "forbidden_tools": []},
            "turns": [{"end_reason": "speak", "commands": commands}],
        })
        self.assertEqual(result["decision"], "reject")
        self.assertIn("TURN_1_MEDICAL_FOLLOWUP_IGNORED", result["reasons"])
        self.assertIn("UNSUPPORTED_DEPARTMENT:骨科", result["reasons"])

    def test_clean_medical_requires_semantic_review(self) -> None:
        commands = [
            self._command("plan"),
            self._command("medical_consult", medical={
                "status": "ok", "questions": [], "departments": [],
                "recommended_destination": "",
            }),
            self._command("speak", params={"text": "请结合医生评估"}),
        ]
        result = self._audit({
            "id": "med-2", "category": "medical", "status": "completed",
            "session_end_reason": "speak", "final": "请结合医生评估",
            "expected": {"required_tools": ["medical_consult"], "forbidden_tools": []},
            "turns": [{"end_reason": "speak", "commands": commands}],
        })
        self.assertEqual(result["decision"], "semantic_review_required")

    def test_null_navigation_target_is_not_a_requirement(self) -> None:
        commands = [
            self._command("plan"),
            self._command("get_system_stats"),
            self._command("speak", params={"text": "运行正常"}),
        ]
        result = self._audit({
            "id": "general-1", "category": "general", "status": "completed",
            "session_end_reason": "speak", "final": "运行正常",
            "expected": {"required_tools": ["get_system_stats"],
                         "forbidden_tools": [], "navigation_target": None},
            "turns": [{"end_reason": "speak", "commands": commands}],
        })
        self.assertEqual(result["decision"], "accept")


if __name__ == "__main__":
    unittest.main()
