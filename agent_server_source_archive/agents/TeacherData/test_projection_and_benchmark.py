from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agents.TeacherData.benchmark_report import build_report
from agents.TeacherData.project_sft import project


class ProjectionAndBenchmarkTests(unittest.TestCase):
    def test_projection_skips_overflow_and_non_act_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trajectories").mkdir()
            (root / "audit.json").write_text(json.dumps({"cases": [{
                "id": "c1", "decision": "accept", "trajectory_files": ["trajectories/t.json"]
            }]}), encoding="utf-8")
            (root / "prompts.json").write_text(json.dumps({"cases": [{
                "id": "c1", "category": "general", "difficulty": "hard",
                "prompt": "问题", "turns": ["问题"],
                "expected": {"required_tools": []}, "risk_tags": ["asr"]
            }]}), encoding="utf-8")
            (root / "trajectories/t.json").write_text(json.dumps({"model_calls": [
                {"output": '<tool>{"tool_call":"plan"}</tool>', "context_stats": {}},
                {"output": '<tool>{"tool_call":"act"}</tool>',
                 "context_stats": {"context_overflow": True}},
                {"output": "plain answer", "context_stats": {}},
            ]}), encoding="utf-8")
            samples, manifest = project(root, {"accept"})
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["action"], "plan")
            self.assertEqual(samples[0]["category"], "general")
            self.assertEqual(samples[0]["tags"]["conversation_type"], "single_turn")
            self.assertEqual(manifest["category_cases"], {"general": 1})
            self.assertEqual(manifest["skipped_context_overflow_calls"], 1)

    def test_benchmark_report_separates_quality_and_execution_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs.json").write_text(json.dumps({"runs": [
                {"id": "a", "category": "medical", "status": "completed",
                 "elapsed_seconds": 2, "token_usage": {"total_tokens_sum": 100}},
                {"id": "b", "category": "navigation", "status": "error",
                 "elapsed_seconds": 4, "token_usage": {"context_overflow": True}},
            ]}), encoding="utf-8")
            (root / "audit.json").write_text(json.dumps({"cases": [
                {"id": "a", "decision": "accept", "trajectory_files": []},
                {"id": "b", "decision": "reject", "trajectory_files": [],
                 "trajectory_errors": ['{"category": "state"}']},
            ]}), encoding="utf-8")
            report = build_report(root)
            self.assertEqual(report["quality"]["strict_accept_rate"], 0.5)
            self.assertEqual(report["quality"]["run_error_rate"], 0.5)
            self.assertEqual(report["tokens"]["context_overflow_cases"], 1)


if __name__ == "__main__":
    unittest.main()
