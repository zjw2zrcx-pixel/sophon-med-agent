from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agents.Medical.extract_medconsult_trajectories import extract


class ExtractMedicalConsultTrajectoriesTests(unittest.TestCase):
    def test_extracts_compacts_and_deduplicates_tool_calls(self) -> None:
        result = {
            "status": "ok",
            "intent": "causes",
            "associations": [],
            "evidence": [
                {
                    "type": "document",
                    "question": "测试疾病的病因",
                    "text": "很长的证据" * 100,
                    "source": "fixture",
                }
            ],
        }
        call = {
            "name": "medical_consult",
            "params": {"query": "测试疾病为什么发生"},
            "result": json.dumps(result, ensure_ascii=False),
            "scenario_id": "case-1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tool_calls.jsonl"
            path.write_text(
                json.dumps(call, ensure_ascii=False) + "\n"
                + json.dumps(call, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            records = extract([path], text_limit=80)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["occurrence_count"], 2)
        self.assertLessEqual(len(records[0]["result"]["evidence"][0]["text"]), 80)


if __name__ == "__main__":
    unittest.main()
