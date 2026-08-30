from __future__ import annotations

import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agents.TeacherData.medical_prompts import MedicalPromptSampler, MIXED_SAFE_TARGETS
from agents.TeacherData.generate import _category_batches, _category_counts


class MedicalPromptSamplerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "medical.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE documents(id INTEGER PRIMARY KEY, question TEXT, "
                "answer TEXT, source TEXT)"
            )
            connection.execute(
                "CREATE TABLE facts(id INTEGER PRIMARY KEY, subject TEXT, aspect TEXT, "
                "answer TEXT, source TEXT, quality REAL)"
            )
            for index in range(1, 9):
                connection.execute(
                    "INSERT INTO documents VALUES(?,?,?,?)",
                    (index, f"训练数据库医疗问题{index}怎么处理",
                     "这是数据库中的参考文本，内容长度足够用于验证来源对齐。" * 2,
                     "fixture/train"),
                )
            connection.execute(
                "INSERT INTO facts VALUES(1,'测试疾病','常见表现',"
                "'数据库事实回答内容足够长','fixture/train',1.0)"
            )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _cases():
        return [
            {"id": "m1", "category": "medical", "prompt": "invented",
             "turns": ["invented"],
             "expected": {"required_tools": ["medical_consult", "query"]}},
            {"id": "m2", "category": "medical", "prompt": "invented",
             "expected": {"required_tools": ["medical_consult"]}},
            {"id": "x1", "category": "mixed", "prompt": "invented",
             "expected": {"required_tools": ["medical_consult", "navigate", "query"],
                          "navigation_target": "挂号处"}},
        ]

    def test_single_turn_prompts_are_database_questions(self):
        cases = MedicalPromptSampler(self.database, seed=7).ground_cases(
            self._cases(), multiturn_ratio=0.0
        )
        self.assertTrue(cases[0]["prompt"].startswith("训练数据库医疗问题"))
        self.assertEqual(cases[0]["turns"], [cases[0]["prompt"]])
        self.assertNotIn("query", cases[0]["expected"]["required_tools"])
        self.assertEqual(cases[0]["medical_source"]["database_table"], "documents")
        self.assertTrue(cases[0]["followup_support"]["knowledge_snippets"])
        self.assertEqual(cases[0]["followup_support"]["max_session_turns"], 3)
        target = cases[2]["expected"]["navigation_target"]
        self.assertIn(target, MIXED_SAFE_TARGETS)
        self.assertIn(f"请带我去{target}", cases[2]["prompt"])

    def test_masked_multiturn_restores_full_fact_question(self):
        cases = MedicalPromptSampler(self.database, seed=7).ground_cases(
            self._cases(), multiturn_ratio=0.34
        )
        first = cases[0]
        self.assertEqual(len(first["turns"]), 2)
        self.assertNotIn("测试疾病", first["turns"][0])
        self.assertIn("测试疾病的常见表现是什么", first["turns"][1])
        self.assertIn("query", first["expected"]["required_tools"])
        self.assertEqual(first["medical_source"]["mask_strategy"], "subject")

    def test_normalize_mixed_preserves_preflight_followup_turns(self):
        cases = MedicalPromptSampler(self.database, seed=7).ground_cases(
            self._cases(), multiturn_ratio=0.0
        )
        mixed = next(case for case in cases if case["category"] == "mixed")
        mixed["turns"] = [mixed["prompt"], "这是预检后提供的补充信息"]
        mixed["preflight_materialized_followup"] = True
        MedicalPromptSampler.normalize_existing_cases(cases)
        self.assertEqual(len(mixed["turns"]), 2)
        self.assertEqual(mixed["turns"][1], "这是预检后提供的补充信息")

    def test_category_allocation_scales_beyond_pilot_ten(self):
        self.assertEqual(
            _category_counts(10),
            {"medical": 3, "navigation": 2, "general": 2, "mixed": 3},
        )
        self.assertEqual(sum(_category_counts(137).values()), 137)
        self.assertGreater(_category_counts(137)["medical"], 0)
        batches = _category_batches(137, 10)
        self.assertEqual(len(batches), 14)
        self.assertEqual(sum(sum(batch.values()) for batch in batches), 137)
        aggregate = Counter()
        for batch in batches:
            aggregate.update(batch)
        self.assertEqual(dict(aggregate), _category_counts(137))


if __name__ == "__main__":
    unittest.main()
