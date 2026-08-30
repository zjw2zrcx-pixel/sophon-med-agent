from __future__ import annotations

import unittest

from agents.TeacherData.split_dataset import split_cases


class SplitDatasetTests(unittest.TestCase):
    def test_same_medical_source_never_crosses_split(self) -> None:
        cases = []
        for index in range(10):
            cases.append({
                "id": f"m{index}", "category": "medical",
                "prompt": f"医疗问题{index}", "turns": [f"医疗问题{index}"],
                "medical_source": {"answer_sha256": "same" if index < 2 else str(index)},
            })
        train, benchmark, manifest = split_cases(cases, 0.2, "fixed")
        locations = {
            case["id"]: case["split"] for case in train + benchmark
        }
        self.assertEqual(locations["m0"], locations["m1"])
        self.assertEqual(len(train) + len(benchmark), 10)
        self.assertEqual(manifest["total"], 10)

    def test_prior_medical_source_keeps_historical_split(self) -> None:
        cases = [{
            "id": "m1", "category": "medical", "prompt": "问题",
            "medical_source": {"answer_sha256": "locked"},
        }, {
            "id": "m2", "category": "medical", "prompt": "另一个问题",
            "medical_source": {"answer_sha256": "free"},
        }]
        train, benchmark, manifest = split_cases(
            cases, 0.5, "fixed", {"medical:locked": "benchmark"}
        )
        locations = {case["id"]: case["split"] for case in train + benchmark}
        self.assertEqual(locations["m1"], "benchmark")
        self.assertEqual(manifest["locked_medical_source_count"], 1)

    def test_prior_prompt_keeps_historical_split(self) -> None:
        cases = [{
            "id": "g1", "category": "general", "prompt": "一周有几天？",
        }, {
            "id": "g2", "category": "general", "prompt": "水的化学式是什么？",
        }]
        train, benchmark, manifest = split_cases(
            cases, 0.5, "fixed", {},
            {"prompt:general:一周有几天": "benchmark"},
        )
        locations = {case["id"]: case["split"] for case in train + benchmark}
        self.assertEqual(locations["g1"], "benchmark")
        self.assertEqual(manifest["locked_prompt_count"], 1)


if __name__ == "__main__":
    unittest.main()
