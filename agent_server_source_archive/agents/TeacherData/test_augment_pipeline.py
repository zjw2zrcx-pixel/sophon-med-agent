"""Regression tests for frozen-root case-level augmentation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agents.TeacherData.augment_pipeline import _tree_hash, run_pipeline


class AugmentPipelineTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[2] / "teacher_trajectories" / "final_v7_1000"

    def test_balanced_pilot_does_not_modify_root(self) -> None:
        before = _tree_hash(self.root)
        with tempfile.TemporaryDirectory(prefix="teacher_aug_test_") as tmp:
            metadata = run_pipeline(self.root, Path(tmp), 4, 1, 1)
            after = _tree_hash(self.root)
            self.assertEqual(before, after)
            self.assertEqual(metadata["root_snapshot_sha256"], before)
            self.assertEqual(metadata["root_case_count"], 4)
            self.assertEqual(metadata["variant_counts"]["original"], 4)

            contracts = [
                json.loads(line)
                for line in (Path(tmp) / "semantic_contracts.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual({row["category"] for row in contracts}, {"medical", "navigation", "mixed", "general"})
            signatures = {row["root_case_id"]: row["semantic_signature"] for row in contracts}
            for line in (Path(tmp) / "lineage.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertEqual(row["semantic_signature"], signatures[row["root_case_id"]])
                self.assertFalse(row["trajectory_changed"])

            mappings = [
                json.loads(line)
                for line in (Path(tmp) / "prompt_mapping.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(mappings), 4)
            self.assertEqual({row["root_case_id"] for row in mappings}, set(signatures))
            for row in mappings:
                self.assertEqual(row["replacement_count"], len(row["replacements"]))
                self.assertEqual(row["rejected_count"], len(row["rejected_candidates"]))
                self.assertTrue(all(item["trajectory_hash"] == row["original"]["trajectory_hash"] for item in row["replacements"]))
                semantic_ids = {
                    item["case_id"] for item in row["replacements"]
                    if item["variant_type"] == "semantic"
                }
                for item in row["replacements"]:
                    if item["variant_type"] == "asr":
                        self.assertIn(item["parent_variant"], semantic_ids)


if __name__ == "__main__":
    unittest.main()
