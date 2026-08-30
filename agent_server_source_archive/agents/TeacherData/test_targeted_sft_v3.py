from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agents.TeacherData.targeted_sft_v3 import (
    compile_reviewed,
    exclusion_registry,
    parse_target_output,
    prepare_semantic_review_requests,
    validate_target,
)


def _scenario(*, evidence_gap: bool = False) -> dict:
    return {
        "schema_version": "targeted-sft-scenario.v1",
        "case_id": "case-1",
        "category": "medical",
        "split": "train",
        "semantic_family_id": "family-1",
        "input": {"prompt_slots": {"version": "suha.v3"}, "model": "model"},
        "medical_context": {
            "status": "ok", "requested_aspect": "病因",
            "evidence_gap": evidence_gap, "departments": [],
            "evidence": [{"evidence_id": "ev-1", "text": "证据内容"}],
        },
        "supervision": {
            "decision_role": "medical_final_synthesis",
            "intent_aspect": "病因", "required_tool": "speak",
            "required_step_id": "s_final",
            "forbidden_tools": ["medical_consult", "navigate", "query"],
            "evidence_ids": ["ev-1"], "evidence_gap": evidence_gap,
            "max_output_chars": 90, "semantic_review_required": True,
        },
        "provenance": {"prompt_protocol": "suha.v3"},
        "scenario_sha256": "scenario-hash",
    }


def _output(text: str) -> str:
    return "<tool>" + json.dumps({
        "tool_call": "act",
        "param": {
            "step_id": "s_final", "action_type": "CALL_TOOL",
            "tool": "speak", "arguments": {"text": text},
        },
    }, ensure_ascii=False, separators=(",", ":")) + "</tool>"


class TargetedSFTV3Tests(unittest.TestCase):
    def test_previous_targeted_batches_are_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "teacher_trajectories" / "targeted_sft_v3_previous"
            batch.mkdir(parents=True)
            row = {
                "prompt": "某病有哪些症状？",
                "semantic_family_id": "family-old",
                "source": {
                    "source": "facts", "id": 17,
                    "answer_sha256": "answer-old",
                },
            }
            (batch / "scenarios.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            registry = exclusion_registry(root)
            self.assertIn(("facts", 17), registry["source_rows"])
            self.assertIn("answer-old", registry["answer_hashes"])
            self.assertIn("family-old", registry["family_ids"])

    def test_parser_requires_strict_act_speak_contract(self):
        parsed, errors = parse_target_output(_output("证据支持该结论。"))
        self.assertEqual(errors, [])
        self.assertEqual(parsed["text"], "证据支持该结论。")
        _, errors = parse_target_output('<tool>{"tool_call":"speak"}</tool>')
        self.assertIn("NOT_ACT", errors)
        self.assertIn("WRONG_STEP_ID", errors)

    def test_evidence_gap_must_be_explicit(self):
        invalid = validate_target(_scenario(evidence_gap=True), _output("建议进一步评估。"))
        self.assertFalse(invalid["deterministic_pass"])
        self.assertIn("EVIDENCE_GAP_NOT_EXPLICIT", invalid["errors"])
        valid = validate_target(
            _scenario(evidence_gap=True),
            _output("现有本地资料未覆盖该病因，建议进一步评估。"),
        )
        self.assertTrue(valid["deterministic_pass"])

    def test_compile_requires_semantic_approval_and_valid_evidence_map(self):
        scenario = _scenario()
        candidate = {
            "case_id": "case-1", "teacher_model": "teacher",
            "output": _output("资料显示该因素与疾病有关。"),
        }
        samples, report = compile_reviewed([scenario], [candidate], [])
        self.assertEqual(samples, [])
        self.assertEqual(report["rejected"]["semantic_review_not_approved"], 1)
        bad_review = {
            "schema_version": "targeted-semantic-review.v1",
            "case_id": "case-1", "decision": "approve",
            "claim_evidence_map": [{
                "claim": "结论", "kind": "medical_fact", "support": "direct",
                "evidence_ids": ["outside"],
            }],
        }
        samples, report = compile_reviewed([scenario], [candidate], [bad_review])
        self.assertEqual(samples, [])
        self.assertEqual(report["rejected"]["invalid_claim_evidence_map"], 1)
        review = {
            **bad_review,
            "claim_evidence_map": [{
                "claim": "结论", "kind": "medical_fact", "support": "direct",
                "evidence_ids": ["ev-1"],
            }],
        }
        samples, report = compile_reviewed([scenario], [candidate], [review])
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(samples[0]["schema_version"], "agent-sft-decision.v3")
        self.assertEqual(samples[0]["action"], "act")
        self.assertEqual(samples[0]["gate"]["human_spot_review"], "required")
        self.assertFalse(report["trainable"])
        self.assertEqual(
            samples[0]["supervision"]["claim_evidence_map"],
            review["claim_evidence_map"],
        )
        ai_samples, ai_report = compile_reviewed(
            [scenario], [candidate], [review], human_review_required=False,
        )
        self.assertEqual(
            ai_samples[0]["gate"]["human_spot_review"],
            "waived_by_user_ai_review",
        )
        self.assertTrue(ai_report["trainable"])

    def test_length_sentence_and_department_gates(self):
        scenario = _scenario()
        no_boundary = validate_target(scenario, _output("没有句末"))
        self.assertIn("INCOMPLETE_SENTENCE_BOUNDARY", no_boundary["errors"])
        department = validate_target(scenario, _output("建议去皮肤科。"))
        self.assertIn("UNSUPPORTED_DEPARTMENT:皮肤科", department["errors"])
        too_long = validate_target(scenario, _output("甲" * 91 + "。"))
        self.assertIn("OUTPUT_TOO_LONG", too_long["errors"])

    def test_gap_sample_requires_uncertainty_claim_mapping(self):
        scenario = _scenario(evidence_gap=True)
        candidate = {
            "case_id": "case-1", "teacher_model": "teacher",
            "output": _output("现有本地资料未覆盖该病因。"),
        }
        empty_review = {
            "schema_version": "targeted-semantic-review.v1",
            "case_id": "case-1", "decision": "approve",
            "claim_evidence_map": [],
        }
        samples, report = compile_reviewed(
            [scenario], [candidate], [empty_review]
        )
        self.assertEqual(samples, [])
        self.assertEqual(report["rejected"]["invalid_claim_evidence_map"], 1)
        review = {
            **empty_review,
            "claim_evidence_map": [{
                "claim": "本地资料未覆盖该病因",
                "kind": "uncertainty", "support": "conservative_policy",
                "evidence_ids": [],
            }],
        }
        samples, report = compile_reviewed([scenario], [candidate], [review])
        self.assertEqual(report["sample_count"], 1)
        self.assertTrue(samples[0]["supervision"]["evidence_gap"])

    def test_review_requests_only_include_deterministic_teacher_outputs(self):
        scenario = _scenario()
        valid = {"case_id": "case-1", "output": _output("证据支持该结论。")}
        requests, report = prepare_semantic_review_requests([scenario], [valid])
        self.assertEqual(report["review_request_count"], 1)
        self.assertEqual(requests[0]["candidate_text"], "证据支持该结论。")
        self.assertEqual(requests[0]["allowed_evidence_ids"], ["ev-1"])
        invalid = {"case_id": "case-1", "output": "普通文本"}
        requests, report = prepare_semantic_review_requests([scenario], [invalid])
        self.assertEqual(requests, [])
        self.assertEqual(report["rejected"]["INVALID_TOOL_ENVELOPE"], 1)

    def test_duplicate_or_unknown_records_never_silently_override(self):
        scenario = _scenario()
        candidate = {"case_id": "case-1", "output": _output("证据支持结论。")}
        requests, report = prepare_semantic_review_requests(
            [scenario], [candidate, candidate, {"case_id": "unknown", "output": "x"}]
        )
        self.assertEqual(requests, [])
        self.assertEqual(report["rejected"]["duplicate_teacher_output"], 1)
        self.assertEqual(report["rejected"]["unknown_teacher_output"], 1)


if __name__ == "__main__":
    unittest.main()
