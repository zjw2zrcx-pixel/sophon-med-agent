import unittest

from agents.TeacherData.extract_failure_corrections import call_failure_codes


class CallFailureAttributionTests(unittest.TestCase):
    def test_repair_is_owned_by_preceding_generation(self):
        calls = [
            {"output": "not an envelope", "benchmark": {"stop_reason": "length"}},
            {
                "output": '<tool>{"tool_call":"act","param":{}}</tool>',
                "attempt_entries": 1,
                "attempt_categories": ["FORMAT_ERROR"],
                "benchmark": {"stop_reason": "stop"},
            },
        ]
        turn = {"terminal_action": "speak", "contract_errors": []}
        first = call_failure_codes(turn, calls, 0)
        second = call_failure_codes(turn, calls, 1)
        self.assertIn("invalid_model_envelope", first)
        self.assertIn("generation_truncated", first)
        self.assertIn("caused_repair_attempt", first)
        self.assertIn("repair:FORMAT_ERROR", first)
        self.assertEqual(second, [])

    def test_terminal_contract_error_is_owned_by_last_generation(self):
        calls = [
            {"output": '<tool>{"tool_call":"act","param":{}}</tool>'},
            {"output": '<tool>{"tool_call":"act","param":{}}</tool>'},
        ]
        turn = {"terminal_action": "speak", "contract_errors": ["navigation_target"]}
        self.assertEqual(call_failure_codes(turn, calls, 0), [])
        self.assertEqual(
            call_failure_codes(turn, calls, 1), ["contract:navigation_target"]
        )


if __name__ == "__main__":
    unittest.main()
