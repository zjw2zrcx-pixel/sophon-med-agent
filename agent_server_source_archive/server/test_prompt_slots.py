from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from agents.API.session import PromptSlots  # noqa: E402
from qwen3_5_history_server import _messages_from_prompt_slots  # noqa: E402


class PromptSlotTests(unittest.TestCase):
    def setUp(self):
        self.slots = PromptSlots()
        self.slots.start_task(
            system="stable system",
            user_input="带我去药房",
            task_policy="navigate when not already committed",
        )

    def test_attempt_refresh_keeps_prior_slots_identical(self):
        before = copy.deepcopy(self.slots.to_request_dict())
        self.slots.fail(
            category="syntax_error",
            error="invalid tool syntax",
            raw="{tool_call ???}",
        )
        after = self.slots.to_request_dict()
        for name in ("system", "user", "history"):
            self.assertEqual(before[name], after[name])
        self.assertNotEqual(before["attempt"], after["attempt"])

    def test_commit_refreshes_history_and_resets_attempt(self):
        self.slots.fail(category="tool_failure", error="timeout", name="navigate")
        before = copy.deepcopy(self.slots.to_request_dict())
        self.slots.commit(
            command_type="tool_call",
            name="navigate",
            params={"target": "药房", "action": "start"},
            model_output='{tool_call:"navigate" param{action:"start", target:"药房"}}',
            result="状态: 成功",
        )
        after = self.slots.to_request_dict()
        self.assertEqual(before["system"], after["system"])
        self.assertEqual(before["user"], after["user"])
        self.assertNotEqual(before["history"], after["history"])
        self.assertEqual(after["attempt"], "[]")

    def test_history_archives_all_entries_but_displays_bounded_tail(self):
        self.slots.history.set_max_visible(2)
        for index in range(4):
            self.slots.commit(
                command_type="tool_call",
                name=f"tool_{index}",
                params={"value": str(index)},
                model_output=f'{{tool_call:"tool_{index}" param{{value:"{index}"}}}}',
                result=f"result {index}",
            )
        self.assertEqual(len(self.slots.history.entries), 4)
        self.assertEqual(self.slots.history.hidden_count, 2)
        visible = self.slots.history.visible_entries()
        self.assertEqual([entry["index"] for entry in visible], [3, 4])
        self.assertEqual(visible[-1]["model_output"], '{tool_call:"tool_3" param{value:"3"}}')
        rendered = self.slots.to_request_dict()["history"]
        self.assertNotIn('"name":"tool_0"', rendered)
        self.assertIn('"name":"tool_3"', rendered)

    def test_new_task_preserves_history_and_clears_attempt(self):
        first_task_id = self.slots.task_id
        self.slots.commit(
            command_type="tool_call",
            name="get_time",
            params={},
            model_output='{tool_call:"get_time" param{}}',
            result="10:00",
        )
        self.slots.fail(category="tool_failure", error="temporary failure")
        self.slots.start_task(system="stable system", user_input="现在去药房")

        self.assertNotEqual(first_task_id, self.slots.task_id)
        self.assertEqual(len(self.slots.history.entries), 1)
        self.assertEqual(self.slots.history.entries[0]["task_id"], first_task_id)
        self.assertEqual(self.slots.attempt, [])

    def test_history_backend_uses_contract_order(self):
        request = self.slots.to_request_dict()
        messages = _messages_from_prompt_slots(request)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        text = messages[1]["content"]
        self.assertLess(text.index("<user>"), text.index("<history>"))
        self.assertLess(text.index("<history>"), text.index("<attempt>"))

    def test_plan_moves_from_history_directive_into_user(self):
        first = self.slots.to_request_dict()
        self.assertIn('kind="planning"', first["history"])
        self.assertNotIn("<plan>", first["user"])
        self.slots.set_plan({
            "goal": "前往药房",
            "steps": ["启动导航", "报告结果"],
            "done_when": "用户收到执行结果",
        })
        second = self.slots.to_request_dict()
        self.assertNotIn('kind="planning"', second["history"])
        self.assertIn("<plan>", second["user"])
        self.assertEqual(len(self.slots.history.entries), 0)

    def test_json_key_order_does_not_change_slot_order(self):
        request = self.slots.to_request_dict()
        reversed_request = dict(reversed(list(request.items())))
        self.assertEqual(
            _messages_from_prompt_slots(request),
            _messages_from_prompt_slots(reversed_request),
        )


if __name__ == "__main__":
    unittest.main()
