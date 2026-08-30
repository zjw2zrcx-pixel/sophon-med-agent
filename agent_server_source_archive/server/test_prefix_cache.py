from __future__ import annotations

import unittest

from prefix_cache import ExactPrefixCacheManager, FullPrefillManager
from agents.API.session import PromptSlots


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False):
        del tokenize, add_generation_prompt, enable_thinking
        return (
            "<system>" + messages[0]["content"] + "</system>"
            + "<user>" + messages[1]["content"] + "</user><assistant>"
        )

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


class FakeModel:
    def __init__(self):
        self.SEQLEN = 10000
        self.history_length = 0
        self.handles = {}
        self.next_handle = 1
        self.pending = 7

    def clear_history(self):
        self.history_length = 0

    def estimate_snapshot_bytes(self):
        return 100

    def save_snapshot(self):
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = (self.history_length, self.pending)
        return handle

    def restore_snapshot(self, handle):
        self.history_length, self.pending = self.handles[handle]
        return self.pending

    def release_snapshot(self, handle):
        self.handles.pop(handle, None)

    def snapshot_bytes(self, handle):
        return 100 if handle in self.handles else 0

    def total_snapshot_bytes(self):
        return len(self.handles) * 100


class FakePipeline:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()

    def prefill_token_ids(self, token_ids, start_position):
        assert self.model.history_length in (0, start_position + 1)
        self.model.history_length = start_position + len(token_ids) + 1
        self.model.pending += 1
        return self.model.pending

    def generate_from_pending(self, pending_token, prompt_token_count):
        del pending_token, prompt_token_count
        yield "ok"


def slots(system="sys", user="user", history="[]", attempt="[]", version="suha.v1"):
    return {
        "version": version,
        "system": system,
        "user": user,
        "history": history,
        "attempt": attempt,
    }


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = FakePipeline()
        self.cache = ExactPrefixCacheManager(
            self.pipeline, max_sessions=2, max_snapshot_bytes=10_000)

    def run_request(self, session_id, value):
        self.assertEqual(list(self.cache.generate(session_id, value)), ["ok"])

    def test_four_slot_rebuild_levels(self):
        self.run_request("one", slots())
        self.assertEqual(self.cache.cold_misses, 1)
        self.run_request("one", slots())
        self.assertEqual(self.cache.hits["A"], 1)
        self.run_request("one", slots(attempt='[{"failed":1}]'))
        self.assertEqual(self.cache.hits["H"], 1)
        self.run_request("one", slots(history='[{"ok":1}]'))
        self.assertEqual(self.cache.hits["U"], 1)
        self.run_request("one", slots(user="new user"))
        self.assertEqual(self.cache.hits["S"], 1)
        self.run_request("one", slots(system="new sys"))
        self.assertEqual(self.cache.cold_misses, 2)

    def test_session_limit_evicts_least_recently_inferred(self):
        self.run_request("one", slots(user="one"))
        self.run_request("two", slots(user="two"))
        self.run_request("one", slots(user="one"))
        self.run_request("three", slots(user="three"))
        self.assertIn("one", self.cache.sessions)
        self.assertIn("three", self.cache.sessions)
        self.assertNotIn("two", self.cache.sessions)

    def test_image_is_rejected(self):
        value = slots()
        value["image"] = "data:image/png;base64,AA=="
        with self.assertRaisesRegex(ValueError, "text-only"):
            list(self.cache.generate("one", value))

    def test_oversized_prompt_is_rejected_before_native_prefill(self):
        value = slots()
        from prefix_cache import prepare_prompt
        prepared = prepare_prompt(self.pipeline.tokenizer, value)
        self.pipeline.model.SEQLEN = len(prepared.token_ids) + 1
        with self.assertRaisesRegex(ValueError, "safe maximum"):
            list(self.cache.generate("too-long", value))
        self.assertEqual(self.pipeline.model.history_length, 0)
        self.assertNotIn("too-long", self.cache.sessions)

    def test_v2_xml_keeps_same_checkpoint_semantics(self):
        base = slots(version="suha.v2")
        self.run_request("xml", base)
        changed = slots(version="suha.v2", history='[{"ok":1}]')
        self.run_request("xml", changed)
        self.assertEqual(self.cache.hits["U"], 1)
        changed["attempt"] = '[{"failed":1}]'
        self.run_request("xml", changed)
        self.assertEqual(self.cache.hits["H"], 1)

    def test_plan_and_state_growth_reuse_append_only_slot_prefixes(self):
        prompt = PromptSlots()
        prompt.start_task(system="sys", user_input="task")
        self.run_request("plan", prompt.to_request_dict())
        prompt.set_plan({"goal": "g", "steps": ["s"], "done_when": "d"})
        self.run_request("plan", prompt.to_request_dict())
        self.assertEqual(self.cache.hits["U"], 1)
        self.assertEqual(self.cache.last_diagnostics["cache_match_mode"], "partial")
        prompt.commit(
            command_type="tool_call", name="lookup", params={"q": "x"},
            model_output="raw", result="ok",
        )
        prompt.append_execution_event({
            "index": 1, "type": "ACTION_APPLIED",
            "state": {"status": "RUNNING"},
        })
        self.run_request("plan", prompt.to_request_dict())
        self.assertEqual(self.cache.hits["H"], 1)
        self.assertEqual(self.cache.last_diagnostics["cache_match_mode"], "partial")
        prompt.fail(category="tool_failure", error="retry")
        self.run_request("plan", prompt.to_request_dict())
        self.assertEqual(self.cache.hits["H"], 2)
        self.assertEqual(self.cache.last_diagnostics["cache_match_mode"], "exact")

    def test_full_prefill_manager_never_reuses_or_snapshots(self):
        manager = FullPrefillManager(self.pipeline)
        request = slots()
        self.assertEqual(list(manager.generate("one", request)), ["ok"])
        first_tokens = manager.last_diagnostics["logical_prompt_tokens"]
        self.assertEqual(list(manager.generate("one", request)), ["ok"])
        diag = manager.last_diagnostics
        self.assertFalse(diag["prefix_cache_enabled"])
        self.assertFalse(diag["cache_hit"])
        self.assertEqual(diag["cache_event"], "disabled_full_prefill")
        self.assertEqual(diag["reused_prefix_tokens"], 0)
        self.assertEqual(diag["physical_prefill_tokens"], first_tokens)
        self.assertEqual(diag["logical_prompt_tokens"], first_tokens)
        self.assertEqual(diag["snapshot_count"], 0)
        self.assertEqual(diag["snapshot_bytes"], 0)
        self.assertEqual(manager.stats()["requests"], 2)

    def test_thinking_mode_is_reported_without_changing_cache_semantics(self):
        manager = ExactPrefixCacheManager(
            self.pipeline, max_sessions=2, max_snapshot_bytes=10_000,
            enable_thinking=True)
        self.assertEqual(list(manager.generate("think", slots())), ["ok"])
        self.assertTrue(manager.last_diagnostics["thinking_enabled"])
        self.assertFalse(manager.last_diagnostics["cache_hit"])

    def test_v3_new_user_reuses_append_only_conversation_prefix(self):
        prompt = PromptSlots()
        prompt.start_task(system="sys", user_input="first")
        self.run_request("conversation", prompt.to_request_dict())

        prompt.start_task(
            system="sys", user_input="second",
            conversation_context="第1轮\n用户: first\n助手: answer",
        )
        self.run_request("conversation", prompt.to_request_dict())
        self.assertEqual(self.cache.hits["C"], 1)
        self.assertEqual(self.cache.last_diagnostics["cache_match_mode"], "partial")
        self.assertGreater(self.cache.last_diagnostics["reused_prefix_tokens"], 0)

    def test_v3_conversation_is_not_mixed_into_current_user(self):
        prompt = PromptSlots()
        prompt.start_task(
            system="sys", user_input="current",
            conversation_context="用户: previous\n助手: previous answer",
        )
        request = prompt.to_request_dict()
        self.assertEqual(request["version"], "suha.v3")
        self.assertIn("previous", request["conversation"])
        self.assertNotIn("previous", request["user"])
        self.assertIn("current", request["user"])

    def test_v3_bounded_conversation_replacement_safely_retreats_to_system(self):
        prompt = PromptSlots()
        prompt.start_task(
            system="sys", user_input="second",
            conversation_context="第1轮\n用户: first\n助手: answer",
        )
        self.run_request("bounded", prompt.to_request_dict())
        prompt.start_task(
            system="sys", user_input="fourth",
            # Whole-turn eviction changed the beginning, so C is not reusable.
            conversation_context="第2轮\n用户: second\n助手: answer",
        )
        self.run_request("bounded", prompt.to_request_dict())
        self.assertEqual(self.cache.last_diagnostics["matched_checkpoint"], "S")
        self.assertEqual(self.cache.last_diagnostics["cache_match_mode"], "exact")


if __name__ == "__main__":
    unittest.main()
