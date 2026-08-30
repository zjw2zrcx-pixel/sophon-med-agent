from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

import httpx


SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER_DIR))

from agents.API.api import API  # noqa: E402
from agents.API.session import Session  # noqa: E402
from agents.API.trajectory import summarize_token_usage  # noqa: E402
import qwen3_5_history_server as history_server  # noqa: E402
import qwen3_5_server as standard_server  # noqa: E402


class FakePipeline:
    def __init__(self, prompt_tokens=0, completion_tokens=0, chunks=None):
        self.last_input_token_count = prompt_tokens
        self.last_output_token_count = completion_tokens
        self.chunks = chunks or []

    def generate(self, *args, **kwargs):
        del args, kwargs
        yield from self.chunks


class TokenUsageTests(unittest.TestCase):
    def test_student_context_overflow_summary_uses_8192_limit(self):
        api = API(context_window_tokens=8192)
        stats = api._context_stats({
            "prompt_tokens": 8190, "completion_tokens": 5, "total_tokens": 8195,
        }, "完成啦")
        self.assertTrue(stats["context_overflow"])
        self.assertTrue(stats["provider_total_overflow"])
        self.assertGreater(stats["overflow_tokens"], 0)
        summary = summarize_token_usage([{
            "usage": {"prompt_tokens": 8190, "completion_tokens": 5,
                      "total_tokens": 8195},
            "context_stats": stats,
        }], 8192)
        self.assertTrue(summary["context_overflow"])
        self.assertEqual(summary["context_overflow_calls"], 1)
        self.assertEqual(summary["peak_prompt_tokens"], 8190)

    def test_hidden_reasoning_overflow_is_not_training_sequence_overflow(self):
        api = API(context_window_tokens=8192)
        stats = api._context_stats({
            "prompt_tokens": 4000, "completion_tokens": 6000,
            "total_tokens": 10000,
        }, '<tool>{"tool_call":"act"}</tool>')
        self.assertTrue(stats["provider_total_overflow"])
        self.assertFalse(stats["context_overflow"])

    def test_usage_helpers_return_openai_fields(self):
        for module in (standard_server, history_server):
            original = module.pipeline
            try:
                module.pipeline = FakePipeline(123, 17)
                self.assertEqual(module._current_usage(), {
                    "prompt_tokens": 123,
                    "completion_tokens": 17,
                    "total_tokens": 140,
                })
            finally:
                module.pipeline = original

    def test_non_stream_collection_returns_text_and_usage(self):
        original = standard_server.pipeline
        try:
            standard_server.pipeline = FakePipeline(21, 2, ["你", "好"])
            text, usage = standard_server._collect_generate([], "text", False)
        finally:
            standard_server.pipeline = original

        self.assertEqual(text, "你好")
        self.assertEqual(usage, {
            "prompt_tokens": 21,
            "completion_tokens": 2,
            "total_tokens": 23,
        })

    def test_agent_uses_non_stream_request_and_retains_usage(self):
        observed = {}

        async def handler(request):
            observed.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "完成"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 31,
                    "completion_tokens": 4,
                    "total_tokens": 35,
                },
            })

        async def run_request():
            api = API(server_url="http://test", default_model="test-model")
            api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            session = Session()
            session.prompt_slots.start_task(
                system="fixed", user_input="执行任务")
            try:
                text = await api.chat_complete(session)
                return text, session.last_usage, session.model_call_records[-1]
            finally:
                await api.close()

        text, usage, record = asyncio.run(run_request())
        self.assertIs(observed["stream"], False)
        self.assertEqual(text, "完成")
        self.assertEqual(usage["total_tokens"], 35)
        self.assertEqual(record["context_stats"]["context_window_tokens"], 8192)
        self.assertFalse(record["context_stats"]["context_overflow"])


if __name__ == "__main__":
    unittest.main()
