"""Concurrency policy tests for local and online model transports."""
from __future__ import annotations

import unittest

from agents.API.api import API


class APIConcurrencyPolicyTest(unittest.TestCase):
    def test_online_models_use_bounded_parallel_gate(self) -> None:
        api = API(online_max_concurrency=2)
        self.assertIs(api._request_gate("deepseek-v4-flash"), api._online_semaphore)
        self.assertIs(api._request_gate("deepseek-v4-pro"), api._online_semaphore)
        self.assertEqual(api._online_semaphore._value, 2)

    def test_local_models_remain_serialized(self) -> None:
        api = API(online_max_concurrency=4)
        self.assertIs(api._request_gate("qwen3.5-4b"), api._request_lock)
        self.assertIs(api._request_gate("custom-local-model"), api._request_lock)


if __name__ == "__main__":
    unittest.main()
