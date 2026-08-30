from __future__ import annotations

import unittest

from thinking_output import split_thinking_output


class ThinkingOutputTests(unittest.TestCase):
    def test_closed_thinking_is_removed_from_visible_action(self):
        thinking, visible, closed = split_thinking_output(
            "分析步骤</think>\n\n<tool>{}</tool>", True)
        self.assertEqual(thinking, "分析步骤")
        self.assertEqual(visible, "<tool>{}</tool>")
        self.assertTrue(closed)

    def test_truncated_thinking_has_no_visible_action(self):
        thinking, visible, closed = split_thinking_output("仍在分析", True)
        self.assertEqual(thinking, "仍在分析")
        self.assertEqual(visible, "")
        self.assertFalse(closed)

    def test_disabled_mode_is_unchanged(self):
        value = "<tool>{}</tool>"
        self.assertEqual(
            split_thinking_output(value, False), ("", value, False))


if __name__ == "__main__":
    unittest.main()
