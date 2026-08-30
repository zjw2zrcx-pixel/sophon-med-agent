#!/usr/bin/env python3
"""Regression checks for the fixed-50-token VITS segmentation policy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vits-melo-tts-zh_en"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vits_tts_server as server
from melo_zh_lexicon_frontend import MeloZhLexiconFrontend


server.frontend = MeloZhLexiconFrontend(ROOT / "vits-melo-tts-zh_en" / "preprocess_assets")


def assert_within_window(segments):
    assert segments
    for segment in segments:
        tokens, _ = server.frontend.convert(segment)
        assert len(tokens) <= server.MAX_INPUT_TOKENS, (segment, len(tokens))


def test_prefers_punctuation_and_fills_window():
    text = "你好。请到一楼取报告。随后到门诊复查。"
    segments = server._segments(text)
    assert_within_window(segments)
    assert "".join(segments) == text
    # The first two short clauses fit together; punctuation is not an automatic
    # split point.
    assert segments[0] == "你好。请到一楼取报告。"


def test_short_comma_clauses_stay_in_one_window():
    text = "你好，我是A,欢迎"
    assert server._segments(text) == [text]


def test_long_clause_is_hard_split_only_when_necessary():
    text = "这是一个没有合适标点而且长度足以超过固定输入窗口的长句子，需要在达到上限时才进行安全截断。"
    segments = server._segments(text)
    assert_within_window(segments)
    assert len(segments) > 1
    assert "".join(segments) == text


if __name__ == "__main__":
    test_prefers_punctuation_and_fills_window()
    test_long_clause_is_hard_split_only_when_necessary()
    print("VITS segmentation checks passed")
