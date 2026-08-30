#!/usr/bin/env python3
"""Regression checks for medical abbreviation pronunciation handling."""
from pathlib import Path

from melo_zh_lexicon_frontend import MeloZhLexiconFrontend


ROOT = Path(__file__).parent
frontend = MeloZhLexiconFrontend(ROOT / "preprocess_assets")


def test_unknown_initialism_is_spelled() -> None:
    x, tones = frontend.convert("VITS")
    assert len(x) == len(tones)
    assert len(x) > 1


def test_medical_abbreviation_expands() -> None:
    expanded_x, expanded_tones = frontend.convert("磁共振成像")
    abbreviation_x, abbreviation_tones = frontend.convert("MRI")
    assert abbreviation_x.tolist() == expanded_x.tolist()
    assert abbreviation_tones.tolist() == expanded_tones.tolist()


def test_unmapped_lowercase_word_still_fails() -> None:
    try:
        frontend.convert("notaword")
    except ValueError:
        return
    raise AssertionError("unknown lowercase words must not silently be spelled")


if __name__ == "__main__":
    test_unknown_initialism_is_spelled()
    test_medical_abbreviation_expands()
    test_unmapped_lowercase_word_still_fails()
    print("medical frontend checks passed")
