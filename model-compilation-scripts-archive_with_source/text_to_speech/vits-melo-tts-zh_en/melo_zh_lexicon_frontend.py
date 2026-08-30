#!/usr/bin/env python3
"""Minimal frontend matching this sherpa-onnx Melo zh_en export's lexicon contract."""
from pathlib import Path
import json
import re
import numpy as np

# The model only has comma/full-stop style pause tokens.  Treat semicolons and
# colons as short pauses so clinical instructions can use normal punctuation.
PUNCT = {"，": ",", "。": ".", "！": "!", "？": "?", "；": ",", "：": ",", ",": ",", ".": ".", "!": "!", "?": "?", ";": ",", ":": ","}


class MeloZhLexiconFrontend:
    def __init__(self, assets_dir: str | Path) -> None:
        assets = Path(assets_dir)
        self.token_id = {line.rsplit(" ", 1)[0]: int(line.rsplit(" ", 1)[1])
                         for line in (assets / "tokens.txt").read_text().splitlines() if line}
        self.lexicon = {}
        for line in (assets / "lexicon.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            word = fields[0]
            half = (len(fields) - 1) // 2
            phones, tones = fields[1:1 + half], fields[1 + half:]
            if len(phones) == len(tones) and all(p in self.token_id for p in phones):
                self.lexicon.setdefault(word, (phones, [int(t) for t in tones]))
        medical_path = assets / "medical_abbreviations.json"
        self.medical_abbreviations = {}
        if medical_path.exists():
            entries = json.loads(medical_path.read_text(encoding="utf-8"))
            if not isinstance(entries, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in entries.items()
            ):
                raise ValueError(f"Invalid medical abbreviation file: {medical_path}")
            self.medical_abbreviations = {
                key.upper(): value for key, value in entries.items()
            }

    def _append_lexicon_item(self, item: str, phones: list[str], tones: list[int]) -> None:
        """Append one lexicon item, including a Chinese expansion if configured."""
        if item in PUNCT.values():
            phones.append(item)
            tones.append(0)
            return

        # Medical abbreviations are deliberately expanded to Chinese clinical
        # names.  This is preferable to guessing an English word pronunciation.
        expansion = self.medical_abbreviations.get(item.upper())
        if expansion is not None:
            for part in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", expansion):
                if not part.isspace():
                    self._append_lexicon_item(part, phones, tones)
            return

        lookup = item.lower() if item.isascii() else item
        if lookup in self.lexicon:
            p, t = self.lexicon[lookup]
            phones.extend(p)
            tones.extend(t)
            return

        # Keep an unknown initialism speakable.  The base lexicon includes the
        # letter names a-z, so "VITS" becomes "V I T S" rather than rejecting
        # the entire request.  Lowercase words still fail loudly: silently
        # spelling arbitrary vocabulary is usually a pronunciation error.
        if item.isascii() and item.isalpha() and item.isupper():
            for letter in item.lower():
                p, t = self.lexicon[letter]
                phones.extend(p)
                tones.extend(t)
            return

        raise ValueError(f"No lexicon entry for {item!r}")

    def convert(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return unpadded IDs with Melo add_blank=1 applied."""
        phones: list[str] = []
        tones: list[int] = []
        # Chinese is intentionally processed character-by-character, matching the
        # supplied lexicon entries.  Preserve case so unknown initialisms can be
        # distinguished from ordinary lowercase English words.
        for part in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", text):
            if part.isspace():
                continue
            item = PUNCT.get(part, part)
            self._append_lexicon_item(item, phones, tones)
        if not phones:
            raise ValueError("text produced no phones")
        x = [0]
        y = [0]
        for phone, tone in zip(phones, tones):
            x.extend((self.token_id[phone], 0))
            y.extend((tone, 0))
        return np.asarray(x, np.int64), np.asarray(y, np.int64)
