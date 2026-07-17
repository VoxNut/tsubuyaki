"""Align a word reading to kanji runs while leaving okurigana plain."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


KANJI_CHAR_RE = re.compile(r"[一-龠々〆ヶ]")
KANA_CHAR_RE = re.compile(r"[ぁ-ゖゝゞァ-ヺヽヾー]")


@dataclass(frozen=True, slots=True)
class RubyAlignmentPart:
    kind: Literal["text", "ruby"]
    text: str
    reading: str | None = None


def _katakana_to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in text
    )


def _character_kind(character: str) -> Literal["kanji", "kana", "other"]:
    if KANJI_CHAR_RE.fullmatch(character):
        return "kanji"
    if KANA_CHAR_RE.fullmatch(character):
        return "kana"
    return "other"


def _surface_runs(
    surface: str,
) -> list[tuple[Literal["kanji", "kana", "other"], str]]:
    runs: list[tuple[Literal["kanji", "kana", "other"], str]] = []
    for character in surface:
        kind = _character_kind(character)
        if runs and runs[-1][0] == kind:
            previous_kind, previous_text = runs[-1]
            runs[-1] = (previous_kind, previous_text + character)
        else:
            runs.append((kind, character))
    return runs


def align_reading_to_kanji(
    surface: str,
    reading: str,
) -> tuple[RubyAlignmentPart, ...] | None:
    """Return kanji-only ruby parts, or ``None`` when alignment is ambiguous.

    Kana runs in ``surface`` act as anchors in ``reading``. Reading text before
    each anchor belongs to the preceding kanji run. This handles ordinary
    okurigana and mixed forms such as ``合わせよう`` and ``申し込む`` without
    guessing how a reading should be divided between unanchored kanji runs.
    """

    if not surface or not reading or not KANJI_CHAR_RE.search(surface):
        return None

    normalized_reading = _katakana_to_hiragana(reading)
    parts: list[RubyAlignmentPart] = []
    pending_kanji_indices: list[int] = []
    reading_cursor = 0

    for kind, text in _surface_runs(surface):
        if kind == "kanji":
            parts.append(RubyAlignmentPart(kind="ruby", text=text))
            pending_kanji_indices.append(len(parts) - 1)
            continue

        if kind == "other":
            parts.append(RubyAlignmentPart(kind="text", text=text))
            continue

        anchor = _katakana_to_hiragana(text)
        search_from = reading_cursor + (1 if pending_kanji_indices else 0)
        anchor_start = normalized_reading.find(anchor, search_from)
        if anchor_start < 0:
            return None

        reading_before_anchor = reading[reading_cursor:anchor_start]
        if pending_kanji_indices:
            if len(pending_kanji_indices) != 1 or not reading_before_anchor:
                return None
            part_index = pending_kanji_indices.pop()
            pending = parts[part_index]
            parts[part_index] = RubyAlignmentPart(
                kind="ruby",
                text=pending.text,
                reading=reading_before_anchor,
            )
        elif reading_before_anchor:
            return None

        parts.append(RubyAlignmentPart(kind="text", text=text))
        reading_cursor = anchor_start + len(anchor)

    remaining_reading = reading[reading_cursor:]
    if pending_kanji_indices:
        if len(pending_kanji_indices) != 1 or not remaining_reading:
            return None
        part_index = pending_kanji_indices.pop()
        pending = parts[part_index]
        parts[part_index] = RubyAlignmentPart(
            kind="ruby",
            text=pending.text,
            reading=remaining_reading,
        )
    elif remaining_reading:
        return None

    if any(part.kind == "ruby" and not part.reading for part in parts):
        return None
    return tuple(parts)
