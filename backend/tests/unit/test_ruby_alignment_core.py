from __future__ import annotations

import pytest

from furigana_aid.inference.engine import RubySegment, _expand_ruby_segment
from furigana_aid.inference.ruby_alignment import align_reading_to_kanji


def _signature(surface: str, reading: str):
    parts = align_reading_to_kanji(surface, reading)
    assert parts is not None
    return [(part.kind, part.text, part.reading) for part in parts]


@pytest.mark.parametrize(
    ("surface", "reading", "expected"),
    [
        (
            "合わせよう",
            "あわせよう",
            [("ruby", "合", "あ"), ("text", "わせよう", None)],
        ),
        (
            "間違い",
            "まちがい",
            [("ruby", "間違", "まちが"), ("text", "い", None)],
        ),
        (
            "お祝い",
            "おいわい",
            [
                ("text", "お", None),
                ("ruby", "祝", "いわ"),
                ("text", "い", None),
            ],
        ),
        (
            "申し込む",
            "もうしこむ",
            [
                ("ruby", "申", "もう"),
                ("text", "し", None),
                ("ruby", "込", "こ"),
                ("text", "む", None),
            ],
        ),
        ("今日", "きょう", [("ruby", "今日", "きょう")]),
        (
            "1人",
            "ひとり",
            [("text", "1", None), ("ruby", "人", "ひとり")],
        ),
    ],
)
def test_reading_is_aligned_only_to_kanji_runs(
    surface: str,
    reading: str,
    expected: list[tuple[str, str, str | None]],
) -> None:
    assert _signature(surface, reading) == expected


def test_ambiguous_unanchored_kanji_runs_are_not_guessed() -> None:
    assert align_reading_to_kanji("第1章", "だいいっしょう") is None


def test_engine_expansion_preserves_prediction_metadata() -> None:
    original = RubySegment(
        type="ruby",
        base="合わせよう",
        reading="あわせよう",
        source="ContextEnsemble",
        confidence=0.94,
        edited=False,
    )

    expanded = _expand_ruby_segment(original)

    assert len(expanded) == 2
    assert expanded[0] == RubySegment(
        type="ruby",
        base="合",
        reading="あ",
        source="ContextEnsemble",
        confidence=0.94,
        edited=False,
    )
    assert expanded[1].type == "text"
    assert expanded[1].text == "わせよう"
