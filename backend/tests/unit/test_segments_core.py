from __future__ import annotations

from dataclasses import dataclass

from furigana_aid.inference.hybrid import (
    CONTEXT_ENSEMBLE,
    MECAB_FALLBACK,
    PLAIN_TEXT,
    HybridDecision,
)
from furigana_aid.inference.segments import (
    MorphToken,
    build_furigana_segments,
    greedy_longest_known_spans,
    plain_text_from_segments,
    tokenize_with_offsets,
)


@dataclass
class Word:
    surface: str


class FakeTagger:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = surfaces

    def __call__(self, text: str):
        del text
        return [Word(surface) for surface in self.surfaces]


def test_repeated_surface_alignment_is_monotonic() -> None:
    tokens = tokenize_with_offsets("日日", FakeTagger(["日", "日"]))
    calls: list[tuple[str, int, int]] = []

    def predict(sentence: str, surface: str, start: int, end: int):
        assert sentence == "日日"
        calls.append((surface, start, end))
        return HybridDecision(
            reading="にち" if start == 0 else "ひ",
            source=CONTEXT_ENSEMBLE,
            confidence=0.8,
        )

    segments = build_furigana_segments(
        "日日",
        tagger=FakeTagger(["日", "日"]),
        known_surfaces={"日"},
        predict_reading=predict,
    )

    assert [(token.start, token.end) for token in tokens] == [(0, 1), (1, 2)]
    assert calls == [("日", 0, 1), ("日", 1, 2)]
    assert [segment.reading for segment in segments] == ["にち", "ひ"]
    assert plain_text_from_segments(segments) == "日日"


def test_greedy_merge_selects_longest_known_span_but_never_five_tokens() -> None:
    sentence = "ABCDE"
    tokens = tuple(
        MorphToken(character, index, index + 1)
        for index, character in enumerate(sentence)
    )
    spans = greedy_longest_known_spans(
        sentence,
        tokens,
        {"AB", "ABC", "ABCD", "ABCDE"},
        max_merge_tokens=4,
    )

    assert [(span.surface, span.start, span.end) for span in spans] == [
        ("ABCD", 0, 4),
        ("E", 4, 5),
    ]


def test_structured_segments_preserve_gaps_offsets_and_prediction_metadata() -> None:
    sentence = "私 は京都!"
    calls: list[tuple[str, int, int]] = []

    def predict(_sentence: str, surface: str, start: int, end: int):
        calls.append((surface, start, end))
        readings = {"私": "わたし", "京都": "きょうと"}
        return HybridDecision(
            reading=readings[surface],
            source=(
                CONTEXT_ENSEMBLE if surface == "私" else MECAB_FALLBACK
            ),
            confidence=0.91 if surface == "私" else None,
            candidate_label_ids=(4, 7) if surface == "私" else (),
        )

    segments = build_furigana_segments(
        sentence,
        tagger=FakeTagger(["私", "は", "京都"]),
        known_surfaces={"私", "京都"},
        predict_reading=predict,
    )

    assert plain_text_from_segments(segments) == sentence
    assert calls == [("私", 0, 1), ("京都", 3, 5)]
    assert [(segment.start, segment.end) for segment in segments] == [
        (0, 1),
        (1, 3),
        (3, 5),
        (5, 6),
    ]
    assert segments[0].kind == "ruby"
    assert segments[0].candidate_label_ids == (4, 7)
    assert segments[0].confidence == 0.91
    assert segments[2].source == MECAB_FALLBACK


def test_unusable_mecab_reading_becomes_plain_text_not_fake_ruby() -> None:
    segments = build_furigana_segments(
        "未知",
        tagger=FakeTagger(["未知"]),
        known_surfaces=set(),
        predict_reading=lambda sentence, surface, start, end: HybridDecision(
            reading=None,
            source=PLAIN_TEXT,
            confidence=None,
        ),
    )

    assert len(segments) == 1
    assert segments[0].kind == "text"
    assert segments[0].source == PLAIN_TEXT
    assert segments[0].reading is None
    assert plain_text_from_segments(segments) == "未知"
