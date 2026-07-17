"""Offset-safe MeCab token alignment, greedy merge and structured segments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Literal, Protocol, Sequence

from .hybrid import PLAIN_TEXT, PredictionSource
from .ruby_alignment import align_reading_to_kanji


# Kept deliberately identical to the notebook for behavioral parity.
KANJI_RE = re.compile(r"[一-龠々〆ヶ]")


class MorphWord(Protocol):
    surface: str


class MorphologicalTagger(Protocol):
    def __call__(self, text: str) -> Iterable[MorphWord]:
        """Yield words in source order."""


class ReadingPrediction(Protocol):
    reading: str | None
    source: PredictionSource
    confidence: float | None


PredictionCallback = Callable[
    [str, str, int, int],
    ReadingPrediction,
]


@dataclass(frozen=True, slots=True)
class MorphToken:
    surface: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SurfaceSpan:
    surface: str
    start: int
    end: int
    first_token_index: int
    next_token_index: int


@dataclass(frozen=True, slots=True)
class FuriganaSegment:
    """A source-aligned text or ruby span.

    Concatenating ``text`` in order must reproduce the original cue exactly.
    HTML is intentionally absent; the frontend will render these fields with
    DOM APIs.
    """

    kind: Literal["text", "ruby"]
    text: str
    start: int
    end: int
    reading: str | None = None
    source: PredictionSource | None = None
    confidence: float | None = None
    candidate_label_ids: tuple[int, ...] = ()
    manual: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment offsets are invalid.")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                "Segment text length must equal its source offset length."
            )
        if self.kind == "ruby":
            if not self.reading:
                raise ValueError("Ruby segments require a non-empty reading.")
            if self.source is None:
                raise ValueError("Ruby segments require a prediction source.")
        elif self.reading is not None:
            raise ValueError("Plain text segments cannot carry a reading.")


def tokenize_with_offsets(
    sentence: str, tagger: MorphologicalTagger
) -> tuple[MorphToken, ...]:
    """Align tagger surfaces by monotonic search, including repeated words."""

    tokens: list[MorphToken] = []
    search_from = 0
    for word in tagger(sentence):
        surface = str(getattr(word, "surface", ""))
        if not surface:
            continue
        token_start = sentence.find(surface, search_from)
        if token_start < 0:
            # Mirrors the notebook: an unalignable tokenizer item is skipped,
            # while the untouched source gap is preserved as plain text later.
            continue
        token_end = token_start + len(surface)
        tokens.append(
            MorphToken(surface=surface, start=token_start, end=token_end)
        )
        search_from = token_end
    return tuple(tokens)


def greedy_longest_known_spans(
    sentence: str,
    tokens: Sequence[MorphToken],
    known_surfaces: Iterable[str],
    *,
    max_merge_tokens: int = 4,
) -> tuple[SurfaceSpan, ...]:
    """Choose the longest contiguous known surface, up to four tokens."""

    if max_merge_tokens < 1:
        raise ValueError("max_merge_tokens must be at least 1.")
    known = set(known_surfaces)
    previous_end = 0
    for token in tokens:
        if token.start < previous_end or token.end > len(sentence):
            raise ValueError("Morph tokens must be ordered and non-overlapping.")
        if sentence[token.start : token.end] != token.surface:
            raise ValueError("Morph token surface does not match its offsets.")
        previous_end = token.end

    spans: list[SurfaceSpan] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        best_next = index + 1
        best_end = token.end
        contiguous_end = token.end

        # For max_merge_tokens=4, the loop examines totals of 2, 3 and 4.
        for next_index in range(
            index + 1,
            min(len(tokens), index + max_merge_tokens),
        ):
            next_token = tokens[next_index]
            if next_token.start != contiguous_end:
                break
            contiguous_end = next_token.end
            candidate = sentence[token.start:contiguous_end]
            if candidate in known:
                best_next = next_index + 1
                best_end = contiguous_end

        spans.append(
            SurfaceSpan(
                surface=sentence[token.start:best_end],
                start=token.start,
                end=best_end,
                first_token_index=index,
                next_token_index=best_next,
            )
        )
        index = best_next
    return tuple(spans)


def _append_segment(
    output: list[FuriganaSegment], segment: FuriganaSegment
) -> None:
    if not segment.text:
        return
    if (
        output
        and output[-1].kind == "text"
        and segment.kind == "text"
        and output[-1].end == segment.start
        and output[-1].source == segment.source
        and output[-1].manual == segment.manual
    ):
        previous = output.pop()
        output.append(
            FuriganaSegment(
                kind="text",
                text=previous.text + segment.text,
                start=previous.start,
                end=segment.end,
                source=segment.source,
                manual=segment.manual,
            )
        )
        return
    output.append(segment)


def build_furigana_segments(
    sentence: str,
    *,
    tagger: MorphologicalTagger,
    known_surfaces: Iterable[str],
    predict_reading: PredictionCallback,
    max_merge_tokens: int = 4,
) -> tuple[FuriganaSegment, ...]:
    """Generate source-aligned structured segments for one cue."""

    tokens = tokenize_with_offsets(sentence, tagger)
    spans = greedy_longest_known_spans(
        sentence,
        tokens,
        known_surfaces,
        max_merge_tokens=max_merge_tokens,
    )
    output: list[FuriganaSegment] = []
    cursor = 0

    for span in spans:
        if span.start > cursor:
            _append_segment(
                output,
                FuriganaSegment(
                    kind="text",
                    text=sentence[cursor:span.start],
                    start=cursor,
                    end=span.start,
                ),
            )

        if KANJI_RE.search(span.surface):
            prediction = predict_reading(
                sentence, span.surface, span.start, span.end
            )
            reading = prediction.reading
            source = prediction.source
            if reading and reading != span.surface and source != PLAIN_TEXT:
                candidate_ids = tuple(
                    int(value)
                    for value in getattr(
                        prediction, "candidate_label_ids", ()
                    )
                )
                aligned_parts = align_reading_to_kanji(
                    span.surface,
                    str(reading),
                )
                if aligned_parts is None:
                    aligned_parts = ()

                part_cursor = span.start
                for part in aligned_parts:
                    part_end = part_cursor + len(part.text)
                    _append_segment(
                        output,
                        FuriganaSegment(
                            kind=part.kind,
                            text=part.text,
                            start=part_cursor,
                            end=part_end,
                            reading=part.reading,
                            source=(source if part.kind == "ruby" else None),
                            confidence=(
                                prediction.confidence
                                if part.kind == "ruby"
                                else None
                            ),
                            candidate_label_ids=(
                                candidate_ids if part.kind == "ruby" else ()
                            ),
                        ),
                    )
                    part_cursor = part_end

                if not aligned_parts:
                    _append_segment(
                        output,
                        FuriganaSegment(
                            kind="ruby",
                            text=span.surface,
                            start=span.start,
                            end=span.end,
                            reading=str(reading),
                            source=source,
                            confidence=prediction.confidence,
                            candidate_label_ids=candidate_ids,
                        ),
                    )
            else:
                _append_segment(
                    output,
                    FuriganaSegment(
                        kind="text",
                        text=span.surface,
                        start=span.start,
                        end=span.end,
                        source=PLAIN_TEXT,
                    ),
                )
        else:
            _append_segment(
                output,
                FuriganaSegment(
                    kind="text",
                    text=span.surface,
                    start=span.start,
                    end=span.end,
                ),
            )
        cursor = span.end

    if cursor < len(sentence):
        _append_segment(
            output,
            FuriganaSegment(
                kind="text",
                text=sentence[cursor:],
                start=cursor,
                end=len(sentence),
            ),
        )
    validate_segment_reconstruction(sentence, output)
    return tuple(output)


def plain_text_from_segments(segments: Sequence[FuriganaSegment]) -> str:
    return "".join(segment.text for segment in segments)


def validate_segment_reconstruction(
    sentence: str, segments: Sequence[FuriganaSegment]
) -> None:
    """Fail if segments overlap, leave gaps, or alter the source text."""

    cursor = 0
    for index, segment in enumerate(segments):
        if segment.start != cursor:
            raise ValueError(
                f"Segment {index} starts at {segment.start}, expected {cursor}."
            )
        if sentence[segment.start : segment.end] != segment.text:
            raise ValueError(
                f"Segment {index} text does not match the original sentence."
            )
        cursor = segment.end
    if cursor != len(sentence):
        raise ValueError(
            f"Segments cover {cursor} characters, expected {len(sentence)}."
        )
    if plain_text_from_segments(segments) != sentence:
        raise ValueError("Segments do not reconstruct the original sentence.")
