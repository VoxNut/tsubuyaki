"""End-to-end cue planning, fallback routing, and batched inference."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from fugashi import Tagger

from furigana_aid.inference.hybrid import (
    HybridDecision,
    HybridTuning,
    contextual_decision,
    non_contextual_decision,
)
from furigana_aid.inference.ruby_alignment import align_reading_to_kanji
from furigana_aid.neural import (
    NeuralModels,
    build_target_centered_ids,
    predict_token_batches,
)


KANJI_RE = re.compile(r"[一-龠々〆ヶ]")
KANA_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+$")
TAG_RE = re.compile(r"<[^>]+>")


class ArtifactBundle(Protocol):
    surface2idx: Mapping[str, int]
    surface_mode_readings: Mapping[str, str]
    id2label: Mapping[int, str]

    def candidates_for(
        self,
        surface: str,
    ) -> tuple[Sequence[int], Sequence[float], int]:
        """Return candidate IDs, priors, and contextual mode label ID."""


@dataclass(frozen=True, slots=True)
class Cue:
    cue_id: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class TextSegment:
    type: str
    text: str


@dataclass(frozen=True, slots=True)
class RubySegment:
    type: str
    base: str
    reading: str
    source: str
    confidence: float | None
    edited: bool = False


Segment = TextSegment | RubySegment


@dataclass(frozen=True, slots=True)
class _PendingContextSegment:
    base: str


InternalSegment = Segment | _PendingContextSegment


@dataclass(frozen=True, slots=True)
class CueResult:
    cue_id: str
    start_ms: int
    end_ms: int
    plain_text: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class TokenSpan:
    surface: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ContextTarget:
    cue_index: int
    segment_index: int
    sentence: str
    surface: str
    start: int
    end: int


def clean_html_text(value: str | None) -> str:
    cleaned = html.unescape(TAG_RE.sub("", value or ""))
    return unicodedata.normalize("NFKC", cleaned).strip()


def katakana_to_hiragana(text: str | None) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in (text or "")
    )


def normalize_reading(text: str | None) -> str:
    normalized = re.sub(r"\s+", "", clean_html_text(text))
    return katakana_to_hiragana(normalized)


def _coalesce_text(segments: list[InternalSegment], text: str) -> None:
    if not text:
        return
    if segments and isinstance(segments[-1], TextSegment):
        previous = segments[-1]
        segments[-1] = TextSegment(
            type="text",
            text=previous.text + text,
        )
    else:
        segments.append(TextSegment(type="text", text=text))


def _append_final_segment(segments: list[Segment], segment: Segment) -> None:
    if isinstance(segment, TextSegment):
        if not segment.text:
            return
        if segments and isinstance(segments[-1], TextSegment):
            previous = segments[-1]
            segments[-1] = TextSegment(
                type="text",
                text=previous.text + segment.text,
            )
            return
    segments.append(segment)


def _expand_ruby_segment(segment: RubySegment) -> tuple[Segment, ...]:
    aligned_parts = align_reading_to_kanji(segment.base, segment.reading)
    if aligned_parts is None:
        return (segment,)

    expanded: list[Segment] = []
    for part in aligned_parts:
        if part.kind == "text":
            expanded.append(TextSegment(type="text", text=part.text))
        else:
            expanded.append(
                RubySegment(
                    type="ruby",
                    base=part.text,
                    reading=part.reading or "",
                    source=segment.source,
                    confidence=segment.confidence,
                    edited=segment.edited,
                )
            )
    return tuple(expanded)


class FuriganaEngine:
    def __init__(
        self,
        *,
        artifacts: ArtifactBundle,
        neural: NeuralModels,
        tuning: HybridTuning,
        max_merge_tokens: int,
        tagger: Tagger | None = None,
    ) -> None:
        if max_merge_tokens <= 0:
            raise ValueError("max_merge_tokens phải dương.")
        self.artifacts = artifacts
        self.neural = neural
        self.tuning = tuning
        self.max_merge_tokens = max_merge_tokens
        self.tagger = tagger or Tagger()
        self.known_surfaces = set(artifacts.surface_mode_readings)
        self.known_surfaces.update(artifacts.surface2idx)

    def _tokenize_with_offsets(self, sentence: str) -> list[TokenSpan]:
        spans: list[TokenSpan] = []
        search_from = 0
        for word in self.tagger(sentence):
            surface = str(word.surface)
            if not surface:
                continue
            start = sentence.find(surface, search_from)
            if start < 0:
                continue
            end = start + len(surface)
            spans.append(TokenSpan(surface=surface, start=start, end=end))
            search_from = end
        return spans

    def _merged_spans(self, sentence: str) -> list[TokenSpan]:
        tokens = self._tokenize_with_offsets(sentence)
        merged: list[TokenSpan] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            best_next = index + 1
            best_end = token.end
            previous_end = token.end
            upper = min(
                len(tokens),
                index + self.max_merge_tokens,
            )
            for next_index in range(index + 1, upper):
                next_token = tokens[next_index]
                if next_token.start != previous_end:
                    break
                previous_end = next_token.end
                candidate = sentence[token.start : next_token.end]
                if candidate in self.known_surfaces:
                    best_next = next_index + 1
                    best_end = next_token.end
            merged.append(
                TokenSpan(
                    surface=sentence[token.start:best_end],
                    start=token.start,
                    end=best_end,
                )
            )
            index = best_next
        return merged

    def _mecab_reading(self, surface: str) -> str | None:
        pieces: list[str] = []
        try:
            for word in self.tagger(surface):
                feature = word.feature
                reading = (
                    getattr(feature, "kana", None)
                    or getattr(feature, "pron", None)
                    or getattr(feature, "reading", None)
                )
                if reading and reading != "*":
                    pieces.append(normalize_reading(str(reading)))
                else:
                    pieces.append(str(word.surface))
        except Exception:
            return None
        combined = "".join(pieces)
        if not combined or not KANA_RE.fullmatch(combined):
            return None
        return combined

    @staticmethod
    def _ruby(surface: str, decision: HybridDecision) -> Segment:
        if not decision.reading:
            return TextSegment(type="text", text=surface)
        return RubySegment(
            type="ruby",
            base=surface,
            reading=decision.reading,
            source=decision.source,
            confidence=decision.confidence,
            edited=False,
        )

    def _plan_cue(
        self,
        cue: Cue,
        cue_index: int,
    ) -> tuple[list[InternalSegment], list[ContextTarget]]:
        segments: list[InternalSegment] = []
        targets: list[ContextTarget] = []
        cursor = 0
        for span in self._merged_spans(cue.text):
            _coalesce_text(segments, cue.text[cursor:span.start])
            if not KANJI_RE.search(span.surface):
                _coalesce_text(segments, span.surface)
            elif span.surface in self.artifacts.surface2idx:
                segment_index = len(segments)
                segments.append(_PendingContextSegment(base=span.surface))
                targets.append(
                    ContextTarget(
                        cue_index=cue_index,
                        segment_index=segment_index,
                        sentence=cue.text,
                        surface=span.surface,
                        start=span.start,
                        end=span.end,
                    )
                )
            else:
                decision = non_contextual_decision(
                    surface=span.surface,
                    surface_mode_readings=(
                        self.artifacts.surface_mode_readings
                    ),
                    mecab_reading=self._mecab_reading(span.surface),
                )
                predicted = self._ruby(span.surface, decision)
                if isinstance(predicted, TextSegment):
                    _coalesce_text(segments, predicted.text)
                else:
                    segments.append(predicted)
            cursor = span.end
        _coalesce_text(segments, cue.text[cursor:])
        if not segments and cue.text:
            segments.append(TextSegment(type="text", text=cue.text))
        return segments, targets

    def _controlled_target_fallback(
        self,
        target: ContextTarget,
    ) -> Segment:
        decision = non_contextual_decision(
            surface=target.surface,
            surface_mode_readings=self.artifacts.surface_mode_readings,
            mecab_reading=self._mecab_reading(target.surface),
        )
        return self._ruby(target.surface, decision)

    def generate_batch(self, cues: Sequence[Cue]) -> list[CueResult]:
        mutable_segments: list[list[InternalSegment]] = []
        all_targets: list[ContextTarget] = []
        for cue_index, cue in enumerate(cues):
            segments, targets = self._plan_cue(cue, cue_index)
            mutable_segments.append(segments)
            all_targets.extend(targets)

        valid_targets: list[ContextTarget] = []
        input_rows: list[list[int]] = []
        for target in all_targets:
            try:
                input_ids = build_target_centered_ids(
                    self.neural.tokenizer,
                    sentence=target.sentence,
                    start=target.start,
                    end=target.end,
                    target_start_id=self.neural.target_start_id,
                    target_end_id=self.neural.target_end_id,
                    max_length=self.neural.max_length,
                )
            except ValueError:
                mutable_segments[target.cue_index][
                    target.segment_index
                ] = self._controlled_target_fallback(target)
                continue
            valid_targets.append(target)
            input_rows.append(input_ids)

        neural_outputs = predict_token_batches(self.neural, input_rows)
        if len(neural_outputs) != len(valid_targets):
            raise RuntimeError(
                "Số neural output không khớp số contextual target."
            )
        for target, output in zip(valid_targets, neural_outputs):
            decision = contextual_decision(
                surface=target.surface,
                linear_logits=output.linear_logits,
                mlp_logits=output.mlp_logits,
                artifacts=self.artifacts,
                tuning=self.tuning,
            )
            mutable_segments[target.cue_index][
                target.segment_index
            ] = self._ruby(target.surface, decision)

        results: list[CueResult] = []
        for cue, segments in zip(cues, mutable_segments):
            if any(
                isinstance(segment, _PendingContextSegment)
                for segment in segments
            ):
                raise RuntimeError(
                    "Contextual target chưa được thay bằng kết quả."
                )
            final_segments = [
                segment
                for segment in segments
                if not isinstance(segment, _PendingContextSegment)
            ]
            expanded_segments: list[Segment] = []
            for segment in final_segments:
                if isinstance(segment, RubySegment):
                    for expanded in _expand_ruby_segment(segment):
                        _append_final_segment(expanded_segments, expanded)
                else:
                    _append_final_segment(expanded_segments, segment)
            final_segments = expanded_segments
            rebuilt = "".join(
                segment.text
                if isinstance(segment, TextSegment)
                else segment.base
                for segment in final_segments
            )
            if rebuilt != cue.text:
                raise RuntimeError(
                    "Structured segments không bảo toàn cue text: "
                    f"{rebuilt!r} != {cue.text!r}."
                )
            results.append(
                CueResult(
                    cue_id=cue.cue_id,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    plain_text=cue.text,
                    segments=tuple(final_segments),
                )
            )
        return results
