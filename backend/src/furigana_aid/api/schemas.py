"""Pydantic contracts shared with the future static frontend."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class PredictionSource(StrEnum):
    CONTEXT_ENSEMBLE = "ContextEnsemble"
    MOST_FREQUENT_LOW_CONFIDENCE = "MostFrequentLowConfidence"
    MOST_FREQUENT = "MostFrequent"
    MECAB_FALLBACK = "MeCabFallback"
    PLAIN_TEXT = "PlainText"
    MANUAL_EDIT = "ManualEdit"


class CueInput(BaseModel):
    cue_id: str = Field(min_length=1, max_length=200)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> "CueInput":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms phải lớn hơn hoặc bằng start_ms.")
        return self


class GenerateBatchRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=200)
    cues: list[CueInput] = Field(min_length=1)


class TextSegment(BaseModel):
    type: Literal["text"] = "text"
    text: str


class RubySegment(BaseModel):
    type: Literal["ruby"] = "ruby"
    base: str
    reading: str
    source: PredictionSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    edited: bool = False


Segment = Annotated[
    TextSegment | RubySegment,
    Field(discriminator="type"),
]


class CueResult(BaseModel):
    cue_id: str
    start_ms: int
    end_ms: int
    plain_text: str
    segments: list[Segment]

    @model_validator(mode="after")
    def validate_segment_round_trip(self) -> "CueResult":
        rebuilt = "".join(
            segment.text
            if isinstance(segment, TextSegment)
            else segment.base
            for segment in self.segments
        )
        if rebuilt != self.plain_text:
            raise ValueError(
                "Segments không ghép lại đúng plain_text: "
                f"{rebuilt!r} != {self.plain_text!r}."
            )
        return self


class GenerateBatchResponse(BaseModel):
    request_id: str
    model_id: str
    model_revision: str | None
    artifact_schema_version: int
    cues: list[CueResult]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    device: str | None
    model_id: str | None
    model_revision: str | None
    artifact_schema_version: int | None
    tokenizer_special_tokens_valid: bool
    error: str | None = None


class VersionResponse(BaseModel):
    api_version: str
    model_id: str | None
    model_revision: str | None
    artifact_schema_version: int | None
