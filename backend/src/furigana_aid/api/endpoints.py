from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from furigana_aid.api.dependencies import get_runtime
from furigana_aid.api.schemas import (
    GenerateBatchRequest,
    GenerateBatchResponse,
    HealthResponse,
    ReadyResponse,
    VersionResponse,
    CueResult,
    Segment,
    TextSegment,
    RubySegment,
)
from furigana_aid.runtime import ModelRuntime
from furigana_aid.inference.engine import Cue

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request, runtime: ModelRuntime | None = Depends(get_runtime)) -> ReadyResponse:
    if runtime is None:
        err = getattr(request.app.state, "runtime_error", "Model runtime is not initialized.")
        return ReadyResponse(
            ready=False,
            device=None,
            model_id=None,
            model_revision=None,
            artifact_schema_version=None,
            tokenizer_special_tokens_valid=False,
            error=err
        )
    return ReadyResponse(
        ready=True,
        device=runtime.device,
        model_id=runtime.model_id,
        model_revision=runtime.model_revision,
        artifact_schema_version=runtime.artifact_schema_version,
        tokenizer_special_tokens_valid=runtime.tokenizer_special_tokens_valid,
    )


@router.get("/version", response_model=VersionResponse)
def version(runtime: ModelRuntime | None = Depends(get_runtime)) -> VersionResponse:
    if runtime is None:
        return VersionResponse(
            api_version="1.0.0",
            model_id=None,
            model_revision=None,
            artifact_schema_version=None,
        )
    return VersionResponse(
        api_version="1.0.0",
        model_id=runtime.model_id,
        model_revision=runtime.model_revision,
        artifact_schema_version=runtime.artifact_schema_version,
    )


@router.post("/furigana/generate-batch", response_model=GenerateBatchResponse)
def generate_batch(
    payload: GenerateBatchRequest,
    request: Request,
    runtime: ModelRuntime = Depends(get_runtime),
) -> GenerateBatchResponse:
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model runtime is not ready.",
        )

    settings = request.app.state.settings

    # Validate limits
    if len(payload.cues) > settings.max_cues_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Số lượng cues vượt quá giới hạn tối đa {settings.max_cues_per_request}."
        )

    total_chars = 0
    engine_cues = []
    for cue in payload.cues:
        if len(cue.text) > settings.max_chars_per_cue:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cue ID {cue.cue_id} vượt quá độ dài ký tự tối đa {settings.max_chars_per_cue}."
            )
        total_chars += len(cue.text)
        engine_cues.append(
            Cue(
                cue_id=cue.cue_id,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
            )
        )

    if total_chars > settings.max_total_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Tổng số ký tự ({total_chars}) vượt quá giới hạn tối đa {settings.max_total_chars}."
        )

    # Run batched inference
    results = runtime.generate_batch(engine_cues)

    # Map back to response schemas
    response_cues = []
    for r in results:
        segments: list[Segment] = []
        for seg in r.segments:
            if seg.type == "text":
                segments.append(TextSegment(type="text", text=seg.text))
            elif seg.type == "ruby":
                segments.append(
                    RubySegment(
                        type="ruby",
                        base=seg.base,
                        reading=seg.reading,
                        source=seg.source,
                        confidence=seg.confidence,
                        edited=seg.edited,
                    )
                )
        response_cues.append(
            CueResult(
                cue_id=r.cue_id,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                plain_text=r.plain_text,
                segments=segments,
            )
        )

    return GenerateBatchResponse(
        request_id=payload.request_id,
        model_id=runtime.model_id,
        model_revision=runtime.model_revision,
        artifact_schema_version=runtime.artifact_schema_version,
        cues=response_cues,
    )
