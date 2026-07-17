"""Production inference primitives for Furigana Aid Reader."""

from .context_window import (
    InvalidTargetSpan,
    TargetCenteredContext,
    TargetExceedsTokenBudget,
    build_target_centered_context,
    mark_target,
)
from .features import TargetMarkerError, pool_target_features
from .hybrid import (
    CONTEXT_ENSEMBLE,
    MECAB_FALLBACK,
    MOST_FREQUENT,
    MOST_FREQUENT_LOW_CONFIDENCE,
    PLAIN_TEXT,
    CsrCandidateIndex,
    HybridDecision,
    HybridTuning,
    calibrated_candidate_prediction,
    candidate_log_softmax,
    contextual_decision,
    non_contextual_decision,
)
from .segments import (
    FuriganaSegment,
    MorphToken,
    SurfaceSpan,
    build_furigana_segments,
    greedy_longest_known_spans,
    plain_text_from_segments,
    tokenize_with_offsets,
    validate_segment_reconstruction,
)

__all__ = [
    "CONTEXT_ENSEMBLE",
    "MECAB_FALLBACK",
    "MOST_FREQUENT",
    "MOST_FREQUENT_LOW_CONFIDENCE",
    "PLAIN_TEXT",
    "CsrCandidateIndex",
    "FuriganaSegment",
    "HybridDecision",
    "HybridTuning",
    "InvalidTargetSpan",
    "MorphToken",
    "SurfaceSpan",
    "TargetCenteredContext",
    "TargetExceedsTokenBudget",
    "TargetMarkerError",
    "build_furigana_segments",
    "build_target_centered_context",
    "calibrated_candidate_prediction",
    "candidate_log_softmax",
    "contextual_decision",
    "greedy_longest_known_spans",
    "mark_target",
    "non_contextual_decision",
    "plain_text_from_segments",
    "pool_target_features",
    "tokenize_with_offsets",
    "validate_segment_reconstruction",
]
