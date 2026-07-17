"""Candidate-restricted HybridTuned decision logic.

The numerical formula mirrors the Kaggle notebook: each model is normalized
inside the valid candidate set, then the two log-probabilities and the surface
prior are combined.  No dense ``surface x label`` mask is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Mapping, Protocol, Sequence


PredictionSource = Literal[
    "ContextEnsemble",
    "MostFrequentLowConfidence",
    "MostFrequent",
    "MeCabFallback",
    "PlainText",
]

CONTEXT_ENSEMBLE: PredictionSource = "ContextEnsemble"
MOST_FREQUENT_LOW_CONFIDENCE: PredictionSource = (
    "MostFrequentLowConfidence"
)
MOST_FREQUENT: PredictionSource = "MostFrequent"
MECAB_FALLBACK: PredictionSource = "MeCabFallback"
PLAIN_TEXT: PredictionSource = "PlainText"


class CandidateArtifact(Protocol):
    """Minimal production artifact interface required by this module."""

    surface2idx: Mapping[str, int]
    surface_mode_readings: Mapping[str, str]
    id2label: Mapping[int, str]

    def candidates_for(
        self, surface: str
    ) -> tuple[Sequence[int], Sequence[float], int]:
        """Return candidate IDs, matching log priors and the mode label ID."""


@dataclass(frozen=True, slots=True)
class CsrCandidateIndex:
    """Validated, dependency-free CSR-like candidate lookup.

    ``candidate_indptr[i]:candidate_indptr[i + 1]`` identifies the only label
    IDs that are valid for ``surfaces[i]``.  Missing labels are absent rather
    than represented by a misleading zero prior.
    """

    surfaces: Sequence[str]
    candidate_indptr: Sequence[int]
    candidate_label_ids: Sequence[int]
    candidate_log_priors: Sequence[float]
    surface_mode_label_ids: Sequence[int]
    surface2idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        surfaces = tuple(str(value) for value in self.surfaces)
        indptr = tuple(int(value) for value in self.candidate_indptr)
        label_ids = tuple(int(value) for value in self.candidate_label_ids)
        priors = tuple(float(value) for value in self.candidate_log_priors)
        mode_ids = tuple(int(value) for value in self.surface_mode_label_ids)

        if not surfaces:
            raise ValueError("CSR candidate index must contain a surface.")
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("CSR candidate surfaces must be unique.")
        if len(indptr) != len(surfaces) + 1 or indptr[0] != 0:
            raise ValueError(
                "candidate_indptr must start at 0 and have N + 1 entries."
            )
        if indptr[-1] != len(label_ids) or len(label_ids) != len(priors):
            raise ValueError(
                "CSR boundary must equal the candidate/prior array length."
            )
        if len(mode_ids) != len(surfaces):
            raise ValueError(
                "surface_mode_label_ids must have one entry per surface."
            )
        for row, (start, end) in enumerate(zip(indptr, indptr[1:])):
            if start < 0 or end <= start:
                raise ValueError(
                    f"Surface row {row} must contain at least one candidate."
                )
            row_ids = label_ids[start:end]
            if any(label_id < 0 for label_id in row_ids):
                raise ValueError("Candidate label IDs cannot be negative.")
            if len(set(row_ids)) != len(row_ids):
                raise ValueError(
                    f"Surface row {row} contains duplicate candidate IDs."
                )
        if any(not math.isfinite(value) for value in priors):
            raise ValueError("Candidate log priors must all be finite.")
        if any(mode_id < -1 for mode_id in mode_ids):
            raise ValueError("Mode label IDs may only use -1 as a sentinel.")

        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "candidate_indptr", indptr)
        object.__setattr__(self, "candidate_label_ids", label_ids)
        object.__setattr__(self, "candidate_log_priors", priors)
        object.__setattr__(self, "surface_mode_label_ids", mode_ids)
        object.__setattr__(
            self,
            "surface2idx",
            {surface: index for index, surface in enumerate(surfaces)},
        )

    def candidates_for(
        self, surface: str
    ) -> tuple[tuple[int, ...], tuple[float, ...], int]:
        try:
            surface_index = self.surface2idx[surface]
        except KeyError as exc:
            raise KeyError(f"Unknown contextual surface: {surface!r}") from exc
        start = self.candidate_indptr[surface_index]
        end = self.candidate_indptr[surface_index + 1]
        return (
            tuple(self.candidate_label_ids[start:end]),
            tuple(self.candidate_log_priors[start:end]),
            self.surface_mode_label_ids[surface_index],
        )


@dataclass(frozen=True, slots=True)
class HybridTuning:
    alpha_mlp: float
    prior_strength: float
    confidence_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_mlp <= 1.0:
            raise ValueError("alpha_mlp must be in [0, 1].")
        if not math.isfinite(self.prior_strength):
            raise ValueError("prior_strength must be finite.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class HybridDecision:
    reading: str | None
    source: PredictionSource
    confidence: float | None
    candidate_label_ids: tuple[int, ...] = ()
    selected_label_id: int | None = None


def candidate_log_softmax(values: Sequence[float]) -> tuple[float, ...]:
    """Return a stable float64-style log-softmax for one candidate row."""

    row = tuple(float(value) for value in values)
    if not row:
        raise ValueError("Candidate log-softmax requires at least one value.")
    if any(not math.isfinite(value) for value in row):
        raise ValueError("Candidate logits must all be finite.")
    maximum = max(row)
    log_normalizer = math.log(sum(math.exp(value - maximum) for value in row))
    return tuple(value - maximum - log_normalizer for value in row)


def calibrated_candidate_prediction(
    linear_logits: Sequence[float],
    mlp_logits: Sequence[float],
    candidate_label_ids: Sequence[int],
    candidate_log_priors: Sequence[float],
    tuning: HybridTuning,
) -> tuple[int, float]:
    """Apply the notebook's calibrated ensemble inside one candidate set."""

    candidate_ids = tuple(int(value) for value in candidate_label_ids)
    priors = tuple(float(value) for value in candidate_log_priors)
    if not candidate_ids:
        raise ValueError("A contextual surface must have a candidate.")
    if len(candidate_ids) != len(priors):
        raise ValueError("Candidate IDs and log priors must have equal length.")
    if any(label_id < 0 for label_id in candidate_ids):
        raise ValueError("Candidate label IDs cannot be negative.")
    if any(not math.isfinite(value) for value in priors):
        raise ValueError("Candidate log priors must all be finite.")

    try:
        linear_values = tuple(
            float(linear_logits[label_id]) for label_id in candidate_ids
        )
        mlp_values = tuple(
            float(mlp_logits[label_id]) for label_id in candidate_ids
        )
    except IndexError as exc:
        raise ValueError("A candidate label ID is outside the logits.") from exc

    linear_logp = candidate_log_softmax(linear_values)
    mlp_logp = candidate_log_softmax(mlp_values)
    scores = tuple(
        (1.0 - tuning.alpha_mlp) * linear_value
        + tuning.alpha_mlp * mlp_value
        + tuning.prior_strength * prior
        for linear_value, mlp_value, prior in zip(
            linear_logp, mlp_logp, priors
        )
    )
    maximum = max(scores)
    weights = tuple(math.exp(score - maximum) for score in scores)
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    best_local_index = max(
        range(len(probabilities)), key=probabilities.__getitem__
    )
    return candidate_ids[best_local_index], probabilities[best_local_index]


def contextual_decision(
    *,
    surface: str,
    linear_logits: Sequence[float],
    mlp_logits: Sequence[float],
    artifacts: CandidateArtifact,
    tuning: HybridTuning,
) -> HybridDecision:
    """Predict one known contextual surface, including strict fallback."""

    candidate_ids, priors, mode_label_id = artifacts.candidates_for(surface)
    prediction_id, confidence = calibrated_candidate_prediction(
        linear_logits,
        mlp_logits,
        candidate_ids,
        priors,
        tuning,
    )
    source: PredictionSource = CONTEXT_ENSEMBLE

    # This is deliberately strict "<", matching the validation-tuned notebook.
    if (
        confidence < tuning.confidence_threshold
        and int(mode_label_id) >= 0
    ):
        prediction_id = int(mode_label_id)
        source = MOST_FREQUENT_LOW_CONFIDENCE

    try:
        reading = artifacts.id2label[int(prediction_id)]
    except KeyError as exc:
        raise ValueError(
            f"Artifact id2label is missing label ID {prediction_id}."
        ) from exc
    return HybridDecision(
        reading=str(reading),
        source=source,
        confidence=float(confidence),
        candidate_label_ids=tuple(int(value) for value in candidate_ids),
        selected_label_id=int(prediction_id),
    )


def non_contextual_decision(
    *,
    surface: str,
    surface_mode_readings: Mapping[str, str],
    mecab_reading: str | None,
) -> HybridDecision:
    """Route a surface that is not supported by the contextual classifier."""

    if surface in surface_mode_readings:
        return HybridDecision(
            reading=str(surface_mode_readings[surface]),
            source=MOST_FREQUENT,
            confidence=None,
        )
    if mecab_reading and mecab_reading != surface:
        return HybridDecision(
            reading=mecab_reading,
            source=MECAB_FALLBACK,
            confidence=None,
        )
    return HybridDecision(
        reading=None,
        source=PLAIN_TEXT,
        confidence=None,
    )
