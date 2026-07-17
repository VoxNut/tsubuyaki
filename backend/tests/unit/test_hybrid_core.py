from __future__ import annotations

from dataclasses import dataclass

import pytest

from furigana_aid.inference.hybrid import (
    CONTEXT_ENSEMBLE,
    MECAB_FALLBACK,
    MOST_FREQUENT,
    MOST_FREQUENT_LOW_CONFIDENCE,
    PLAIN_TEXT,
    CsrCandidateIndex,
    HybridTuning,
    calibrated_candidate_prediction,
    contextual_decision,
    non_contextual_decision,
)


@dataclass
class FakeArtifacts:
    index: CsrCandidateIndex
    id2label: dict[int, str]
    surface_mode_readings: dict[str, str]

    @property
    def surface2idx(self) -> dict[str, int]:
        return dict(self.index.surface2idx)

    def candidates_for(self, surface: str):
        return self.index.candidates_for(surface)


def test_csr_candidate_lookup_uses_only_ragged_row() -> None:
    index = CsrCandidateIndex(
        surfaces=["今日", "日"],
        candidate_indptr=[0, 2, 5],
        candidate_label_ids=[0, 2, 1, 3, 4],
        candidate_log_priors=[-0.2, -1.7, -0.1, -2.0, -3.0],
        surface_mode_label_ids=[0, 3],
    )

    assert index.candidates_for("今日") == (
        (0, 2),
        (-0.2, -1.7),
        0,
    )
    assert index.candidates_for("日")[0] == (1, 3, 4)
    with pytest.raises(KeyError, match="Unknown contextual surface"):
        index.candidates_for("未知")


def test_candidate_prediction_ignores_non_candidate_global_maximum() -> None:
    prediction_id, confidence = calibrated_candidate_prediction(
        linear_logits=[0.0, 100.0, 2.0],
        mlp_logits=[0.0, 100.0, 2.0],
        candidate_label_ids=[0, 2],
        candidate_log_priors=[0.0, 0.0],
        tuning=HybridTuning(
            alpha_mlp=0.9,
            prior_strength=0.3,
            confidence_threshold=0.45,
        ),
    )

    assert prediction_id == 2
    assert confidence == pytest.approx(0.8807970779778823)


def test_low_confidence_comparison_is_strictly_less_than_threshold() -> None:
    artifacts = FakeArtifacts(
        index=CsrCandidateIndex(
            surfaces=["日"],
            candidate_indptr=[0, 2],
            candidate_label_ids=[0, 1],
            candidate_log_priors=[0.0, 0.0],
            surface_mode_label_ids=[1],
        ),
        id2label={0: "にち", 1: "ひ"},
        surface_mode_readings={"日": "ひ"},
    )
    equal = contextual_decision(
        surface="日",
        linear_logits=[0.0, 0.0],
        mlp_logits=[0.0, 0.0],
        artifacts=artifacts,
        tuning=HybridTuning(0.9, 0.3, 0.5),
    )
    below = contextual_decision(
        surface="日",
        linear_logits=[0.0, 0.0],
        mlp_logits=[0.0, 0.0],
        artifacts=artifacts,
        tuning=HybridTuning(0.9, 0.3, 0.500001),
    )

    assert equal.source == CONTEXT_ENSEMBLE
    assert equal.reading == "にち"
    assert equal.confidence == pytest.approx(0.5)
    assert below.source == MOST_FREQUENT_LOW_CONFIDENCE
    assert below.reading == "ひ"
    # The notebook keeps ensemble confidence after choosing the mode reading.
    assert below.confidence == pytest.approx(0.5)


def test_mode_sentinel_disables_low_confidence_fallback() -> None:
    artifacts = FakeArtifacts(
        index=CsrCandidateIndex(
            surfaces=["日"],
            candidate_indptr=[0, 2],
            candidate_label_ids=[0, 1],
            candidate_log_priors=[0.0, 0.0],
            surface_mode_label_ids=[-1],
        ),
        id2label={0: "にち", 1: "じつ"},
        surface_mode_readings={"日": "ひ"},
    )

    decision = contextual_decision(
        surface="日",
        linear_logits=[0.0, 0.0],
        mlp_logits=[0.0, 0.0],
        artifacts=artifacts,
        tuning=HybridTuning(0.9, 0.3, 0.9),
    )
    assert decision.source == CONTEXT_ENSEMBLE
    assert decision.selected_label_id == 0


def test_non_contextual_sources_are_explicit() -> None:
    modes = {"東京": "とうきょう"}

    mode = non_contextual_decision(
        surface="東京",
        surface_mode_readings=modes,
        mecab_reading="unused",
    )
    mecab = non_contextual_decision(
        surface="京都",
        surface_mode_readings=modes,
        mecab_reading="きょうと",
    )
    plain = non_contextual_decision(
        surface="未知",
        surface_mode_readings=modes,
        mecab_reading="未知",
    )

    assert (mode.reading, mode.source) == ("とうきょう", MOST_FREQUENT)
    assert (mecab.reading, mecab.source) == ("きょうと", MECAB_FALLBACK)
    assert (plain.reading, plain.source) == (None, PLAIN_TEXT)
