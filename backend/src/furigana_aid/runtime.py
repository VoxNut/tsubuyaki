"""Application-owned model lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from furigana_aid.artifacts import ArtifactBundle, load_artifact_bundle
from furigana_aid.config import Settings
from furigana_aid.device import resolve_device
from furigana_aid.inference.engine import Cue, CueResult, FuriganaEngine
from furigana_aid.inference.hybrid import HybridTuning
from furigana_aid.model_source import (
    ResolvedModelSource,
    resolve_model_source,
)
from furigana_aid.neural import NeuralModels, load_neural_models


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    source: ResolvedModelSource
    artifacts: ArtifactBundle
    neural: NeuralModels
    engine: FuriganaEngine
    model_revision: str | None

    @property
    def device(self) -> str:
        return self.neural.device.type

    @property
    def model_id(self) -> str:
        return self.artifacts.manifest.model_id

    @property
    def artifact_schema_version(self) -> int:
        return self.artifacts.manifest.schema_version

    @property
    def tokenizer_special_tokens_valid(self) -> bool:
        return self.artifacts.tokenizer_special_tokens_validated

    def generate_batch(self, cues: Sequence[Cue]) -> list[CueResult]:
        return self.engine.generate_batch(cues)


def build_runtime(settings: Settings) -> ModelRuntime:
    source = resolve_model_source(settings)
    artifacts = load_artifact_bundle(source.directory)
    if (
        source.remote
        and artifacts.manifest.model_revision is not None
        and artifacts.manifest.model_revision
        != source.configured_revision
    ):
        raise RuntimeError(
            "Pinned Hugging Face revision không khớp model_revision "
            "được khai báo trong artifact manifest."
        )

    device = resolve_device(settings.device)
    neural = load_neural_models(
        source.directory,
        device=device,
        expected_num_labels=artifacts.manifest.num_labels,
        expected_input_dim=artifacts.manifest.mlp.input_dim,
        expected_start_id=artifacts.manifest.target_start_token_id,
        expected_end_id=artifacts.manifest.target_end_token_id,
        max_length=artifacts.manifest.max_length,
        batch_size=settings.inference_batch_size,
    )
    tuning = HybridTuning(
        alpha_mlp=artifacts.manifest.calibration["alpha_mlp"],
        prior_strength=artifacts.manifest.calibration[
            "prior_strength"
        ],
        confidence_threshold=artifacts.manifest.calibration[
            "confidence_threshold"
        ],
    )
    engine = FuriganaEngine(
        artifacts=artifacts,
        neural=neural,
        tuning=tuning,
        max_merge_tokens=artifacts.manifest.max_merge_tokens,
    )
    revision = (
        source.configured_revision
        if source.remote
        else artifacts.manifest.model_revision
    )
    return ModelRuntime(
        source=source,
        artifacts=artifacts,
        neural=neural,
        engine=engine,
        model_revision=revision,
    )
