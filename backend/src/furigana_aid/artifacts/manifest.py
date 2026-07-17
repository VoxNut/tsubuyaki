"""Typed validation for the Phase 0 inference artifact manifest."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Mapping


SUPPORTED_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_CALIBRATION_KEYS = (
    "alpha_mlp",
    "prior_strength",
    "confidence_threshold",
)
EXPECTED_DECISION_POLICY = {
    "candidate_order": "ascending_label_id",
    "fallback_condition": (
        "confidence < confidence_threshold and surface_mode_label_id >= 0"
    ),
    "context_source": "ContextEnsemble",
    "low_confidence_source": "MostFrequentLowConfidence",
    "non_contextual_seen_source": "MostFrequent",
    "unseen_source": "MeCabFallback",
}


class ArtifactValidationError(RuntimeError):
    """Raised when model artifacts cannot be trusted for inference."""


def require_sha256(value: Any, field_name: str) -> str:
    digest = str(value)
    if SHA256_RE.fullmatch(digest) is None:
        raise ArtifactValidationError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )
    return digest


def require_safe_file_hashes(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactValidationError(f"{field_name} must be a non-empty object.")

    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = str(raw_name)
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ArtifactValidationError(
                f"{field_name} contains an unsafe filename: {name!r}."
            )
        if name in result:
            raise ArtifactValidationError(
                f"{field_name} contains duplicate filename {name!r}."
            )
        result[name] = require_sha256(raw_digest, f"{field_name}.{name}")
    return MappingProxyType(result)


def _require_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ArtifactValidationError(f"manifest.{key} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"manifest.{key} must be an integer."
        ) from exc
    if minimum is not None and result < minimum:
        raise ArtifactValidationError(
            f"manifest.{key} must be >= {minimum}; received {result}."
        )
    return result


def _require_non_empty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(
            f"manifest.{key} must be a non-empty string."
        )
    return value


@dataclass(frozen=True, slots=True)
class MlpManifest:
    """MLP architecture information embedded in the artifact manifest."""

    input_dim: int
    layers: tuple[str, ...]
    feature_pooling: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        num_labels: int,
    ) -> "MlpManifest":
        if not isinstance(value, Mapping):
            raise ArtifactValidationError(
                "manifest.mlp_architecture must be an object."
            )
        try:
            input_dim = int(value.get("input_dim"))
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                "manifest.mlp_architecture.input_dim must be an integer."
            ) from exc
        if input_dim <= 0:
            raise ArtifactValidationError(
                "manifest.mlp_architecture.input_dim must be positive."
            )

        raw_layers = value.get("layers")
        raw_pooling = value.get("feature_pooling")
        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or any(not isinstance(item, str) or not item for item in raw_layers)
        ):
            raise ArtifactValidationError(
                "manifest.mlp_architecture.layers must be a non-empty string list."
            )
        if (
            not isinstance(raw_pooling, list)
            or len(raw_pooling) != 4
            or any(not isinstance(item, str) or not item for item in raw_pooling)
        ):
            raise ArtifactValidationError(
                "manifest.mlp_architecture.feature_pooling must contain four strings."
            )

        expected_layers = (
            f"LayerNorm({input_dim})",
            f"Linear({input_dim},512)",
            "GELU",
            "Dropout(0.25)",
            "Linear(512,256)",
            "GELU",
            "Dropout(0.15)",
            f"Linear(256,{num_labels})",
        )
        layers = tuple(raw_layers)
        if layers != expected_layers:
            raise ArtifactValidationError(
                "manifest.mlp_architecture.layers does not match the audited MLP."
            )
        return cls(
            input_dim=input_dim,
            layers=layers,
            feature_pooling=tuple(raw_pooling),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Validated fields needed to load and serve a model snapshot."""

    schema_version: int
    model_id: str
    model_revision: str | None
    dataset_revision: str | None
    num_labels: int
    num_surfaces: int
    num_mode_surfaces: int
    num_candidates: int
    target_start_token: str
    target_end_token: str
    target_start_token_id: int
    target_end_token_id: int
    max_length: int
    max_merge_tokens: int
    label_mapping_sha256: str
    surface_order_sha256: str
    surface_mode_mapping_sha256: str
    calibration: Mapping[str, float]
    decision_policy: Mapping[str, str]
    mlp: MlpManifest
    files: Mapping[str, str]
    model_files: Mapping[str, str]

    @classmethod
    def from_mapping(cls, payload: Any) -> "ArtifactManifest":
        if not isinstance(payload, Mapping):
            raise ArtifactValidationError("artifact_manifest.json must be an object.")

        schema_version = _require_int(payload, "schema_version", minimum=1)
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ArtifactValidationError(
                "Unsupported artifact schema version "
                f"{schema_version}; expected {SUPPORTED_SCHEMA_VERSION}."
            )

        num_labels = _require_int(payload, "num_labels", minimum=1)
        sentinel = _require_int(payload, "surface_mode_label_id_missing_value")
        if sentinel != -1:
            raise ArtifactValidationError(
                "surface_mode_label_id_missing_value must be -1."
            )

        start_token = _require_non_empty_string(payload, "target_start_token")
        end_token = _require_non_empty_string(payload, "target_end_token")
        start_token_id = _require_int(
            payload, "target_start_token_id", minimum=0
        )
        end_token_id = _require_int(payload, "target_end_token_id", minimum=0)
        if start_token == end_token or start_token_id == end_token_id:
            raise ArtifactValidationError(
                "Target marker tokens and IDs must be distinct."
            )

        raw_calibration = payload.get("calibration")
        if not isinstance(raw_calibration, Mapping):
            raise ArtifactValidationError("manifest.calibration must be an object.")
        calibration: dict[str, float] = {}
        for key in REQUIRED_CALIBRATION_KEYS:
            try:
                value = float(raw_calibration[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactValidationError(
                    f"manifest.calibration.{key} is missing or invalid."
                ) from exc
            if not math.isfinite(value):
                raise ArtifactValidationError(
                    f"manifest.calibration.{key} must be finite."
                )
            calibration[key] = value

        raw_policy = payload.get("decision_policy")
        if not isinstance(raw_policy, Mapping):
            raise ArtifactValidationError(
                "manifest.decision_policy must be an object."
            )
        policy = {str(key): str(value) for key, value in raw_policy.items()}
        if policy != EXPECTED_DECISION_POLICY:
            raise ArtifactValidationError(
                "manifest.decision_policy does not match the audited notebook."
            )

        raw_revision = payload.get("model_revision")
        if raw_revision is not None and (
            not isinstance(raw_revision, str) or not raw_revision
        ):
            raise ArtifactValidationError(
                "manifest.model_revision must be null or a non-empty string."
            )
        raw_dataset_revision = payload.get("dataset_revision")
        if raw_dataset_revision is not None and (
            not isinstance(raw_dataset_revision, str)
            or not raw_dataset_revision
        ):
            raise ArtifactValidationError(
                "manifest.dataset_revision must be null or a non-empty string."
            )

        return cls(
            schema_version=schema_version,
            model_id=_require_non_empty_string(payload, "model_id"),
            model_revision=raw_revision,
            dataset_revision=raw_dataset_revision,
            num_labels=num_labels,
            num_surfaces=_require_int(payload, "num_surfaces", minimum=1),
            num_mode_surfaces=_require_int(
                payload, "num_mode_surfaces", minimum=1
            ),
            num_candidates=_require_int(payload, "num_candidates", minimum=1),
            target_start_token=start_token,
            target_end_token=end_token,
            target_start_token_id=start_token_id,
            target_end_token_id=end_token_id,
            max_length=_require_int(payload, "max_length", minimum=4),
            max_merge_tokens=_require_int(
                payload, "max_merge_tokens", minimum=1
            ),
            label_mapping_sha256=require_sha256(
                payload.get("label_mapping_sha256"),
                "manifest.label_mapping_sha256",
            ),
            surface_order_sha256=require_sha256(
                payload.get("surface_order_sha256"),
                "manifest.surface_order_sha256",
            ),
            surface_mode_mapping_sha256=require_sha256(
                payload.get("surface_mode_mapping_sha256"),
                "manifest.surface_mode_mapping_sha256",
            ),
            calibration=MappingProxyType(calibration),
            decision_policy=MappingProxyType(policy),
            mlp=MlpManifest.from_mapping(
                payload.get("mlp_architecture"),
                num_labels=num_labels,
            ),
            files=require_safe_file_hashes(
                payload.get("files"), "manifest.files"
            ),
            model_files=require_safe_file_hashes(
                payload.get("model_files"), "manifest.model_files"
            ),
        )

