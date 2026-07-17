"""Fail-fast, pickle-free loader for Furigana Aid inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .manifest import (
    ArtifactManifest,
    ArtifactValidationError,
    REQUIRED_CALIBRATION_KEYS,
    require_sha256,
)


REQUIRED_CHECKSUM_FILES = frozenset(
    {
        "artifact_manifest.json",
        "inference_artifacts.npz",
        "surfaces.json",
        "surface_mode_readings.json",
    }
)
REQUIRED_ARRAYS = frozenset(
    {
        "candidate_indptr",
        "candidate_label_ids",
        "candidate_log_priors",
        "surface_mode_label_ids",
    }
)
EXPECTED_DTYPES = {
    "candidate_indptr": np.dtype(np.int64),
    "candidate_label_ids": np.dtype(np.int32),
    "candidate_log_priors": np.dtype(np.float32),
    "surface_mode_label_ids": np.dtype(np.int32),
}
MLP_WEIGHTS_FILE = "target_aware_mlp.safetensors"
MLP_METADATA_FILE = "target_aware_mlp.metadata.json"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactValidationError(f"Cannot read protected file: {path}.") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"Required JSON file is missing: {path}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Invalid JSON file: {path}.") from exc


def _parse_and_verify_checksums(model_dir: Path) -> Mapping[str, str]:
    checksum_path = model_dir / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ArtifactValidationError(
            f"Required checksum file is missing: {checksum_path}."
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(
            f"Cannot read checksum file: {checksum_path}."
        ) from exc

    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw_digest, raw_filename = line.split(None, 1)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"Malformed checksum line {line_number}."
            ) from exc
        filename = raw_filename.strip()
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            raise ArtifactValidationError(
                f"Unsafe checksum filename on line {line_number}: {filename!r}."
            )
        if filename in expected:
            raise ArtifactValidationError(
                f"Duplicate checksum entry for {filename!r}."
            )
        expected[filename] = require_sha256(
            raw_digest, f"checksums.sha256 line {line_number}"
        )

    missing = REQUIRED_CHECKSUM_FILES - expected.keys()
    if missing:
        raise ArtifactValidationError(
            f"checksums.sha256 is missing required files: {sorted(missing)}."
        )
    for filename, expected_digest in expected.items():
        path = model_dir / filename
        if not path.is_file():
            raise ArtifactValidationError(
                f"Checksum references a missing file: {filename!r}."
            )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ArtifactValidationError(
                f"SHA-256 mismatch for {filename!r}: "
                f"expected {expected_digest}, received {actual_digest}."
            )
    return MappingProxyType(expected)


def _verify_manifest_files(
    model_dir: Path,
    manifest: ArtifactManifest,
    checksums: Mapping[str, str],
) -> None:
    expected_artifact_files = {
        "inference_artifacts.npz",
        "surfaces.json",
        "surface_mode_readings.json",
    }
    if set(manifest.files) != expected_artifact_files:
        raise ArtifactValidationError(
            "manifest.files must list exactly the three logical artifact files."
        )
    for filename, digest in manifest.files.items():
        if checksums.get(filename) != digest:
            raise ArtifactValidationError(
                f"manifest.files checksum disagrees with checksums.sha256 "
                f"for {filename!r}."
            )

    required_model_files = {
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "model.safetensors",
        MLP_WEIGHTS_FILE,
        MLP_METADATA_FILE,
    }
    missing = required_model_files - manifest.model_files.keys()
    if missing:
        raise ArtifactValidationError(
            f"manifest.model_files is missing production files: {sorted(missing)}."
        )
    if any(filename.lower().endswith((".pt", ".bin")) for filename in manifest.model_files):
        raise ArtifactValidationError(
            "manifest.model_files must not use pickle-based model checkpoints."
        )

    for filename, expected_digest in manifest.model_files.items():
        path = model_dir / filename
        if not path.is_file():
            raise ArtifactValidationError(
                f"Manifest references a missing model file: {filename!r}."
            )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ArtifactValidationError(
                f"Model-file SHA-256 mismatch for {filename!r}: "
                f"expected {expected_digest}, received {actual_digest}."
            )


def _load_label_mapping(
    model_dir: Path,
    manifest: ArtifactManifest,
) -> tuple[Mapping[str, int], Mapping[int, str]]:
    payload = _read_json(model_dir / "config.json")
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("config.json must be an object.")
    raw_label2id = payload.get("label2id")
    raw_id2label = payload.get("id2label")
    if not isinstance(raw_label2id, Mapping) or not isinstance(
        raw_id2label, Mapping
    ):
        raise ArtifactValidationError(
            "config.json must contain label2id and id2label objects."
        )

    try:
        label2id = {str(label): int(index) for label, index in raw_label2id.items()}
        id2label = {int(index): str(label) for index, label in raw_id2label.items()}
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "config.json contains a non-integer label ID."
        ) from exc

    expected_ids = list(range(len(label2id)))
    if sorted(label2id.values()) != expected_ids:
        raise ArtifactValidationError(
            "config.json label2id IDs must be contiguous from zero."
        )
    if len(id2label) != len(label2id):
        raise ArtifactValidationError(
            "config.json label2id and id2label sizes differ."
        )
    for label, index in label2id.items():
        if id2label.get(index) != label:
            raise ArtifactValidationError(
                f"config.json label mappings are not inverse at {label!r}/{index}."
            )
    if len(label2id) != manifest.num_labels:
        raise ArtifactValidationError(
            "config.json label count does not match manifest.num_labels."
        )

    actual_hash = hashlib.sha256(_canonical_json_bytes(label2id)).hexdigest()
    if actual_hash != manifest.label_mapping_sha256:
        raise ArtifactValidationError(
            "config.json label mapping hash does not match the manifest."
        )
    return MappingProxyType(label2id), MappingProxyType(id2label)


def _validate_token_markers(
    model_dir: Path,
    manifest: ArtifactManifest,
) -> None:
    added_tokens = _read_json(model_dir / "added_tokens.json")
    if not isinstance(added_tokens, Mapping):
        raise ArtifactValidationError("added_tokens.json must be an object.")
    expected = {
        manifest.target_start_token: manifest.target_start_token_id,
        manifest.target_end_token: manifest.target_end_token_id,
    }
    for token, token_id in expected.items():
        try:
            recorded_id = int(added_tokens[token])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"added_tokens.json is missing marker {token!r}."
            ) from exc
        if recorded_id != token_id:
            raise ArtifactValidationError(
                f"Tokenizer marker ID mismatch for {token!r}: "
                f"manifest={token_id}, added_tokens.json={recorded_id}."
            )

    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    if not isinstance(tokenizer_config, Mapping):
        raise ArtifactValidationError("tokenizer_config.json must be an object.")
    special_tokens = tokenizer_config.get("additional_special_tokens")
    if not isinstance(special_tokens, list) or any(
        token not in special_tokens for token in expected
    ):
        raise ArtifactValidationError(
            "tokenizer_config.json does not register both target markers."
        )
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, Mapping):
        raise ArtifactValidationError(
            "tokenizer_config.json is missing added_tokens_decoder."
        )
    for token, token_id in expected.items():
        entry = decoder.get(str(token_id))
        if (
            not isinstance(entry, Mapping)
            or entry.get("content") != token
            or entry.get("special") is not True
        ):
            raise ArtifactValidationError(
                f"added_tokens_decoder is invalid for marker {token!r}."
            )

    special_map = _read_json(model_dir / "special_tokens_map.json")
    if not isinstance(special_map, Mapping):
        raise ArtifactValidationError("special_tokens_map.json must be an object.")
    raw_additional = special_map.get("additional_special_tokens")
    if not isinstance(raw_additional, list):
        raise ArtifactValidationError(
            "special_tokens_map.json is missing additional_special_tokens."
        )
    mapped_contents = {
        item if isinstance(item, str) else item.get("content")
        for item in raw_additional
        if isinstance(item, (str, Mapping))
    }
    if any(token not in mapped_contents for token in expected):
        raise ArtifactValidationError(
            "special_tokens_map.json does not contain both target markers."
        )


@dataclass(frozen=True, slots=True)
class MlpMetadata:
    """Validated metadata paired with the safe MLP safetensors file."""

    weights_file: str
    weights_sha256: str
    input_dim: int
    num_labels: int
    layers: tuple[str, ...]
    tuning: Mapping[str, float]
    tensors: Mapping[str, Mapping[str, Any]]


def _load_mlp_metadata(
    model_dir: Path,
    manifest: ArtifactManifest,
) -> MlpMetadata:
    payload = _read_json(model_dir / MLP_METADATA_FILE)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ArtifactValidationError(
            "target_aware_mlp.metadata.json has an unsupported schema."
        )
    if payload.get("weights_file") != MLP_WEIGHTS_FILE:
        raise ArtifactValidationError(
            "MLP metadata must reference target_aware_mlp.safetensors."
        )
    weights_sha256 = require_sha256(
        payload.get("weights_sha256"), "MLP metadata weights_sha256"
    )
    if weights_sha256 != manifest.model_files[MLP_WEIGHTS_FILE]:
        raise ArtifactValidationError(
            "MLP metadata weight hash disagrees with manifest.model_files."
        )

    try:
        input_dim = int(payload.get("input_dim"))
        num_labels = int(payload.get("num_labels"))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "MLP metadata input_dim/num_labels must be integers."
        ) from exc
    if input_dim != manifest.mlp.input_dim or num_labels != manifest.num_labels:
        raise ArtifactValidationError(
            "MLP metadata dimensions do not match the artifact manifest."
        )

    architecture = payload.get("architecture")
    raw_layers = (
        architecture.get("layers") if isinstance(architecture, Mapping) else None
    )
    if not isinstance(raw_layers, list) or tuple(raw_layers) != manifest.mlp.layers:
        raise ArtifactValidationError(
            "MLP metadata layers do not match the artifact manifest."
        )

    raw_tuning = payload.get("tuning")
    if not isinstance(raw_tuning, Mapping):
        raise ArtifactValidationError("MLP metadata tuning must be an object.")
    tuning: dict[str, float] = {}
    for key in REQUIRED_CALIBRATION_KEYS:
        try:
            value = float(raw_tuning[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"MLP metadata tuning is missing {key!r}."
            ) from exc
        if not math.isfinite(value) or not math.isclose(
            value,
            manifest.calibration[key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactValidationError(
                f"MLP tuning {key!r} does not match manifest.calibration."
            )
        tuning[key] = value

    raw_tensors = payload.get("tensors")
    if not isinstance(raw_tensors, Mapping):
        raise ArtifactValidationError("MLP metadata tensors must be an object.")
    expected_shapes = {
        "network.0.bias": [input_dim],
        "network.0.weight": [input_dim],
        "network.1.bias": [512],
        "network.1.weight": [512, input_dim],
        "network.4.bias": [256],
        "network.4.weight": [256, 512],
        "network.7.bias": [num_labels],
        "network.7.weight": [num_labels, 256],
    }
    if set(raw_tensors) != set(expected_shapes):
        raise ArtifactValidationError(
            "MLP metadata tensor names do not match the audited architecture."
        )
    tensors: dict[str, Mapping[str, Any]] = {}
    for name, expected_shape in expected_shapes.items():
        entry = raw_tensors[name]
        if (
            not isinstance(entry, Mapping)
            or entry.get("shape") != expected_shape
            or entry.get("dtype") != "float32"
        ):
            raise ArtifactValidationError(
                f"MLP tensor metadata is invalid for {name!r}."
            )
        tensors[name] = MappingProxyType(
            {"shape": tuple(expected_shape), "dtype": "float32"}
        )

    return MlpMetadata(
        weights_file=MLP_WEIGHTS_FILE,
        weights_sha256=weights_sha256,
        input_dim=input_dim,
        num_labels=num_labels,
        layers=tuple(raw_layers),
        tuning=MappingProxyType(tuning),
        tensors=MappingProxyType(tensors),
    )


def _load_surfaces(
    model_dir: Path,
    manifest: ArtifactManifest,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    surfaces_payload = _read_json(model_dir / "surfaces.json")
    if (
        not isinstance(surfaces_payload, list)
        or not surfaces_payload
        or any(
            not isinstance(surface, str) or not surface
            for surface in surfaces_payload
        )
    ):
        raise ArtifactValidationError(
            "surfaces.json must be a non-empty string list."
        )
    if surfaces_payload != sorted(surfaces_payload) or len(
        set(surfaces_payload)
    ) != len(surfaces_payload):
        raise ArtifactValidationError(
            "surfaces.json must be unique and sorted."
        )
    if len(surfaces_payload) != manifest.num_surfaces:
        raise ArtifactValidationError(
            "surfaces.json count does not match manifest.num_surfaces."
        )
    actual_surface_hash = hashlib.sha256(
        _canonical_json_bytes(surfaces_payload)
    ).hexdigest()
    if actual_surface_hash != manifest.surface_order_sha256:
        raise ArtifactValidationError(
            "surfaces.json order hash does not match the manifest."
        )

    mode_payload = _read_json(model_dir / "surface_mode_readings.json")
    if not isinstance(mode_payload, Mapping) or mode_payload.get(
        "schema_version"
    ) != 1:
        raise ArtifactValidationError(
            "surface_mode_readings.json has an unsupported schema."
        )
    mode_surfaces = mode_payload.get("surfaces")
    mode_readings = mode_payload.get("readings")
    if (
        not isinstance(mode_surfaces, list)
        or not isinstance(mode_readings, list)
        or len(mode_surfaces) != len(mode_readings)
        or any(
            not isinstance(value, str) or not value
            for value in mode_surfaces + mode_readings
        )
    ):
        raise ArtifactValidationError(
            "surface_mode_readings.json must contain equal non-empty string lists."
        )
    if mode_surfaces != sorted(mode_surfaces) or len(set(mode_surfaces)) != len(
        mode_surfaces
    ):
        raise ArtifactValidationError(
            "surface_mode_readings surfaces must be unique and sorted."
        )
    if len(mode_surfaces) != manifest.num_mode_surfaces:
        raise ArtifactValidationError(
            "Mode-surface count does not match manifest.num_mode_surfaces."
        )
    actual_mode_hash = hashlib.sha256(
        _canonical_json_bytes(mode_payload)
    ).hexdigest()
    if actual_mode_hash != manifest.surface_mode_mapping_sha256:
        raise ArtifactValidationError(
            "surface_mode_readings.json hash does not match the manifest."
        )
    full_modes = dict(zip(mode_surfaces, mode_readings, strict=True))
    missing_contextual = set(surfaces_payload) - full_modes.keys()
    if missing_contextual:
        raise ArtifactValidationError(
            "Full mode mapping is missing contextual surfaces."
        )
    return tuple(surfaces_payload), MappingProxyType(full_modes)


def _load_arrays(
    model_dir: Path,
    manifest: ArtifactManifest,
    surfaces: tuple[str, ...],
    surface_mode_readings: Mapping[str, str],
    label2id: Mapping[str, int],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    path = model_dir / "inference_artifacts.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_ARRAYS:
                raise ArtifactValidationError(
                    "inference_artifacts.npz has missing or unexpected arrays."
                )
            for name, expected_dtype in EXPECTED_DTYPES.items():
                if archive[name].dtype != expected_dtype:
                    raise ArtifactValidationError(
                        f"{name} dtype is {archive[name].dtype}; "
                        f"expected {expected_dtype}."
                    )
            indptr = np.array(archive["candidate_indptr"], copy=True)
            candidate_ids = np.array(archive["candidate_label_ids"], copy=True)
            priors = np.array(archive["candidate_log_priors"], copy=True)
            mode_ids = np.array(archive["surface_mode_label_ids"], copy=True)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Cannot safely load NPZ artifact with allow_pickle=False: {path}."
        ) from exc

    if indptr.ndim != 1 or indptr.shape != (manifest.num_surfaces + 1,):
        raise ArtifactValidationError("candidate_indptr has an invalid shape.")
    if candidate_ids.ndim != 1 or priors.shape != candidate_ids.shape:
        raise ArtifactValidationError(
            "candidate_label_ids/candidate_log_priors shapes are invalid."
        )
    if mode_ids.ndim != 1 or mode_ids.shape != (manifest.num_surfaces,):
        raise ArtifactValidationError(
            "surface_mode_label_ids has an invalid shape."
        )
    if (
        int(indptr[0]) != 0
        or int(indptr[-1]) != len(candidate_ids)
        or np.any(np.diff(indptr) <= 0)
    ):
        raise ArtifactValidationError(
            "candidate_indptr boundaries must define non-empty rows."
        )
    if len(candidate_ids) != manifest.num_candidates:
        raise ArtifactValidationError(
            "Candidate count does not match manifest.num_candidates."
        )
    if (
        len(candidate_ids) == 0
        or int(candidate_ids.min()) < 0
        or int(candidate_ids.max()) >= manifest.num_labels
    ):
        raise ArtifactValidationError("Candidate label IDs are out of range.")
    if np.any((mode_ids < -1) | (mode_ids >= manifest.num_labels)):
        raise ArtifactValidationError("Surface mode label IDs are out of range.")
    if not np.isfinite(priors).all():
        raise ArtifactValidationError("Candidate log priors contain NaN or Inf.")

    for surface_index, surface in enumerate(surfaces):
        start = int(indptr[surface_index])
        end = int(indptr[surface_index + 1])
        row_ids = candidate_ids[start:end]
        row_priors = priors[start:end]
        if np.any(np.diff(row_ids.astype(np.int64, copy=False)) <= 0):
            raise ArtifactValidationError(
                f"Candidate IDs are not strictly ascending for surface {surface!r}."
            )
        probability_sum = float(np.exp(row_priors.astype(np.float64)).sum())
        if not np.isclose(
            probability_sum,
            1.0,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ArtifactValidationError(
                f"Candidate priors are not normalized for surface {surface!r}."
            )
        expected_mode_id = int(label2id.get(surface_mode_readings[surface], -1))
        actual_mode_id = int(mode_ids[surface_index])
        if actual_mode_id != expected_mode_id:
            raise ArtifactValidationError(
                f"Surface mode label ID is inconsistent for {surface!r}."
            )
        if expected_mode_id >= 0 and not np.any(row_ids == expected_mode_id):
            raise ArtifactValidationError(
                f"Surface mode is not a candidate for {surface!r}."
            )

    for array in (indptr, candidate_ids, priors, mode_ids):
        array.setflags(write=False)
    return indptr, candidate_ids, priors, mode_ids


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """All immutable lookup data required by the inference engine."""

    model_dir: Path
    manifest: ArtifactManifest
    mlp_metadata: MlpMetadata
    surfaces: tuple[str, ...]
    surface2idx: Mapping[str, int]
    surface_mode_readings: Mapping[str, str]
    label2id: Mapping[str, int]
    id2label: Mapping[int, str]
    candidate_indptr: NDArray[np.int64]
    candidate_label_ids: NDArray[np.int32]
    candidate_log_priors: NDArray[np.float32]
    surface_mode_label_ids: NDArray[np.int32]
    tokenizer_special_tokens_validated: bool = True

    def candidates_for(
        self,
        surface: str,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], int]:
        """Return the CSR candidate row for a contextual surface."""

        try:
            surface_index = self.surface2idx[surface]
        except KeyError as exc:
            raise KeyError(f"Unknown contextual surface: {surface!r}.") from exc
        start = int(self.candidate_indptr[surface_index])
        end = int(self.candidate_indptr[surface_index + 1])
        return (
            self.candidate_label_ids[start:end],
            self.candidate_log_priors[start:end],
            int(self.surface_mode_label_ids[surface_index]),
        )


def load_artifact_bundle(
    model_dir: str | Path,
    *,
    expected_model_revision: str | None = None,
) -> ArtifactBundle:
    """Load and fully validate a local or downloaded model snapshot.

    This loader never deserializes pickle. Every logical artifact is checked
    against ``checksums.sha256`` and every production model/tokenizer/MLP file
    is checked against ``manifest.model_files`` before inference can start.
    """

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactValidationError(
            f"Model artifact directory does not exist: {root}."
        )

    checksums = _parse_and_verify_checksums(root)
    manifest_payload = _read_json(root / "artifact_manifest.json")
    manifest = ArtifactManifest.from_mapping(manifest_payload)
    if (
        expected_model_revision is not None
        and manifest.model_revision != expected_model_revision
    ):
        raise ArtifactValidationError(
            "Artifact model revision mismatch: "
            f"expected {expected_model_revision!r}, "
            f"received {manifest.model_revision!r}."
        )

    _verify_manifest_files(root, manifest, checksums)
    label2id, id2label = _load_label_mapping(root, manifest)
    _validate_token_markers(root, manifest)
    mlp_metadata = _load_mlp_metadata(root, manifest)
    surfaces, surface_mode_readings = _load_surfaces(root, manifest)
    indptr, candidate_ids, priors, mode_ids = _load_arrays(
        root,
        manifest,
        surfaces,
        surface_mode_readings,
        label2id,
    )
    surface2idx = MappingProxyType(
        {surface: index for index, surface in enumerate(surfaces)}
    )
    return ArtifactBundle(
        model_dir=root,
        manifest=manifest,
        mlp_metadata=mlp_metadata,
        surfaces=surfaces,
        surface2idx=surface2idx,
        surface_mode_readings=surface_mode_readings,
        label2id=label2id,
        id2label=id2label,
        candidate_indptr=indptr,
        candidate_label_ids=candidate_ids,
        candidate_log_priors=priors,
        surface_mode_label_ids=mode_ids,
    )

