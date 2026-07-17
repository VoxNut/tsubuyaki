from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from furigana_aid.artifacts import ArtifactValidationError, load_artifact_bundle
from furigana_aid.artifacts import loader as loader_module


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_checksums(model_dir: Path) -> None:
    names = [
        "inference_artifacts.npz",
        "surfaces.json",
        "surface_mode_readings.json",
        "artifact_manifest.json",
    ]
    (model_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(model_dir / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _rewrite_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(model_dir / "artifact_manifest.json", manifest)
    _write_checksums(model_dir)


@pytest.fixture()
def valid_model_dir(tmp_path: Path) -> Path:
    label2id = {"きょう": 0, "こんにち": 1}
    id2label = {"0": "きょう", "1": "こんにち"}
    config = {
        "label2id": label2id,
        "id2label": id2label,
        "hidden_size": 2,
        "vocab_size": 12,
    }
    _write_json(tmp_path / "config.json", config)

    added_tokens = {"[TGT]": 10, "[/TGT]": 11}
    _write_json(tmp_path / "added_tokens.json", added_tokens)
    _write_json(
        tmp_path / "tokenizer_config.json",
        {
            "additional_special_tokens": ["[TGT]", "[/TGT]"],
            "added_tokens_decoder": {
                "10": {"content": "[TGT]", "special": True},
                "11": {"content": "[/TGT]", "special": True},
            },
        },
    )
    _write_json(
        tmp_path / "special_tokens_map.json",
        {
            "additional_special_tokens": [
                {"content": "[TGT]"},
                {"content": "[/TGT]"},
            ]
        },
    )
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"test-bert-safetensors")
    (tmp_path / "target_aware_mlp.safetensors").write_bytes(
        b"test-mlp-safetensors"
    )

    mlp_layers = [
        "LayerNorm(8)",
        "Linear(8,512)",
        "GELU",
        "Dropout(0.25)",
        "Linear(512,256)",
        "GELU",
        "Dropout(0.15)",
        "Linear(256,2)",
    ]
    calibration = {
        "alpha_mlp": 0.9,
        "prior_strength": 0.3,
        "confidence_threshold": 0.45,
    }
    mlp_metadata = {
        "schema_version": 1,
        "weights_file": "target_aware_mlp.safetensors",
        "weights_sha256": _sha256(
            tmp_path / "target_aware_mlp.safetensors"
        ),
        "input_dim": 8,
        "num_labels": 2,
        "architecture": {"layers": mlp_layers},
        "tuning": calibration,
        "tensors": {
            "network.0.bias": {"shape": [8], "dtype": "float32"},
            "network.0.weight": {"shape": [8], "dtype": "float32"},
            "network.1.bias": {"shape": [512], "dtype": "float32"},
            "network.1.weight": {"shape": [512, 8], "dtype": "float32"},
            "network.4.bias": {"shape": [256], "dtype": "float32"},
            "network.4.weight": {"shape": [256, 512], "dtype": "float32"},
            "network.7.bias": {"shape": [2], "dtype": "float32"},
            "network.7.weight": {"shape": [2, 256], "dtype": "float32"},
        },
    }
    _write_json(
        tmp_path / "target_aware_mlp.metadata.json",
        mlp_metadata,
    )

    surfaces = ["今日"]
    mode_payload = {
        "schema_version": 1,
        "description": "test fixture",
        "surfaces": surfaces,
        "readings": ["きょう"],
    }
    _write_json(tmp_path / "surfaces.json", surfaces)
    _write_json(tmp_path / "surface_mode_readings.json", mode_payload)
    np.savez_compressed(
        tmp_path / "inference_artifacts.npz",
        candidate_indptr=np.asarray([0, 2], dtype=np.int64),
        candidate_label_ids=np.asarray([0, 1], dtype=np.int32),
        candidate_log_priors=np.log(
            np.asarray([0.75, 0.25], dtype=np.float32)
        ).astype(np.float32),
        surface_mode_label_ids=np.asarray([0], dtype=np.int32),
    )

    artifact_names = [
        "inference_artifacts.npz",
        "surfaces.json",
        "surface_mode_readings.json",
    ]
    model_names = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "model.safetensors",
        "target_aware_mlp.safetensors",
        "target_aware_mlp.metadata.json",
    ]
    manifest = {
        "schema_version": 1,
        "model_id": "test/furigana-model",
        "model_revision": "0123456789abcdef",
        "dataset_revision": "fedcba9876543210",
        "num_labels": 2,
        "num_surfaces": 1,
        "num_mode_surfaces": 1,
        "num_candidates": 2,
        "surface_mode_label_id_missing_value": -1,
        "target_start_token": "[TGT]",
        "target_end_token": "[/TGT]",
        "target_start_token_id": 10,
        "target_end_token_id": 11,
        "max_length": 192,
        "max_merge_tokens": 4,
        "normalization_description": "fixture",
        "label_mapping_sha256": hashlib.sha256(
            _json_bytes(label2id)
        ).hexdigest(),
        "surface_order_sha256": hashlib.sha256(
            _json_bytes(surfaces)
        ).hexdigest(),
        "surface_mode_mapping_sha256": hashlib.sha256(
            _json_bytes(mode_payload)
        ).hexdigest(),
        "calibration": calibration,
        "decision_policy": {
            "candidate_order": "ascending_label_id",
            "fallback_condition": (
                "confidence < confidence_threshold and "
                "surface_mode_label_id >= 0"
            ),
            "context_source": "ContextEnsemble",
            "low_confidence_source": "MostFrequentLowConfidence",
            "non_contextual_seen_source": "MostFrequent",
            "unseen_source": "MeCabFallback",
        },
        "mlp_architecture": {
            "input_dim": 8,
            "layers": mlp_layers,
            "feature_pooling": ["cls", "start", "target", "end"],
        },
        "files": {
            name: _sha256(tmp_path / name) for name in artifact_names
        },
        "model_files": {
            name: _sha256(tmp_path / name) for name in model_names
        },
    }
    _write_json(tmp_path / "artifact_manifest.json", manifest)
    _write_checksums(tmp_path)
    return tmp_path


def test_loads_valid_bundle_and_candidate_lookup(valid_model_dir: Path) -> None:
    bundle = load_artifact_bundle(
        valid_model_dir,
        expected_model_revision="0123456789abcdef",
    )

    assert bundle.manifest.schema_version == 1
    assert bundle.tokenizer_special_tokens_validated is True
    assert bundle.label2id["きょう"] == 0
    assert bundle.id2label[1] == "こんにち"
    candidate_ids, priors, mode_id = bundle.candidates_for("今日")
    assert candidate_ids.tolist() == [0, 1]
    assert np.exp(priors.astype(np.float64)).sum() == pytest.approx(1.0)
    assert mode_id == 0
    assert candidate_ids.flags.writeable is False
    with pytest.raises(KeyError, match="Unknown contextual surface"):
        bundle.candidates_for("未知")


def test_npz_is_loaded_with_pickle_disabled(
    valid_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = loader_module.np.load
    observed: list[bool | None] = []

    def spy_load(*args: Any, **kwargs: Any) -> Any:
        observed.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(loader_module.np, "load", spy_load)
    load_artifact_bundle(valid_model_dir)
    assert observed == [False]


def test_rejects_checksum_mismatch(valid_model_dir: Path) -> None:
    with (valid_model_dir / "surfaces.json").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_artifact_bundle(valid_model_dir)


def test_rejects_label_mapping_hash_mismatch(valid_model_dir: Path) -> None:
    config_path = valid_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["label2id"] = {"きょう": 1, "こんにち": 0}
    config["id2label"] = {"0": "こんにち", "1": "きょう"}
    _write_json(config_path, config)

    manifest_path = valid_model_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_files"]["config.json"] = _sha256(config_path)
    _rewrite_manifest(valid_model_dir, manifest)

    with pytest.raises(ArtifactValidationError, match="mapping hash"):
        load_artifact_bundle(valid_model_dir)


def test_rejects_token_marker_id_mismatch(valid_model_dir: Path) -> None:
    added_tokens_path = valid_model_dir / "added_tokens.json"
    _write_json(added_tokens_path, {"[TGT]": 9, "[/TGT]": 11})
    manifest_path = valid_model_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_files"]["added_tokens.json"] = _sha256(added_tokens_path)
    _rewrite_manifest(valid_model_dir, manifest)

    with pytest.raises(ArtifactValidationError, match="marker ID mismatch"):
        load_artifact_bundle(valid_model_dir)


def test_rejects_wrong_npz_dtype(valid_model_dir: Path) -> None:
    artifact_path = valid_model_dir / "inference_artifacts.npz"
    np.savez_compressed(
        artifact_path,
        candidate_indptr=np.asarray([0, 2], dtype=np.int64),
        candidate_label_ids=np.asarray([0, 1], dtype=np.int64),
        candidate_log_priors=np.log(
            np.asarray([0.75, 0.25], dtype=np.float32)
        ).astype(np.float32),
        surface_mode_label_ids=np.asarray([0], dtype=np.int32),
    )
    manifest_path = valid_model_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["inference_artifacts.npz"] = _sha256(artifact_path)
    _rewrite_manifest(valid_model_dir, manifest)

    with pytest.raises(ArtifactValidationError, match="dtype"):
        load_artifact_bundle(valid_model_dir)


def test_rejects_mlp_metadata_dimension_mismatch(
    valid_model_dir: Path,
) -> None:
    metadata_path = valid_model_dir / "target_aware_mlp.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_dim"] = 16
    _write_json(metadata_path, metadata)
    manifest_path = valid_model_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_files"]["target_aware_mlp.metadata.json"] = _sha256(
        metadata_path
    )
    _rewrite_manifest(valid_model_dir, manifest)

    with pytest.raises(ArtifactValidationError, match="dimensions"):
        load_artifact_bundle(valid_model_dir)


def test_loads_real_recovered_artifacts_when_available() -> None:
    configured = os.environ.get("FURIGANA_TEST_MODEL_DIR")
    candidate = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[6] / "model"
    )
    if not (candidate / "artifact_manifest.json").is_file():
        pytest.skip("Recovered Phase 0 model artifacts are not present.")

    bundle = load_artifact_bundle(candidate)
    assert bundle.manifest.num_labels == 1491
    assert bundle.manifest.num_surfaces == 1004
    assert bundle.manifest.num_mode_surfaces == 26210
    assert bundle.manifest.num_candidates == 2506
    assert bundle.mlp_metadata.weights_file == "target_aware_mlp.safetensors"

