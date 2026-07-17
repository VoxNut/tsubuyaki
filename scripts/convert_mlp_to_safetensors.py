"""Convert the trusted notebook MLP checkpoint to a pickle-free format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


EXPECTED_KEYS = {
    "network.0.weight",
    "network.0.bias",
    "network.1.weight",
    "network.1.bias",
    "network.4.weight",
    "network.4.bias",
    "network.7.weight",
    "network.7.bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path, metadata_path: Path) -> None:
    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("MLP checkpoint phải là dictionary.")
    required_payload = {
        "state_dict",
        "input_dim",
        "num_labels",
        "architecture",
        "tuning",
    }
    missing_payload = required_payload - set(payload)
    if missing_payload:
        raise ValueError(
            f"MLP checkpoint thiếu khóa: {sorted(missing_payload)}."
        )

    raw_state = payload["state_dict"]
    if not isinstance(raw_state, dict):
        raise TypeError("state_dict của MLP không phải dictionary.")
    state = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in raw_state.items()
        if isinstance(tensor, torch.Tensor)
    }
    if set(state) != EXPECTED_KEYS:
        raise ValueError(
            "Tên tensor MLP không khớp kiến trúc đã audit. "
            f"missing={sorted(EXPECTED_KEYS - set(state))}, "
            f"extra={sorted(set(state) - EXPECTED_KEYS)}"
        )

    input_dim = int(payload["input_dim"])
    num_labels = int(payload["num_labels"])
    expected_shapes = {
        "network.0.weight": (input_dim,),
        "network.0.bias": (input_dim,),
        "network.1.weight": (512, input_dim),
        "network.1.bias": (512,),
        "network.4.weight": (256, 512),
        "network.4.bias": (256,),
        "network.7.weight": (num_labels, 256),
        "network.7.bias": (num_labels,),
    }
    actual_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in state.items()
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Shape tensor MLP không khớp metadata: "
            f"expected={expected_shapes}, actual={actual_shapes}."
        )
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("Tất cả tensor MLP phải là float32.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(destination),
        metadata={
            "format": "pt",
            "schema_version": "1",
            "architecture": "target-aware-reading-mlp",
        },
    )
    loaded = load_file(str(destination), device="cpu")
    if set(loaded) != set(state):
        raise RuntimeError("Safetensors round-trip làm thay đổi tên tensor.")
    for name, expected in state.items():
        if not torch.equal(loaded[name], expected):
            raise RuntimeError(
                f"Safetensors round-trip làm thay đổi tensor {name!r}."
            )

    tuning = {
        str(name): float(value)
        for name, value in dict(payload["tuning"]).items()
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source_format": "trusted-local-pt-loaded-with-weights-only",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "weights_file": destination.name,
        "weights_sha256": sha256_file(destination),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dim": input_dim,
        "num_labels": num_labels,
        "architecture": {
            "description_from_checkpoint": str(payload["architecture"]),
            "layers": [
                f"LayerNorm({input_dim})",
                f"Linear({input_dim},512)",
                "GELU",
                "Dropout(0.25)",
                "Linear(512,256)",
                "GELU",
                "Dropout(0.15)",
                f"Linear(256,{num_labels})",
            ],
        },
        "tuning": tuning,
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for name, tensor in sorted(state.items())
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": str(source.resolve()),
                "destination": str(destination.resolve()),
                "metadata": str(metadata_path.resolve()),
                "num_tensors": len(state),
                "weights_sha256": metadata["weights_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parents[2]
    model_dir = workspace_root / "model"
    parser.add_argument(
        "--source",
        type=Path,
        default=model_dir / "target_aware_mlp.pt",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=model_dir / "target_aware_mlp.safetensors",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=model_dir / "target_aware_mlp.metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.source.resolve(),
        args.destination.resolve(),
        args.metadata.resolve(),
    )


if __name__ == "__main__":
    main()
