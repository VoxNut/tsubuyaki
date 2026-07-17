"""Load and batch the BERT classification head plus target-aware MLP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class ReadingMLP(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True, slots=True)
class NeuralModels:
    tokenizer: PreTrainedTokenizerBase
    classifier: PreTrainedModel
    mlp: ReadingMLP
    device: torch.device
    target_start_id: int
    target_end_id: int
    max_length: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class NeuralOutput:
    linear_logits: np.ndarray
    mlp_logits: np.ndarray


def load_neural_models(
    model_dir: Path,
    *,
    device: torch.device,
    expected_num_labels: int,
    expected_input_dim: int,
    expected_start_id: int,
    expected_end_id: int,
    max_length: int,
    batch_size: int,
) -> NeuralModels:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    marker_ids = {
        "[TGT]": int(tokenizer.convert_tokens_to_ids("[TGT]")),
        "[/TGT]": int(tokenizer.convert_tokens_to_ids("[/TGT]")),
    }
    if marker_ids != {
        "[TGT]": expected_start_id,
        "[/TGT]": expected_end_id,
    }:
        raise ValueError(
            "Tokenizer target marker IDs không khớp manifest: "
            f"{marker_ids!r}."
        )
    additional = set(tokenizer.additional_special_tokens or ())
    if {"[TGT]", "[/TGT]"} - additional:
        raise ValueError(
            "Tokenizer thiếu [TGT]/[/TGT] trong additional special tokens."
        )

    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    if int(classifier.config.num_labels) != expected_num_labels:
        raise ValueError(
            "Classifier num_labels không khớp artifact manifest."
        )
    classifier.to(device).eval()

    metadata_path = model_dir / "target_aware_mlp.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_dim = int(metadata["input_dim"])
    num_labels = int(metadata["num_labels"])
    if input_dim != expected_input_dim or num_labels != expected_num_labels:
        raise ValueError(
            "MLP metadata không khớp manifest/model: "
            f"input_dim={input_dim}, num_labels={num_labels}."
        )
    mlp = ReadingMLP(input_dim, num_labels)
    weights = load_file(
        str(model_dir / "target_aware_mlp.safetensors"),
        device="cpu",
    )
    mlp.load_state_dict(weights, strict=True)
    mlp.to(device).eval()

    return NeuralModels(
        tokenizer=tokenizer,
        classifier=classifier,
        mlp=mlp,
        device=device,
        target_start_id=expected_start_id,
        target_end_id=expected_end_id,
        max_length=max_length,
        batch_size=batch_size,
    )


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, dict):
        value = value["input_ids"]
    return [int(token_id) for token_id in value]


def build_target_centered_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    sentence: str,
    start: int,
    end: int,
    target_start_id: int,
    target_end_id: int,
    max_length: int,
) -> list[int]:
    """Build a balanced token window that can never truncate the target."""

    if not 0 <= start < end <= len(sentence):
        raise ValueError(
            f"Target offsets không hợp lệ: {start}:{end}/{len(sentence)}."
        )
    prefix_ids = _as_token_ids(
        tokenizer(
            sentence[:start],
            add_special_tokens=False,
        )
    )
    target_ids = _as_token_ids(
        tokenizer(
            sentence[start:end],
            add_special_tokens=False,
        )
    )
    suffix_ids = _as_token_ids(
        tokenizer(
            sentence[end:],
            add_special_tokens=False,
        )
    )
    if not target_ids:
        raise ValueError("Target không tạo ra token nào.")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    context_budget = (
        max_length
        - special_tokens
        - 2
        - len(target_ids)
    )
    if context_budget < 0:
        raise ValueError(
            "Target vượt MAX_LENGTH ngay cả khi bỏ toàn bộ context."
        )

    left_quota = context_budget // 2
    right_quota = context_budget - left_quota
    left_count = min(len(prefix_ids), left_quota)
    right_count = min(len(suffix_ids), right_quota)
    remaining = context_budget - left_count - right_count
    if remaining:
        add_left = min(len(prefix_ids) - left_count, remaining)
        left_count += add_left
        remaining -= add_left
    if remaining:
        right_count += min(len(suffix_ids) - right_count, remaining)

    left_context = prefix_ids[-left_count:] if left_count else []
    right_context = suffix_ids[:right_count] if right_count else []
    core_ids = (
        left_context
        + [target_start_id]
        + target_ids
        + [target_end_id]
        + right_context
    )
    input_ids = tokenizer.build_inputs_with_special_tokens(core_ids)
    if len(input_ids) > max_length:
        raise AssertionError(
            f"Target-centered window dài {len(input_ids)} > {max_length}."
        )
    if input_ids.count(target_start_id) != 1:
        raise ValueError("Target window không có đúng một [TGT].")
    if input_ids.count(target_end_id) != 1:
        raise ValueError("Target window không có đúng một [/TGT].")
    if input_ids.index(target_end_id) <= input_ids.index(target_start_id) + 1:
        raise ValueError("Không có target token nằm giữa hai marker.")
    return input_ids


def pool_target_features(
    input_ids: torch.Tensor,
    hidden_state: torch.Tensor,
    *,
    target_start_id: int,
    target_end_id: int,
) -> torch.Tensor:
    if input_ids.ndim != 2 or hidden_state.ndim != 3:
        raise ValueError("input_ids/hidden_state phải có batch dimension.")
    if hidden_state.shape[:2] != input_ids.shape:
        raise ValueError("Hidden state shape không khớp input IDs.")

    start_mask = input_ids.eq(target_start_id)
    end_mask = input_ids.eq(target_end_id)
    if not start_mask.sum(dim=1).eq(1).all():
        raise ValueError("Mỗi item phải có đúng một [TGT].")
    if not end_mask.sum(dim=1).eq(1).all():
        raise ValueError("Mỗi item phải có đúng một [/TGT].")

    start_pos = start_mask.to(torch.int64).argmax(dim=1)
    end_pos = end_mask.to(torch.int64).argmax(dim=1)
    if not (end_pos > start_pos + 1).all():
        raise ValueError("Target marker order/span không hợp lệ.")

    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(
        sequence_length,
        device=input_ids.device,
    ).unsqueeze(0)
    inside = (
        (positions > start_pos.unsqueeze(1))
        & (positions < end_pos.unsqueeze(1))
    )
    weights = inside.unsqueeze(-1).to(hidden_state.dtype)
    target_mean = (
        (hidden_state * weights).sum(dim=1)
        / weights.sum(dim=1).clamp_min(1.0)
    )
    row_ids = torch.arange(batch_size, device=input_ids.device)
    return torch.cat(
        [
            hidden_state[:, 0],
            hidden_state[row_ids, start_pos],
            target_mean,
            hidden_state[row_ids, end_pos],
        ],
        dim=-1,
    )


def predict_token_batches(
    models: NeuralModels,
    input_id_rows: Sequence[Sequence[int]],
) -> list[NeuralOutput]:
    outputs: list[NeuralOutput] = []
    for batch_start in range(
        0,
        len(input_id_rows),
        models.batch_size,
    ):
        rows = input_id_rows[
            batch_start : batch_start + models.batch_size
        ]
        features = [
            {
                "input_ids": [int(value) for value in row],
                "attention_mask": [1] * len(row),
                "token_type_ids": [0] * len(row),
            }
            for row in rows
        ]
        batch = models.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        if int(batch["input_ids"].shape[1]) > models.max_length:
            raise ValueError("Batch chứa input vượt MAX_LENGTH.")
        batch = {
            name: tensor.to(models.device)
            for name, tensor in batch.items()
        }

        with torch.inference_mode():
            classifier_output = models.classifier(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            pooled = pool_target_features(
                batch["input_ids"],
                classifier_output.hidden_states[-1],
                target_start_id=models.target_start_id,
                target_end_id=models.target_end_id,
            )
            mlp_logits = models.mlp(pooled.float())

        linear_rows = (
            classifier_output.logits.float().cpu().numpy()
        )
        mlp_rows = mlp_logits.float().cpu().numpy()
        outputs.extend(
            NeuralOutput(
                linear_logits=np.array(linear_row, copy=True),
                mlp_logits=np.array(mlp_row, copy=True),
            )
            for linear_row, mlp_row in zip(linear_rows, mlp_rows)
        )
    return outputs
