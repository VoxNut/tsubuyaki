"""Target-aware feature pooling for the notebook's MLP architecture."""

from __future__ import annotations

from typing import Any


class TargetMarkerError(RuntimeError):
    """A tokenized row does not contain one valid marked target span."""


def pool_target_features(
    input_ids: Any,
    final_hidden_state: Any,
    *,
    target_start_id: int,
    target_end_id: int,
) -> Any:
    """Concatenate ``CLS + start marker + target mean + end marker``.

    ``final_hidden_state`` must be the final encoder layer with shape
    ``[batch, sequence, hidden]``.  This is four *positions* from one layer,
    not a concatenation of the last four BERT layers.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency startup failure
        raise RuntimeError(
            "pool_target_features requires PyTorch to be installed."
        ) from exc

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")
    if not isinstance(final_hidden_state, torch.Tensor):
        raise TypeError("final_hidden_state must be a torch.Tensor.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if final_hidden_state.ndim != 3:
        raise ValueError(
            "final_hidden_state must have shape [batch, sequence, hidden]."
        )
    if tuple(input_ids.shape) != tuple(final_hidden_state.shape[:2]):
        raise ValueError(
            "input_ids and final_hidden_state batch/sequence shapes differ."
        )
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Feature pooling cannot operate on an empty batch.")
    if int(target_start_id) == int(target_end_id):
        raise ValueError("Target marker IDs must be distinct.")

    start_mask = input_ids.eq(int(target_start_id))
    end_mask = input_ids.eq(int(target_end_id))
    start_counts = start_mask.sum(dim=1)
    end_counts = end_mask.sum(dim=1)
    invalid_count_rows = torch.nonzero(
        start_counts.ne(1) | end_counts.ne(1), as_tuple=False
    ).flatten()
    if invalid_count_rows.numel():
        rows = invalid_count_rows.detach().cpu().tolist()
        raise TargetMarkerError(
            "Each row must contain exactly one [TGT] and one [/TGT] marker; "
            f"invalid rows: {rows}."
        )

    start_positions = start_mask.to(torch.int64).argmax(dim=1)
    end_positions = end_mask.to(torch.int64).argmax(dim=1)
    invalid_order_rows = torch.nonzero(
        end_positions <= start_positions + 1, as_tuple=False
    ).flatten()
    if invalid_order_rows.numel():
        rows = invalid_order_rows.detach().cpu().tolist()
        raise TargetMarkerError(
            "Every marked target must contain at least one token between "
            f"ordered markers; invalid rows: {rows}."
        )

    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(
        sequence_length, device=input_ids.device
    ).unsqueeze(0)
    inside_mask = (
        positions > start_positions.unsqueeze(1)
    ) & (
        positions < end_positions.unsqueeze(1)
    )
    inside_weights = inside_mask.unsqueeze(-1).to(
        dtype=final_hidden_state.dtype
    )
    target_mean = (
        (final_hidden_state * inside_weights).sum(dim=1)
        / inside_weights.sum(dim=1)
    )

    batch_indices = torch.arange(batch_size, device=input_ids.device)
    cls_vector = final_hidden_state[:, 0]
    start_vector = final_hidden_state[batch_indices, start_positions]
    end_vector = final_hidden_state[batch_indices, end_positions]
    return torch.cat(
        (cls_vector, start_vector, target_mean, end_vector), dim=-1
    )
