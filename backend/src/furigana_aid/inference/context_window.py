"""Target-centered token budgeting that never truncates the marked target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class ContextTokenizer(Protocol):
    unk_token_id: int | None

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Sequence[int]:
        """Tokenize a text fragment without padding."""

    def convert_tokens_to_ids(self, token: str) -> int:
        """Return the vocabulary ID of an additional special token."""

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return the number of model-level tokens such as CLS and SEP."""

    def build_inputs_with_special_tokens(
        self, token_ids_0: Sequence[int]
    ) -> Sequence[int]:
        """Wrap a single sequence with the model's required special tokens."""


class InvalidTargetSpan(ValueError):
    """The supplied character span does not identify a usable target."""


class TargetExceedsTokenBudget(ValueError):
    """The target and mandatory markers alone exceed ``max_length``."""


@dataclass(frozen=True, slots=True)
class TargetCenteredContext:
    """One unpadded model input and metadata needed for safe batching."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    target_start_position: int
    target_end_position: int
    left_context_tokens: int
    target_tokens: int
    right_context_tokens: int
    original_start: int
    original_end: int
    surface: str

    def as_model_input(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
        }


def mark_target(
    text: str,
    start: int,
    end: int,
    *,
    target_start_token: str = "[TGT]",
    target_end_token: str = "[/TGT]",
) -> str:
    """Build the notebook-compatible, untruncated marked string."""

    _validate_span(text, start, end)
    return (
        text[:start]
        + target_start_token
        + text[start:end]
        + target_end_token
        + text[end:]
    )


def _validate_span(text: str, start: int, end: int) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("Target offsets must be integers.")
    if start < 0 or end > len(text) or start >= end:
        raise InvalidTargetSpan(
            f"Invalid target span [{start}, {end}) for text length {len(text)}."
        )


def _allocate_context_budget(
    left_length: int, right_length: int, budget: int
) -> tuple[int, int]:
    """Split a budget around the target and reclaim unused side capacity."""

    if budget < 0:
        raise ValueError("Context budget cannot be negative.")
    left_take = min(left_length, budget // 2)
    right_take = min(right_length, budget - left_take)
    remaining = budget - left_take - right_take

    if remaining:
        additional_left = min(left_length - left_take, remaining)
        left_take += additional_left
        remaining -= additional_left
    if remaining:
        right_take += min(right_length - right_take, remaining)
    return left_take, right_take


def _marker_id(
    tokenizer: ContextTokenizer, token: str, *, label: str
) -> int:
    marker_id = int(tokenizer.convert_tokens_to_ids(token))
    if marker_id < 0:
        raise ValueError(f"{label} token {token!r} has an invalid ID.")
    if (
        tokenizer.unk_token_id is not None
        and marker_id == int(tokenizer.unk_token_id)
    ):
        raise ValueError(
            f"Tokenizer does not contain required {label} token {token!r}."
        )
    return marker_id


def build_target_centered_context(
    text: str,
    start: int,
    end: int,
    *,
    tokenizer: ContextTokenizer,
    max_length: int,
    target_start_token: str = "[TGT]",
    target_end_token: str = "[/TGT]",
) -> TargetCenteredContext:
    """Tokenize around a target while preserving both markers and all target IDs.

    The left and right fragments are tokenized separately.  Only their outer
    context is cropped; the tokenized target and its markers are mandatory.
    This avoids the notebook's unsafe default right-truncation behavior.
    """

    _validate_span(text, start, end)
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    left_ids = tuple(
        int(value)
        for value in tokenizer.encode(
            text[:start], add_special_tokens=False
        )
    )
    target_ids = tuple(
        int(value)
        for value in tokenizer.encode(
            text[start:end], add_special_tokens=False
        )
    )
    right_ids = tuple(
        int(value)
        for value in tokenizer.encode(
            text[end:], add_special_tokens=False
        )
    )
    if not target_ids:
        raise InvalidTargetSpan("The target produced no tokenizer IDs.")

    start_marker_id = _marker_id(
        tokenizer, target_start_token, label="target-start"
    )
    end_marker_id = _marker_id(
        tokenizer, target_end_token, label="target-end"
    )
    if start_marker_id == end_marker_id:
        raise ValueError("Target marker IDs must be distinct.")

    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    if special_count < 0:
        raise ValueError("Tokenizer reported a negative special-token count.")
    mandatory_length = special_count + len(target_ids) + 2
    if mandatory_length > max_length:
        raise TargetExceedsTokenBudget(
            "Target plus markers and model special tokens requires "
            f"{mandatory_length} tokens, exceeding max_length={max_length}."
        )

    context_budget = max_length - mandatory_length
    left_take, right_take = _allocate_context_budget(
        len(left_ids), len(right_ids), context_budget
    )
    core_ids = (
        left_ids[len(left_ids) - left_take :]
        + (start_marker_id,)
        + target_ids
        + (end_marker_id,)
        + right_ids[:right_take]
    )
    input_ids = tuple(
        int(value)
        for value in tokenizer.build_inputs_with_special_tokens(core_ids)
    )
    if len(input_ids) > max_length:
        raise ValueError(
            "Tokenizer added more model special tokens than it reported."
        )
    if len(input_ids) != len(core_ids) + special_count:
        raise ValueError(
            "Tokenizer special-token count is inconsistent with built input."
        )
    if input_ids.count(start_marker_id) != 1:
        raise ValueError("Built input must contain exactly one start marker.")
    if input_ids.count(end_marker_id) != 1:
        raise ValueError("Built input must contain exactly one end marker.")

    start_position = input_ids.index(start_marker_id)
    end_position = input_ids.index(end_marker_id)
    if end_position - start_position - 1 != len(target_ids):
        raise ValueError("Built input did not preserve the complete target.")

    return TargetCenteredContext(
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        target_start_position=start_position,
        target_end_position=end_position,
        left_context_tokens=left_take,
        target_tokens=len(target_ids),
        right_context_tokens=right_take,
        original_start=start,
        original_end=end,
        surface=text[start:end],
    )
