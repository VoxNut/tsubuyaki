from __future__ import annotations

import pytest

from furigana_aid.inference.context_window import (
    InvalidTargetSpan,
    TargetExceedsTokenBudget,
    build_target_centered_context,
    mark_target,
)


class CharacterTokenizer:
    unk_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [1000 + ord(character) for character in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"[TGT]": 900, "[/TGT]": 901}.get(token, 0)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, token_ids_0):
        return [101, *token_ids_0, 102]


@pytest.mark.parametrize(
    ("text", "start", "end", "expected_left", "expected_right"),
    [
        ("ABabcdefgh", 0, 2, 0, 4),
        ("abcdEFghij", 4, 6, 2, 2),
        ("abcdefghIJ", 8, 10, 4, 0),
    ],
)
def test_target_centering_reclaims_budget_at_start_middle_and_end(
    text: str,
    start: int,
    end: int,
    expected_left: int,
    expected_right: int,
) -> None:
    context = build_target_centered_context(
        text,
        start,
        end,
        tokenizer=CharacterTokenizer(),
        max_length=10,
    )

    assert len(context.input_ids) == 10
    assert context.left_context_tokens == expected_left
    assert context.right_context_tokens == expected_right
    assert context.input_ids[context.target_start_position] == 900
    assert context.input_ids[context.target_end_position] == 901
    assert (
        context.target_end_position - context.target_start_position
        == context.target_tokens + 1
    )
    assert context.surface == text[start:end]


def test_multi_token_target_is_never_partially_truncated() -> None:
    context = build_target_centered_context(
        "abcdXYZefgh",
        4,
        7,
        tokenizer=CharacterTokenizer(),
        max_length=9,
    )

    between_markers = context.input_ids[
        context.target_start_position + 1 : context.target_end_position
    ]
    assert between_markers == tuple(
        1000 + ord(character) for character in "XYZ"
    )
    assert context.target_tokens == 3
    assert context.left_context_tokens + context.right_context_tokens == 2


def test_target_larger_than_budget_fails_instead_of_losing_marker() -> None:
    with pytest.raises(TargetExceedsTokenBudget, match="exceeding max_length"):
        build_target_centered_context(
            "ABCDEFGHIJ",
            0,
            10,
            tokenizer=CharacterTokenizer(),
            max_length=10,
        )


def test_invalid_offsets_and_missing_marker_are_rejected() -> None:
    with pytest.raises(InvalidTargetSpan):
        build_target_centered_context(
            "abc",
            2,
            2,
            tokenizer=CharacterTokenizer(),
            max_length=10,
        )

    tokenizer = CharacterTokenizer()
    tokenizer.convert_tokens_to_ids = lambda token: 0  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="does not contain required"):
        build_target_centered_context(
            "abc",
            1,
            2,
            tokenizer=tokenizer,
            max_length=10,
        )


def test_mark_target_uses_original_character_offsets() -> None:
    assert mark_target("今日は日曜日", 3, 6) == (
        "今日は[TGT]日曜日[/TGT]"
    )
