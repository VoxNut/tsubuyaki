from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from furigana_aid.inference.features import (  # noqa: E402
    TargetMarkerError,
    pool_target_features,
)


def test_pooling_concatenates_four_positions_from_final_hidden_state() -> None:
    input_ids = torch.tensor([[101, 900, 10, 11, 901, 102]])
    hidden = torch.tensor(
        [
            [
                [0.0, 0.5],
                [1.0, 1.5],
                [2.0, 2.5],
                [3.0, 3.5],
                [4.0, 4.5],
                [5.0, 5.5],
            ]
        ]
    )

    features = pool_target_features(
        input_ids,
        hidden,
        target_start_id=900,
        target_end_id=901,
    )

    assert features.shape == (1, 8)
    assert torch.equal(
        features,
        torch.tensor(
            [[0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 4.0, 4.5]]
        ),
    )


@pytest.mark.parametrize(
    "input_ids",
    [
        [[101, 10, 901, 102]],  # no start marker
        [[101, 900, 10, 102]],  # no end marker
        [[101, 900, 10, 900, 901, 102]],  # duplicate start
        [[101, 901, 10, 900, 102]],  # reversed
        [[101, 900, 901, 102]],  # empty target
    ],
)
def test_pooling_rejects_invalid_marker_layout(input_ids) -> None:
    ids = torch.tensor(input_ids)
    hidden = torch.zeros((*ids.shape, 3))

    with pytest.raises(TargetMarkerError):
        pool_target_features(
            ids,
            hidden,
            target_start_id=900,
            target_end_id=901,
        )
