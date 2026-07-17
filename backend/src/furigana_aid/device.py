"""CPU-first device selection."""

from __future__ import annotations

from typing import Literal

import torch


DevicePreference = Literal["auto", "cpu", "cuda"]


def resolve_device(preference: DevicePreference) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FURIGANA_DEVICE=cuda nhưng CUDA không khả dụng. "
                "Kiểm tra GPU, driver và PyTorch CUDA wheel."
            )
        return torch.device("cuda")
    if preference == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    raise ValueError(f"Device preference không hỗ trợ: {preference!r}.")
