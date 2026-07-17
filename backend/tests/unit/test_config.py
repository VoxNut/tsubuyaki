from __future__ import annotations

import pytest
from pydantic import ValidationError

from furigana_aid.config import Settings


def _configure_remote(monkeypatch: pytest.MonkeyPatch, revision: str) -> None:
    monkeypatch.delenv("FURIGANA_MODEL_LOCAL_DIR", raising=False)
    monkeypatch.setenv("FURIGANA_HF_MODEL_REPO", "owner/private-model")
    monkeypatch.setenv("FURIGANA_HF_MODEL_REVISION", revision)


def test_remote_revision_accepts_immutable_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "A" * 40
    _configure_remote(monkeypatch, revision)

    settings = Settings(_env_file=None)

    assert settings.hf_model_revision == revision.lower()


def test_remote_revision_rejects_branch_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_remote(monkeypatch, "main")

    with pytest.raises(ValidationError, match="commit SHA 40 ký tự"):
        Settings(_env_file=None)
