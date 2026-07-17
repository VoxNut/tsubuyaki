"""Resolve a local model directory or a pinned Hugging Face snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from furigana_aid.config import Settings


@dataclass(frozen=True, slots=True)
class ResolvedModelSource:
    directory: Path
    configured_revision: str | None
    remote: bool


def resolve_model_source(settings: Settings) -> ResolvedModelSource:
    if settings.model_local_dir is not None:
        directory = settings.model_local_dir.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"FURIGANA_MODEL_LOCAL_DIR không tồn tại: {directory}"
            )
        return ResolvedModelSource(
            directory=directory,
            configured_revision=None,
            remote=False,
        )

    if settings.hf_model_repo is None or settings.hf_model_revision is None:
        raise RuntimeError(
            "Remote model thiếu repository hoặc immutable revision."
        )
    directory = Path(
        snapshot_download(
            repo_id=settings.hf_model_repo,
            revision=settings.hf_model_revision,
            token=settings.hf_token,
            cache_dir=str(settings.hf_cache_dir),
            local_files_only=False,
        )
    ).resolve()
    return ResolvedModelSource(
        directory=directory,
        configured_revision=settings.hf_model_revision,
        remote=True,
    )
