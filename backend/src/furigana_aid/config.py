"""Environment-only application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings.

    Model visibility is deliberately absent: a private repository supplies
    ``HF_TOKEN`` while a public repository simply leaves it unset.
    """

    model_config = SettingsConfigDict(
        env_prefix="FURIGANA_",
        env_file="backend/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_local_dir: Path | None = None
    hf_model_repo: str | None = None
    hf_model_revision: str | None = None
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "FURIGANA_HF_TOKEN"),
        repr=False,
    )
    hf_cache_dir: Path = Path("/tmp/huggingface")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    inference_batch_size: int = Field(default=16, ge=1, le=128)

    max_cues_per_request: int = Field(default=64, ge=1, le=512)
    max_chars_per_cue: int = Field(default=2_000, ge=1, le=20_000)
    max_total_chars: int = Field(default=32_000, ge=1, le=500_000)
    cors_origins: tuple[str, ...] = (
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    )
    log_level: str = "INFO"

    @field_validator("hf_model_repo", "hf_model_revision", mode="before")
    @classmethod
    def blank_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_local_dir", mode="before")
    @classmethod
    def blank_path_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            return tuple(
                item.strip()
                for item in value.split(",")
                if item.strip()
            )
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if normalized not in allowed:
            raise ValueError(
                f"log_level phải thuộc {sorted(allowed)}, nhận {value!r}."
            )
        return normalized

    @model_validator(mode="after")
    def validate_model_source(self) -> "Settings":
        if self.model_local_dir is not None:
            return self
        if self.hf_model_repo is None:
            raise ValueError(
                "Cần FURIGANA_MODEL_LOCAL_DIR hoặc "
                "FURIGANA_HF_MODEL_REPO."
            )
        if self.hf_model_revision is None:
            raise ValueError(
                "Remote model phải pin bằng "
                "FURIGANA_HF_MODEL_REVISION."
            )
        return self

    def redacted_summary(self) -> dict[str, object]:
        return {
            "model_local_dir": (
                str(self.model_local_dir)
                if self.model_local_dir is not None
                else None
            ),
            "hf_model_repo": self.hf_model_repo,
            "hf_model_revision": self.hf_model_revision,
            "hf_token_configured": bool(self.hf_token),
            "device": self.device,
            "inference_batch_size": self.inference_batch_size,
        }
