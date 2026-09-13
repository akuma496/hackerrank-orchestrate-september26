"""Typed runtime settings loaded from environment variables and an optional ``.env`` file.

Secrets are wrapped in :class:`pydantic.SecretStr`, so ``repr``/``str``/JSON never reveal them.
No data APIs are configured: every financial input comes from ``dataset/``.
"""

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from dotenv import dotenv_values
from pydantic import SecretStr, field_validator, model_validator

from buy_or_wait.observability.structured_logging import LogLevel
from buy_or_wait.schemas.base import StrictModel
from buy_or_wait.schemas.primitives import ShortText
from buy_or_wait.schemas.state import MAX_REFLECTION_LIMIT, MAX_STEP_LIMIT
from buy_or_wait.schemas.tools import DEFAULT_HORIZON_DAYS, MAX_FORECAST_DAYS

CODE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = CODE_ROOT.parent
ENV_FILE_VARIABLE: Final[str] = "BOW_ENV_FILE"


class LlmProvider(StrEnum):
    NONE = "none"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class AppSettings(StrictModel):
    dataset_dir: Path
    output_path: Path
    llm_cache_dir: Path
    log_level: LogLevel
    log_file: Path | None
    llm_provider: LlmProvider
    anthropic_api_key: SecretStr | None
    gemini_api_key: SecretStr | None
    planner_model: ShortText
    evidence_model: ShortText
    max_reflections: int
    max_planner_steps: int
    forecast_horizon_days: int

    @field_validator("max_reflections")
    @classmethod
    def _bounded_reflections(cls, value: int) -> int:
        if not 0 <= value <= MAX_REFLECTION_LIMIT:
            raise ValueError(f"BOW_MAX_REFLECTIONS must be within 0..{MAX_REFLECTION_LIMIT}")
        return value

    @field_validator("max_planner_steps")
    @classmethod
    def _bounded_steps(cls, value: int) -> int:
        if not 1 <= value <= MAX_STEP_LIMIT:
            raise ValueError(f"BOW_MAX_PLANNER_STEPS must be within 1..{MAX_STEP_LIMIT}")
        return value

    @field_validator("forecast_horizon_days")
    @classmethod
    def _bounded_horizon(cls, value: int) -> int:
        if not 1 <= value <= MAX_FORECAST_DAYS:
            raise ValueError(f"BOW_FORECAST_HORIZON_DAYS must be within 1..{MAX_FORECAST_DAYS}")
        return value

    @model_validator(mode="after")
    def _provider_has_key(self) -> Self:
        if self.llm_provider is LlmProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when BOW_LLM_PROVIDER=anthropic")
        if self.llm_provider is LlmProvider.GEMINI and self.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required when BOW_LLM_PROVIDER=gemini")
        return self

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None, env_file: Path | None = None
    ) -> Self:
        """Merge ``.env`` (lowest precedence) with the process environment (highest)."""
        process_env = dict(os.environ if environ is None else environ)
        candidate = env_file or Path(process_env.get(ENV_FILE_VARIABLE, REPO_ROOT / ".env"))
        file_values = (
            {key: value for key, value in dotenv_values(candidate).items() if value is not None}
            if candidate.is_file()
            else {}
        )
        values = {**file_values, **process_env}

        def text(key: str, default: str) -> str:
            raw = values.get(key, "").strip()
            return raw or default

        def secret(key: str) -> SecretStr | None:
            raw = values.get(key, "").strip()
            return SecretStr(raw) if raw else None

        def path(key: str, default: str) -> Path:
            resolved = Path(text(key, default))
            return resolved if resolved.is_absolute() else REPO_ROOT / resolved

        def integer(key: str, default: int) -> int:
            raw = text(key, str(default))
            if not raw.isdigit():
                raise ValueError(f"{key} must be a non-negative integer")
            return int(raw)

        log_file_text = text("BOW_LOG_FILE", "")
        return cls(
            dataset_dir=path("BOW_DATASET_DIR", "dataset"),
            output_path=path("BOW_OUTPUT_PATH", "output.csv"),
            llm_cache_dir=path("BOW_LLM_CACHE_DIR", "code/.cache/llm"),
            log_level=LogLevel(text("BOW_LOG_LEVEL", LogLevel.INFO.value).upper()),
            log_file=path("BOW_LOG_FILE", log_file_text) if log_file_text else None,
            llm_provider=LlmProvider(text("BOW_LLM_PROVIDER", LlmProvider.NONE.value).lower()),
            anthropic_api_key=secret("ANTHROPIC_API_KEY"),
            gemini_api_key=secret("GEMINI_API_KEY"),
            planner_model=text("BOW_PLANNER_MODEL", "claude-opus-5"),
            evidence_model=text("BOW_EVIDENCE_MODEL", "claude-opus-5"),
            max_reflections=integer("BOW_MAX_REFLECTIONS", 2),
            max_planner_steps=integer("BOW_MAX_PLANNER_STEPS", 40),
            forecast_horizon_days=integer("BOW_FORECAST_HORIZON_DAYS", DEFAULT_HORIZON_DAYS),
        )
