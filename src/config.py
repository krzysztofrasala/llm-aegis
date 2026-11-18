"""Configuration and provider settings for LLM Aegis.

Settings are loaded from environment variables / a ``.env`` file and can be
overridden at runtime (for example by CLI flags) via :meth:`Settings.target_config`
and :meth:`Settings.judge_config`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
# Ollama ignores the key, but the OpenAI SDK requires a non-empty value.
OLLAMA_PLACEHOLDER_API_KEY = "ollama"


class ProviderType(StrEnum):
    """Supported execution modes."""

    OLLAMA = "ollama"
    API = "api"


class ProviderConfig(BaseModel):
    """Resolved connection settings for a single model (target or judge)."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderType
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key: SecretStr
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_tokens: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def _require_key_for_cloud(self) -> ProviderConfig:
        if self.provider is ProviderType.API and not self.api_key.get_secret_value().strip():
            raise ValueError(
                "Cloud API mode requires an API key "
                "(set LLM_API_KEY / JUDGE_API_KEY or pass --api-key)."
            )
        return self


class Settings(BaseSettings):
    """Application settings sourced from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Target model (system under test)
    llm_provider: ProviderType = ProviderType.OLLAMA
    llm_model: str = DEFAULT_OLLAMA_MODEL
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL

    # Judge model (evaluator). Empty values fall back to the target settings.
    judge_provider: ProviderType | None = None
    judge_model: str = ""
    judge_api_key: SecretStr = SecretStr("")
    judge_base_url: str = ""

    # Request tuning
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    # Judges are often reasoning models (gpt-oss, qwen) that spend tokens
    # "thinking" before the verdict, so they get a larger budget by default.
    judge_max_tokens: int = Field(default=2048, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _empty_strings_to_none(cls, data: object) -> object:
        """Treat blank optional enum values from ``.env`` (``JUDGE_PROVIDER=``) as unset."""
        if isinstance(data, dict) and data.get("judge_provider") == "":
            data = {**data, "judge_provider": None}
        return data

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #
    def target_config(
        self,
        *,
        provider: ProviderType | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> ProviderConfig:
        """Build the target model config; explicit arguments override settings."""
        provider = provider or self.llm_provider
        return self._build(
            provider=provider,
            model=model or self.llm_model,
            api_key=api_key or self.llm_api_key.get_secret_value(),
            base_url=base_url or self._default_base_url(provider),
            max_tokens=self.max_tokens,
        )

    def judge_config(
        self,
        *,
        provider: ProviderType | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        target: ProviderConfig | None = None,
    ) -> ProviderConfig:
        """Build the judge model config.

        Precedence for every field: explicit argument > ``JUDGE_*`` setting >
        the resolved target config (only when the judge uses the same provider
        as the target, so one local model can judge itself with zero config).
        """
        target = target or self.target_config()
        provider = provider or self.judge_provider or target.provider

        if provider is target.provider:
            default_model = target.model
            default_key = target.api_key.get_secret_value()
            default_url = target.base_url
        else:
            default_model = DEFAULT_OLLAMA_MODEL if provider is ProviderType.OLLAMA else ""
            default_key = self.llm_api_key.get_secret_value()
            default_url = self._default_base_url(provider)

        return self._build(
            provider=provider,
            model=model or self.judge_model or default_model,
            api_key=api_key or self.judge_api_key.get_secret_value() or default_key,
            base_url=base_url or self.judge_base_url or default_url,
            max_tokens=self.judge_max_tokens,
        )

    def _default_base_url(self, provider: ProviderType) -> str:
        return self.ollama_base_url if provider is ProviderType.OLLAMA else self.llm_base_url

    def _build(
        self,
        *,
        provider: ProviderType,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
    ) -> ProviderConfig:
        if provider is ProviderType.OLLAMA:
            api_key = api_key or OLLAMA_PLACEHOLDER_API_KEY
        return ProviderConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            timeout_seconds=self.request_timeout_seconds,
            max_tokens=max_tokens,
        )


def get_settings() -> Settings:
    """Load settings from the environment (a fresh instance on every call)."""
    return Settings()
