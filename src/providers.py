"""Unified LLM client.

Both local Ollama (``/v1``) and cloud providers expose an OpenAI-compatible API,
so a single adapter built on the official ``openai`` SDK covers every mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import openai
from openai import OpenAI

from src.config import ProviderConfig, ProviderType


class ProviderError(Exception):
    """Raised when a model call fails or returns no usable content."""


@dataclass(frozen=True)
class CompletionResult:
    """Model output plus wall-clock latency of the request."""

    text: str
    latency_ms: float


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(self, config: ProviderConfig, client: OpenAI | None = None) -> None:
        self.config = config
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
            # Fail fast and let the runner decide how to record the error.
            max_retries=1,
        )

    @property
    def model(self) -> str:
        return self.config.model

    def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send a single-turn chat request and return the assistant text."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=self.config.max_tokens,
            )
        except openai.APIConnectionError as exc:
            raise ProviderError(self._connection_hint(exc)) from exc
        except openai.APIStatusError as exc:
            message = (
                f"{self.config.provider.value} returned HTTP {exc.status_code} "
                f"for model '{self.config.model}': {exc.message}"
            )
            if self.config.provider is ProviderType.OLLAMA and exc.status_code == 404:
                message += (
                    f" - pull it with `ollama pull {self.config.model}` "
                    "or choose an installed model (`ollama list`) via --model."
                )
            raise ProviderError(message) from exc
        except openai.OpenAIError as exc:
            raise ProviderError(f"LLM request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if not response.choices:
            raise ProviderError("LLM response contained no choices")
        # Content can be None (e.g. refusals / tool calls); treat it as empty text.
        text = response.choices[0].message.content or ""
        return CompletionResult(text=text, latency_ms=latency_ms)

    def _connection_hint(self, exc: Exception) -> str:
        if self.config.provider is ProviderType.OLLAMA:
            return (
                f"Cannot reach Ollama at {self.config.base_url}. "
                f"Is the daemon running (`ollama serve`) and the model pulled "
                f"(`ollama pull {self.config.model}`)? ({exc})"
            )
        return f"Cannot reach {self.config.base_url}: {exc}"


def create_client(config: ProviderConfig) -> LLMClient:
    """Build an :class:`LLMClient` for a resolved provider config."""
    return LLMClient(config)
