from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest
from pydantic import ValidationError

from src.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    ProviderType,
    Settings,
)
from src.providers import LLMClient, ProviderError, create_client

# The SDK only stores these objects, so mocks keep the tests independent of its HTTP library.
_REQUEST = MagicMock(name="request")


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def _sdk(content: str | None = "hello") -> MagicMock:
    sdk = MagicMock()
    message = MagicMock(content=content)
    sdk.chat.completions.create.return_value.choices = [MagicMock(message=message)]
    return sdk


def _client(sdk: MagicMock, **overrides: object) -> LLMClient:
    return LLMClient(_settings(**overrides).target_config(), client=sdk)


class TestConfigResolution:
    def test_defaults_are_local_ollama(self) -> None:
        target = _settings().target_config()
        assert target.provider is ProviderType.OLLAMA
        assert target.model == DEFAULT_OLLAMA_MODEL
        assert target.base_url == DEFAULT_OLLAMA_BASE_URL
        assert target.api_key.get_secret_value()  # SDK needs a non-empty placeholder

    def test_cli_arguments_override_settings(self) -> None:
        settings = _settings(llm_api_key="from-env", llm_model="env-model")
        target = settings.target_config(
            provider=ProviderType.API,
            model="cli-model",
            api_key="from-cli",
            base_url="https://example.test/v1",
        )
        assert (target.model, target.base_url) == (
            "cli-model",
            "https://example.test/v1",
        )
        assert target.api_key.get_secret_value() == "from-cli"

    def test_api_mode_uses_cloud_base_url_and_env_key(self) -> None:
        target = _settings(llm_provider="api", llm_api_key="sk-env").target_config()
        assert target.base_url == "https://api.openai.com/v1"
        assert target.api_key.get_secret_value() == "sk-env"

    def test_api_mode_without_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires an API key"):
            _settings().target_config(provider=ProviderType.API, model="gpt-4o-mini")

    def test_environment_variables_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "llama3.2:1b")
        monkeypatch.setenv("JUDGE_PROVIDER", "")  # blank line copied from .env.example
        settings = _settings()
        assert settings.target_config().model == "llama3.2:1b"
        assert settings.judge_provider is None

    def test_judge_inherits_target_by_default(self) -> None:
        settings = _settings()
        target = settings.target_config(model="custom", base_url="http://box:11434/v1")
        judge = settings.judge_config(target=target)
        assert (judge.provider, judge.model, judge.base_url) == (
            target.provider,
            "custom",
            "http://box:11434/v1",
        )

    def test_judge_model_can_differ_on_same_provider(self) -> None:
        settings = _settings(llm_provider="api", llm_api_key="sk-env", judge_model="gpt-4o")
        target = settings.target_config(model="gpt-4o-mini")
        judge = settings.judge_config(target=target)
        assert (target.model, judge.model) == ("gpt-4o-mini", "gpt-4o")
        assert judge.api_key.get_secret_value() == "sk-env"

    def test_cloud_judge_for_local_target_does_not_inherit_local_url(self) -> None:
        settings = _settings(judge_provider="api", judge_model="gpt-4o", judge_api_key="sk-j")
        judge = settings.judge_config(target=settings.target_config())
        assert judge.provider is ProviderType.API
        assert judge.base_url == "https://api.openai.com/v1"
        assert judge.api_key.get_secret_value() == "sk-j"

    def test_local_judge_for_cloud_target_does_not_inherit_cloud_model(self) -> None:
        settings = _settings(llm_provider="api", llm_api_key="sk-env")
        target = settings.target_config(model="gpt-4o-mini")
        judge = settings.judge_config(provider=ProviderType.OLLAMA, target=target)
        assert (judge.model, judge.base_url) == (
            DEFAULT_OLLAMA_MODEL,
            DEFAULT_OLLAMA_BASE_URL,
        )

    def test_target_and_judge_have_separate_token_budgets(self) -> None:
        settings = _settings()
        target = settings.target_config()
        judge = settings.judge_config(target=target)
        assert (target.max_tokens, judge.max_tokens) == (512, 2048)

    def test_token_budgets_are_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_TOKENS", "256")
        monkeypatch.setenv("JUDGE_MAX_TOKENS", "4096")
        settings = _settings()
        target = settings.target_config()
        assert target.max_tokens == 256
        assert settings.judge_config(target=target).max_tokens == 4096

    def test_api_key_is_not_exposed_in_repr(self) -> None:
        target = _settings(llm_provider="api", llm_api_key="sk-secret").target_config()
        assert "sk-secret" not in repr(target)
        assert "sk-secret" not in target.model_dump_json()


class TestLLMClient:
    def test_complete_sends_system_and_user_messages(self) -> None:
        sdk = _sdk("the answer")
        result = _client(sdk, max_tokens=99).complete(
            "attack", system_prompt="be safe", temperature=0.3
        )
        assert result.text == "the answer"
        assert result.latency_ms >= 0
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_OLLAMA_MODEL
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 99
        assert kwargs["messages"] == [
            {"role": "system", "content": "be safe"},
            {"role": "user", "content": "attack"},
        ]

    def test_system_message_is_omitted_when_not_given(self) -> None:
        sdk = _sdk()
        _client(sdk).complete("hi")
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.0

    def test_none_content_becomes_empty_text(self) -> None:
        assert _client(_sdk(None)).complete("hi").text == ""

    def test_empty_choices_raise(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.return_value.choices = []
        with pytest.raises(ProviderError, match="no choices"):
            _client(sdk).complete("hi")

    def test_connection_error_has_ollama_hint(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.side_effect = openai.APIConnectionError(request=_REQUEST)
        with pytest.raises(ProviderError, match="ollama serve") as excinfo:
            _client(sdk).complete("hi")
        assert f"ollama pull {DEFAULT_OLLAMA_MODEL}" in str(excinfo.value)

    def test_connection_error_in_api_mode_has_no_ollama_hint(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.side_effect = openai.APIConnectionError(request=_REQUEST)
        client = _client(sdk, llm_provider="api", llm_api_key="sk-x")
        with pytest.raises(ProviderError) as excinfo:
            client.complete("hi")
        assert "ollama" not in str(excinfo.value).lower()
        assert "https://api.openai.com/v1" in str(excinfo.value)

    def test_http_status_error_is_wrapped(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.side_effect = openai.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body=None
        )
        with pytest.raises(ProviderError, match="HTTP 401") as excinfo:
            _client(sdk, llm_provider="api", llm_api_key="sk-secret").complete("hi")
        assert "sk-secret" not in str(excinfo.value)

    def test_missing_ollama_model_suggests_pull(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.side_effect = openai.NotFoundError(
            "model not found", response=MagicMock(status_code=404), body=None
        )
        with pytest.raises(ProviderError, match="HTTP 404") as excinfo:
            _client(sdk).complete("hi")
        assert f"ollama pull {DEFAULT_OLLAMA_MODEL}" in str(excinfo.value)

    def test_timeout_is_wrapped(self) -> None:
        sdk = _sdk()
        sdk.chat.completions.create.side_effect = openai.APITimeoutError(request=_REQUEST)
        with pytest.raises(ProviderError):
            _client(sdk).complete("hi")

    def test_create_client_builds_sdk_from_config(self) -> None:
        config = _settings(request_timeout_seconds=7).target_config(base_url="http://box:1234/v1")
        client = create_client(config)
        assert client.model == DEFAULT_OLLAMA_MODEL
        assert str(client._client.base_url) == "http://box:1234/v1/"
        assert client._client.timeout == 7
