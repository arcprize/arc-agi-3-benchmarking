import pytest

from benchmarking import runtime_clients


class _FakeOpenAIClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeAnthropicClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.unit
class TestBuildModelRuntimeClient:
    def test_builds_openai_client_from_runtime_sdk(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(runtime_clients, "OpenAIClient", _FakeOpenAIClient)

        client = runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "openai-python"},
            client_config={
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            config_id="openai-config",
        )

        assert isinstance(client, _FakeOpenAIClient)
        assert client.kwargs == {
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-openai-key",
        }

    def test_reads_configured_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        monkeypatch.setattr(runtime_clients, "OpenAIClient", _FakeOpenAIClient)

        client = runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "openai-python"},
            client_config={"api_key_env": "OPENROUTER_API_KEY"},
            config_id="openrouter-config",
        )

        assert client.kwargs["api_key"] == "test-openrouter-key"

    def test_missing_api_key_raises_config_specific_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            runtime_clients.build_model_runtime_client(
                runtime_config={"sdk": "openai-python"},
                client_config={
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                },
                config_id="openai-config",
            )

        assert str(exc_info.value) == (
            "No OPENAI_API_KEY set. "
            "The selected model config 'openai-config' requires "
            "the OPENAI_API_KEY environment variable to be set in your .env file."
        )

    def test_blank_api_key_raises_config_specific_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")

        with pytest.raises(ValueError, match="No OPENAI_API_KEY set"):
            runtime_clients.build_model_runtime_client(
                runtime_config={"sdk": "openai-python"},
                client_config={"api_key_env": "OPENAI_API_KEY"},
                config_id="openai-config",
            )

    def test_missing_api_key_env_raises_clear_error(self):
        with pytest.raises(ValueError) as exc_info:
            runtime_clients.build_model_runtime_client(
                runtime_config={"sdk": "openai-python"},
                client_config={"api_key_env": "   "},
                config_id="bad-config",
            )

        assert (
            str(exc_info.value)
            == "Model config 'bad-config' is missing client.api_key_env."
        )

    def test_builds_anthropic_client_from_runtime_sdk(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setattr(
            runtime_clients,
            "AnthropicClient",
            _FakeAnthropicClient,
        )

        client = runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "anthropic-python"},
            client_config={"api_key_env": "ANTHROPIC_API_KEY"},
            config_id="anthropic-config",
        )

        assert isinstance(client, _FakeAnthropicClient)
        assert client.kwargs == {"api_key": "test-anthropic-key"}

    def test_anthropic_client_uses_default_anthropic_api_key_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setattr(
            runtime_clients,
            "AnthropicClient",
            _FakeAnthropicClient,
        )

        client = runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "anthropic-python"},
            client_config={},
            config_id="anthropic-config",
        )

        assert client.kwargs == {"api_key": "test-anthropic-key"}

    def test_missing_anthropic_api_key_raises_config_specific_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            runtime_clients.build_model_runtime_client(
                runtime_config={"sdk": "anthropic-python"},
                client_config={"api_key_env": "ANTHROPIC_API_KEY"},
                config_id="anthropic-config",
            )

        assert str(exc_info.value) == (
            "No ANTHROPIC_API_KEY set. "
            "The selected model config 'anthropic-config' requires "
            "the ANTHROPIC_API_KEY environment variable to be set in your .env file."
        )

    def test_unsupported_runtime_sdk_raises_clear_error(self):
        with pytest.raises(ValueError) as exc_info:
            runtime_clients.build_model_runtime_client(
                runtime_config={"sdk": "google-genai"},
                client_config={"api_key_env": "GOOGLE_API_KEY"},
                config_id="google-config",
            )

        assert str(exc_info.value) == (
            "Model config 'google-config' uses unsupported runtime "
            "(sdk='google-genai')."
        )
