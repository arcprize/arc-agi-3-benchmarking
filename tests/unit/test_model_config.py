from pathlib import Path

import pytest
import yaml

import benchmarking.model_config as model_config


def _write_model_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configs: list[dict],
) -> Path:
    config_path = tmp_path / "model_configs.yaml"
    config_path.write_text(yaml.safe_dump(configs, sort_keys=False))
    monkeypatch.setattr(model_config, "MODEL_CONFIG_PATH", config_path)
    return config_path


def _valid_config(
    config_id: str,
    *,
    runtime_sdk: str = "openai-python",
    runtime_api: str = "chat_completions",
    **overrides: dict,
) -> dict:
    config = {
        "id": config_id,
        "runtime": {
            "sdk": runtime_sdk,
            "api": runtime_api,
            "state": "manual_rolling",
        },
        "client": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "request": {
            "model": "gpt-5.4-2026-03-05",
            "max_completion_tokens": 128_000,
        },
        "agent": {"MAX_CONTEXT_LENGTH": 175_000},
        "pricing": {"input": 2.50, "output": 15.00},
    }
    config.update(overrides)
    return config


@pytest.mark.unit
class TestModelConfig:
    def test_load_model_configs_reads_id(self, tmp_path, monkeypatch):
        _write_model_configs(tmp_path, monkeypatch, [_valid_config("chat-config")])

        configs = model_config.load_model_configs()

        assert configs[0]["id"] == "chat-config"

    def test_load_model_configs_accepts_responses_api(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("responses-config", runtime_api="responses")],
        )

        configs = model_config.load_model_configs()

        assert configs[0]["runtime"]["api"] == "responses"

    def test_load_model_configs_accepts_anthropic_messages_runtime_pair(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "anthropic-messages-config",
                    runtime_sdk="anthropic-python",
                    runtime_api="messages",
                    client={"api_key_env": "ANTHROPIC_API_KEY"},
                    request={"model": "claude-opus-4-6", "max_tokens": 128_000},
                )
            ],
        )

        configs = model_config.load_model_configs()

        assert configs[0]["runtime"] == {
            "sdk": "anthropic-python",
            "api": "messages",
            "state": "manual_rolling",
        }

    @pytest.mark.parametrize(
        ("runtime_sdk", "runtime_api"),
        [
            ("anthropic-python", "chat_completions"),
            ("openai-python", "messages"),
        ],
    )
    def test_load_model_configs_rejects_unsupported_runtime_pairs(
        self,
        tmp_path,
        monkeypatch,
        runtime_sdk,
        runtime_api,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "bad-runtime-pair",
                    runtime_sdk=runtime_sdk,
                    runtime_api=runtime_api,
                )
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            model_config.load_model_configs()

        message = str(exc_info.value)
        assert (
            f"Model config 'bad-runtime-pair' uses unsupported runtime "
            f"(sdk={runtime_sdk!r}, api={runtime_api!r})."
        ) in message
        assert "(sdk='anthropic-python', api='messages')" in message
        assert "(sdk='openai-python', api='chat_completions')" in message
        assert "(sdk='openai-python', api='responses')" in message

    @pytest.mark.parametrize(
        "client_field",
        [
            "base_url",
        ],
    )
    def test_load_model_configs_rejects_openai_compat_client_fields_for_native_anthropic(
        self,
        tmp_path,
        monkeypatch,
        client_field,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "bad-anthropic-client-field",
                    runtime_sdk="anthropic-python",
                    runtime_api="messages",
                    client={
                        "api_key_env": "ANTHROPIC_API_KEY",
                        client_field: "https://api.anthropic.com/v1/",
                    },
                    request={"model": "claude-opus-4-7", "max_tokens": 20_000},
                )
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            model_config.load_model_configs()

        message = str(exc_info.value)
        assert "OpenAI-compatible client field(s)" in message
        assert client_field in message

    @pytest.mark.parametrize(
        "request_field",
        [
            "extra_body",
            "max_completion_tokens",
            "max_output_tokens",
        ],
    )
    def test_load_model_configs_rejects_openai_compat_request_fields_for_native_anthropic(
        self,
        tmp_path,
        monkeypatch,
        request_field,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "bad-anthropic-request-field",
                    runtime_sdk="anthropic-python",
                    runtime_api="messages",
                    client={"api_key_env": "ANTHROPIC_API_KEY"},
                    request={
                        "model": "claude-opus-4-7",
                        "max_tokens": 20_000,
                        request_field: {},
                    },
                )
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            model_config.load_model_configs()

        message = str(exc_info.value)
        assert "OpenAI-compatible request field(s)" in message
        assert request_field in message

    def test_load_model_configs_rejects_missing_runtime_sdk(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "missing-runtime-sdk",
                    runtime={
                        "api": "chat_completions",
                        "state": "manual_rolling",
                    },
                )
            ],
        )

        with pytest.raises(ValueError, match="missing runtime.sdk"):
            model_config.load_model_configs()

    def test_load_model_configs_rejects_missing_runtime_api(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "missing-runtime-api",
                    runtime={
                        "sdk": "openai-python",
                        "state": "manual_rolling",
                    },
                )
            ],
        )

        with pytest.raises(ValueError, match="missing runtime.api"):
            model_config.load_model_configs()

    def test_load_model_configs_rejects_invalid_runtime_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "bad-runtime-state",
                    runtime={
                        "sdk": "openai-python",
                        "api": "chat_completions",
                        "state": "previous_response_id",
                    },
                )
            ],
        )

        with pytest.raises(
            ValueError,
            match="uses runtime.state='previous_response_id'",
        ):
            model_config.load_model_configs()

    def test_load_model_configs_accepts_boolean_analysis_mode(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "analysis-config",
                    agent={"MAX_CONTEXT_LENGTH": 175_000, "analysis_mode": True},
                )
            ],
        )

        configs = model_config.load_model_configs()

        assert configs[0]["agent"]["analysis_mode"] is True

    def test_load_model_configs_allows_omitted_analysis_mode(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("normal-config", agent={"MAX_CONTEXT_LENGTH": 175_000})],
        )

        configs = model_config.load_model_configs()

        assert "analysis_mode" not in configs[0]["agent"]

    def test_load_model_configs_rejects_non_boolean_analysis_mode(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config(
                    "bad-analysis-config",
                    agent={"MAX_CONTEXT_LENGTH": 175_000, "analysis_mode": "true"},
                )
            ],
        )

        with pytest.raises(
            ValueError,
            match="agent.analysis_mode must be a boolean",
        ):
            model_config.load_model_configs()

    def test_load_model_configs_rejects_non_mapping_agent_section(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("bad-agent-section", agent=True)],
        )

        with pytest.raises(
            ValueError,
            match="section 'agent' must be a mapping",
        ):
            model_config.load_model_configs()

    def test_load_model_configs_rejects_entries_missing_id(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                {
                    "runtime": {
                        "sdk": "openai-python",
                        "api": "chat_completions",
                        "state": "manual_rolling",
                    },
                    "client": {"api_key_env": "OPENAI_API_KEY"},
                    "request": {"model": "gpt-5.4"},
                }
            ],
        )

        with pytest.raises(ValueError, match="missing required 'id'"):
            model_config.load_model_configs()

    def test_load_model_configs_rejects_duplicate_ids(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config("duplicate-id"),
                _valid_config("duplicate-id"),
            ],
        )

        with pytest.raises(ValueError, match="Duplicate model config id 'duplicate-id'"):
            model_config.load_model_configs()

    def test_list_model_config_ids_returns_id_values_not_name(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("chat-config", name="legacy-display-name")],
        )

        assert model_config.list_model_config_ids() == ["chat-config"]

    def test_get_model_config_by_id_succeeds(self, tmp_path, monkeypatch):
        _write_model_configs(tmp_path, monkeypatch, [_valid_config("lookup-success")])

        config = model_config.get_model_config("lookup-success")

        assert config["id"] == "lookup-success"
        assert config["request"]["model"] == "gpt-5.4-2026-03-05"

    def test_get_model_config_by_missing_id_lists_available_ids(
        self,
        tmp_path,
        monkeypatch,
    ):
        config_path = _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config("available-a"),
                _valid_config("available-b"),
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            model_config.get_model_config("missing-id")

        message = str(exc_info.value)
        assert f"Model config 'missing-id' not found in {config_path}" in message
        assert "Available configs: available-a, available-b" in message

    def test_load_model_configs_requires_runtime_section(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("missing-runtime", runtime=None)],
        )

        with pytest.raises(ValueError, match="missing required section 'runtime'"):
            model_config.load_model_configs()

    def test_load_model_configs_requires_client_section(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("missing-client", client=None)],
        )

        with pytest.raises(ValueError, match="missing required section 'client'"):
            model_config.load_model_configs()

    def test_load_model_configs_requires_request_section(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [_valid_config("missing-request", request=None)],
        )

        with pytest.raises(ValueError, match="missing required section 'request'"):
            model_config.load_model_configs()

    def test_legacy_name_only_entry_fails_clearly(self, tmp_path, monkeypatch):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                {
                    "name": "legacy-only",
                    "runtime": {
                        "sdk": "openai-python",
                        "api": "chat_completions",
                        "state": "manual_rolling",
                    },
                    "client": {"api_key_env": "OPENAI_API_KEY"},
                    "request": {"model": "gpt-5.4"},
                }
            ],
        )

        with pytest.raises(
            ValueError,
            match="uses legacy field 'name'. Rename it to 'id'",
        ):
            model_config.load_model_configs()

    def test_checked_in_config_ids_include_chat_and_responses_models(self):
        config_ids = set(model_config.list_model_config_ids())

        assert {
            "anthropic-opus-4-7-medium",
            "anthropic-opus-4-7-low-thinking",
            "google-gemini-3-1-pro-preview",
            "openai-gpt-5-4-2026-03-05",
            "openai-gpt-5-4-2026-03-05-high",
            "openai-gpt-5-4-2026-03-05-low",
            "openai-gpt-5-4-2026-03-05-responses",
            "openai-gpt-5-4-2026-03-05-responses-analysis",
            "openai-gpt-5-4-2026-03-05-xhigh",
            "openai-gpt-5.4-openrouter",
            "xai-grok-4-20-beta-0309-reasoning",
        } <= config_ids

    @pytest.mark.parametrize(
        "config_id",
        [
            "anthropic-opus-4-7-medium",
            "anthropic-opus-4-7-low-thinking",
        ],
    )
    def test_checked_in_anthropic_configs_use_native_runtime(self, config_id):
        config = model_config.get_model_config(config_id)

        assert config["id"] == config_id
        assert config["runtime"] == {
            "sdk": "anthropic-python",
            "api": "messages",
            "state": "manual_rolling",
        }
        assert config["client"] == {"api_key_env": "ANTHROPIC_API_KEY"}
        assert config["request"]["model"] == "claude-opus-4-7"
        assert isinstance(config["request"]["max_tokens"], int)
        assert config["request"]["max_tokens"] > 0

    def test_checked_in_anthropic_configs_use_adaptive_thinking(
        self,
    ):
        configs = [
            model_config.get_model_config("anthropic-opus-4-7-medium"),
            model_config.get_model_config("anthropic-opus-4-7-low-thinking"),
        ]

        assert all(
            config["request"]["thinking"] == {"type": "adaptive"}
            for config in configs
        )
        assert configs[0]["request"]["output_config"] == {"effort": "medium"}
        assert configs[1]["request"]["output_config"] == {"effort": "low"}

    def test_checked_in_primary_anthropic_config_enables_streaming(self):
        config = model_config.get_model_config("anthropic-opus-4-7-medium")

        assert config["request"]["stream"] is True

    def test_checked_in_anthropic_configs_do_not_use_openai_compat_fields(self):
        configs = [
            config
            for config in model_config.load_model_configs()
            if config["id"].startswith("anthropic-")
        ]

        assert configs
        assert all(config["runtime"]["sdk"] == "anthropic-python" for config in configs)
        assert all(config["runtime"]["api"] == "messages" for config in configs)
        assert all("base_url" not in config["client"] for config in configs)
        assert all("extra_body" not in config["request"] for config in configs)

    def test_checked_in_analysis_config_enables_reasoning_summary_replay(self):
        config = model_config.get_model_config(
            "openai-gpt-5-4-2026-03-05-responses-analysis"
        )

        assert config["agent"]["analysis_mode"] is True
        assert config["runtime"]["api"] == "responses"
        assert config["request"]["reasoning"] == {
            "effort": "low",
            "summary": "auto",
        }

    def test_list_model_config_ids_supports_mixed_chat_and_responses_configs(
        self,
        tmp_path,
        monkeypatch,
    ):
        _write_model_configs(
            tmp_path,
            monkeypatch,
            [
                _valid_config("chat-config", runtime_api="chat_completions"),
                _valid_config("responses-config", runtime_api="responses"),
            ],
        )

        assert model_config.list_model_config_ids() == [
            "chat-config",
            "responses-config",
        ]
