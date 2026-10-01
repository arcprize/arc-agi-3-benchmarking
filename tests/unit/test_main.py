from unittest.mock import MagicMock, patch

import pytest

import main as cli_main


@pytest.mark.unit
class TestMainCliHelpers:
    def test_build_parser_supports_runtime_flags(self):
        args = cli_main.build_parser().parse_args(["--config", "openai-gpt-5.4-openrouter"])

        assert args.list_games is False
        assert args.list_configs is False
        assert args.config == "openai-gpt-5.4-openrouter"

    def test_list_model_config_ids_reads_checked_in_configs(self):
        configs = cli_main.list_model_config_ids()

        assert "openai-gpt-5.4-openrouter" in configs
        assert "anthropic-opus-4-7-medium" in configs
        assert "anthropic-opus-4-7-low-thinking" in configs

    def test_validate_required_model_api_key_uses_selected_config_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        cli_main.validate_required_model_api_key("openai-gpt-5.4-openrouter")

    def test_validate_required_model_api_key_uses_agent_model_config_id_when_config_omitted(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(
            cli_main.BenchmarkingAgent,
            "MODEL_CONFIG_ID",
            "openai-gpt-5.4-openrouter",
        )

        cli_main.validate_required_model_api_key(None)

    def test_validate_required_model_api_key_rejects_blank_values(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")

        with pytest.raises(ValueError, match="No OPENROUTER_API_KEY set"):
            cli_main.validate_required_model_api_key("openai-gpt-5.4-openrouter")

    def test_validate_required_model_api_key_lists_available_configs_for_unknown_id(
        self,
    ):
        with pytest.raises(ValueError) as exc_info:
            cli_main.validate_required_model_api_key("does-not-exist")

        message = str(exc_info.value)
        assert "Model config 'does-not-exist' not found" in message
        assert "Available configs:" in message
        assert "openai-gpt-5.4-openrouter" in message

    def test_validate_required_model_api_key_rejects_missing_client_api_key_env(self):
        with patch(
            "main.get_model_config",
            return_value={"client": {}},
        ):
            with pytest.raises(ValueError) as exc_info:
                cli_main.validate_required_model_api_key("missing-api-key-env")

        assert (
            str(exc_info.value)
            == "Model config 'missing-api-key-env' is missing client.api_key_env."
        )

    @pytest.mark.parametrize(
        "config_id",
        [
            "openai-gpt-5-4-2026-03-05",
            "openai-gpt-5-4-2026-03-05-responses",
        ],
    )
    def test_validate_required_model_api_key_rejects_missing_openai_key_for_chat_and_responses_configs(
        self,
        monkeypatch,
        config_id,
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            cli_main.validate_required_model_api_key(config_id)

        assert str(exc_info.value) == (
            "No OPENAI_API_KEY set. "
            f"The selected model config '{config_id}' requires "
            "the OPENAI_API_KEY environment variable to be set in your .env file."
        )

    def test_fetch_available_games_parses_game_ids(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"game_id": "ls20"},
            {"game_id": "ls21"},
        ]

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None

        with patch("main.requests.Session", return_value=mock_session):
            games = cli_main.fetch_available_games("https://example.com")

        assert games == ["ls20", "ls21"]
        mock_session.get.assert_called_once_with(
            "https://example.com/api/games",
            timeout=10,
        )


def _args(*argv: str):
    return cli_main.build_parser().parse_args(list(argv))


@pytest.mark.unit
class TestRehydrationCli:
    def test_parser_collects_repeated_rehydrate_pairs(self):
        args = _args("--rehydrate", "recording=r.jsonl", "--rehydrate", "state=s.json")
        assert args.rehydrate == ["recording=r.jsonl", "state=s.json"]
        assert _args().rehydrate is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, False), ("", False), ("false", False), ("0", False),
         ("FALSE", False), ("true", True), ("TRUE", True), ("1", True), (" true ", True)],
    )
    def test_tag_env_var_parsing(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv(cli_main.REHYDRATION_TAG_ENV, raising=False)
        else:
            monkeypatch.setenv(cli_main.REHYDRATION_TAG_ENV, value)
        assert cli_main.rehydration_tag_enabled() is expected

    @pytest.mark.parametrize("value", ["yes", "ture", "2"])
    def test_tag_env_var_rejects_unexpected_values(self, monkeypatch, value):
        monkeypatch.setenv(cli_main.REHYDRATION_TAG_ENV, value)
        with pytest.raises(ValueError, match=cli_main.REHYDRATION_TAG_ENV):
            cli_main.rehydration_tag_enabled()

    def test_no_rehydrate_returns_nothing(self, monkeypatch):
        monkeypatch.delenv(cli_main.REHYDRATION_TAG_ENV, raising=False)
        assert cli_main.resolve_rehydration(_args(), ["g"]) == (None, [])

    def test_tag_without_rehydrate_warns_and_adds_no_tag(self, monkeypatch, caplog):
        monkeypatch.setenv(cli_main.REHYDRATION_TAG_ENV, "true")
        assert cli_main.resolve_rehydration(_args(), ["g"]) == (None, [])
        assert "--rehydrate is not" in caplog.text

    def test_invalid_tag_value_fails_even_without_rehydrate(self, monkeypatch):
        monkeypatch.setenv(cli_main.REHYDRATION_TAG_ENV, "yes")
        with pytest.raises(ValueError):
            cli_main.resolve_rehydration(_args(), ["g"])

    def test_rehydrate_requires_config(self, monkeypatch):
        monkeypatch.delenv(cli_main.REHYDRATION_TAG_ENV, raising=False)
        with pytest.raises(ValueError, match="requires --config"):
            cli_main.resolve_rehydration(_args("--rehydrate", "state=s"), ["g"])

    @pytest.mark.parametrize(("tag_value", "tags"), [("false", []), ("true", ["rehydrated"])])
    def test_rehydrate_prepares_inputs_and_tags(self, monkeypatch, tag_value, tags):
        monkeypatch.setenv(cli_main.REHYDRATION_TAG_ENV, tag_value)
        prepared = MagicMock()
        calls = {}
        monkeypatch.setattr(cli_main, "parse_rehydrate_args", lambda pairs: ("args", pairs))

        def fake_prepare(args, *, config_id, game_ids):
            calls.update(args=args, config_id=config_id, game_ids=game_ids)
            return prepared

        monkeypatch.setattr(cli_main, "prepare_rehydration", fake_prepare)
        args = _args("-c", "cfg", "--rehydrate", "recording=r", "--rehydrate", "state=s")

        assert cli_main.resolve_rehydration(args, ["ls20-abc"]) == (prepared, tags)
        assert calls == {
            "args": ("args", ["recording=r", "state=s"]),
            "config_id": "cfg",
            "game_ids": ["ls20-abc"],
        }
