import logging
import signal
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


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Run main() offline; returns the patched Swarm class."""
    # main() writes logs.log to the cwd and attaches root-logger handlers.
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(cli_main, "print_requested_resource_lists", lambda *a, **k: False)
    monkeypatch.setattr(cli_main, "validate_required_model_api_key", lambda _id: None)
    monkeypatch.setattr(cli_main, "fetch_available_games", lambda _url: ["ls20-abc"])
    swarm = MagicMock()
    monkeypatch.setattr(cli_main, "Swarm", swarm)
    return swarm


@pytest.mark.unit
class TestRehydrationCli:
    def test_no_rehydrate_returns_nothing(self):
        assert cli_main.resolve_rehydration(_args(), ["g"]) is None

    def test_rehydrate_requires_config(self):
        with pytest.raises(ValueError, match="requires --config"):
            cli_main.resolve_rehydration(_args("--rehydrate", "state=s"), ["g"])

    def test_rehydrate_prepares_inputs(self, monkeypatch):
        prepared = MagicMock()
        calls = {}
        monkeypatch.setattr(cli_main, "parse_rehydrate_args", lambda pairs: ("args", pairs))

        def fake_prepare(args, *, config_id, game_ids):
            calls.update(args=args, config_id=config_id, game_ids=game_ids)
            return prepared

        monkeypatch.setattr(cli_main, "prepare_rehydration", fake_prepare)
        args = _args("-c", "cfg", "--rehydrate", "recording=r", "--rehydrate", "state=s")

        assert cli_main.resolve_rehydration(args, ["ls20-abc"]) is prepared
        assert calls == {
            "args": ("args", ["recording=r", "state=s"]),
            "config_id": "cfg",
            "game_ids": ["ls20-abc"],
        }

    def test_main_exits_with_rehydration_code_on_invalid_inputs(
        self, monkeypatch, cli_env
    ):
        monkeypatch.setattr(
            "sys.argv",
            ["main.py", "-g", "ls20", "-c", "cfg", "--rehydrate", "state=missing.json"],
        )

        with pytest.raises(SystemExit) as excinfo:
            cli_main.main()

        assert excinfo.value.code == cli_main.REHYDRATION_EXIT_CODE == 3
        cli_env.assert_not_called()


@pytest.fixture
def shutdown_state(monkeypatch, caplog):
    """Reset the shutdown globals and restore signal handlers after the test."""
    monkeypatch.setattr(cli_main, "_shutting_down", False)
    monkeypatch.setattr(cli_main, "_swarm", None)
    caplog.set_level(logging.INFO)
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


@pytest.mark.unit
@pytest.mark.usefixtures("shutdown_state")
class TestShutdownSignals:
    @pytest.mark.parametrize(
        ("signum", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)]
    )
    def test_exit_code_without_swarm(self, caplog, signum, code):
        with pytest.raises(SystemExit) as excinfo:
            cli_main.handle_shutdown_signal(signum, None)

        assert excinfo.value.code == code
        assert "SHUTDOWN: no scorecard opened" in _messages(caplog)

    def test_nested_signal_during_close_is_ignored(self, caplog, monkeypatch):
        swarm = MagicMock()
        swarm.card_id = "card-123"
        swarm.close_scorecard.side_effect = (
            lambda: cli_main.handle_shutdown_signal(signal.SIGINT, None)
        )
        monkeypatch.setattr(cli_main, "_swarm", swarm)

        with pytest.raises(SystemExit) as excinfo:
            cli_main.handle_shutdown_signal(signal.SIGTERM, None)

        assert excinfo.value.code == 143
        swarm.request_shutdown.assert_called_once()
        swarm.close_scorecard.assert_called_once()
        assert "Ignoring SIGINT: shutdown already in progress." in _messages(caplog)

    def test_signal_during_startup_exits_before_swarm(self, caplog, monkeypatch, cli_env):
        monkeypatch.setattr("sys.argv", ["main.py", "-g", "ls20", "-c", "cfg"])

        def fetch_games(_url):
            signal.raise_signal(signal.SIGTERM)
            return ["ls20-abc"]

        monkeypatch.setattr(cli_main, "fetch_available_games", fetch_games)

        with pytest.raises(SystemExit) as excinfo:
            cli_main.main()

        assert excinfo.value.code == 143
        cli_env.assert_not_called()
        assert "SHUTDOWN: no scorecard opened" in _messages(caplog)
