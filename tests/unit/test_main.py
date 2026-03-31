from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

import main as cli_main


@pytest.mark.unit
class TestMainCliHelpers:
    def test_build_parser_supports_list_flags(self):
        args = cli_main.build_parser().parse_args(["--list-agents"])

        assert args.list_agents is True
        assert args.list_games is False
        assert args.list_models is False
        assert args.agent is None

    def test_list_model_config_ids_reads_checked_in_configs(self):
        configs = cli_main.list_model_config_ids()

        assert "openai-gpt-5.4-openrouter" in configs
        assert "anthropic-opus-4-6" in configs

    def test_list_agent_names_filters_out_recordings(self):
        with patch.dict(
            "main.AVAILABLE_AGENTS",
            {
                "random": object(),
                "llm": object(),
                "ls20.random.80.guid.recording.jsonl": object(),
            },
            clear=True,
        ):
            agents = cli_main.list_agent_names()

        assert agents == ["llm", "random"]

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

    def test_maybe_handle_list_requests_prints_requested_lists(self, capsys):
        args = Namespace(list_agents=True, list_models=True, list_games=False)

        with (
            patch("main.list_agent_names", return_value=["random", "llm"]),
            patch("main.list_model_config_ids", return_value=["openai-gpt-5.4-openrouter"]),
        ):
            handled = cli_main.maybe_handle_list_requests(args)

        captured = capsys.readouterr()

        assert handled is True
        assert captured.out == (
            "Agents:\n"
            "random\n"
            "llm\n"
            "\n"
            "Models:\n"
            "openai-gpt-5.4-openrouter\n"
        )

    def test_maybe_handle_list_requests_returns_false_when_unused(self):
        args = Namespace(list_agents=False, list_models=False, list_games=False)

        assert cli_main.maybe_handle_list_requests(args) is False
