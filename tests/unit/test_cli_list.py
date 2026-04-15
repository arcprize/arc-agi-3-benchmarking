from argparse import Namespace
from unittest.mock import patch

import pytest

from benchmarking.cli_list import print_requested_resource_lists


def _fetch_games(_root_url: str) -> list[str]:
    return ["ls20-game-1", "ls21-game-2"]


@pytest.mark.unit
class TestCliList:
    def test_print_requested_resource_lists_prints_requested_lists(self, capsys):
        args = Namespace(list_configs=True, list_games=False)

        with patch(
            "benchmarking.cli_list.list_model_config_ids",
            return_value=["openai-gpt-5.4-openrouter"],
        ):
            handled = print_requested_resource_lists(
                args,
                root_url="https://example.com",
                fetch_available_games=_fetch_games,
            )

        captured = capsys.readouterr()

        assert handled is True
        assert captured.out == (
            "Configs:\n"
            "- openai-gpt-5.4-openrouter\n"
        )

    def test_print_requested_resource_lists_prints_games_and_configs_together(
        self,
        capsys,
    ):
        args = Namespace(list_configs=True, list_games=True)

        with patch(
            "benchmarking.cli_list.list_model_config_ids",
            return_value=["openai-gpt-5.4-openrouter"],
        ):
            handled = print_requested_resource_lists(
                args,
                root_url="https://example.com",
                fetch_available_games=_fetch_games,
            )

        captured = capsys.readouterr()

        assert handled is True
        assert captured.out == (
            "Configs:\n"
            "- openai-gpt-5.4-openrouter\n"
            "\n"
            "Games:\n"
            "- ls20\n"
            "- ls21\n"
        )

    def test_print_requested_resource_lists_returns_false_when_no_list_flags_are_set(
        self,
    ):
        args = Namespace(list_configs=False, list_games=False)

        assert (
            print_requested_resource_lists(
                args,
                root_url="https://example.com",
                fetch_available_games=_fetch_games,
            )
            is False
        )
