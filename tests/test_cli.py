from contextlib import redirect_stdout
from io import StringIO

from arcagi3.utils.cli import handle_list_games, normalize_game_selector, resolve_game_selector


class DummyGameClient:
    def __init__(self, games):
        self._games = games

    def list_games(self):
        return list(self._games)


def test_normalize_game_selector_handles_simple_uppercase_title():
    assert normalize_game_selector(" LS20 ") == "ls20"


def test_normalize_game_selector_strips_hash_when_supported():
    assert normalize_game_selector("LS20-016295F7601E") == "ls20"


def test_resolve_game_selector_matches_title_alias_case_insensitively():
    client = DummyGameClient(
        [
            {"game_id": "ls20-016295f7601e", "title": "LS20"},
            {"game_id": "g50t-5849a774", "title": "G50T"},
        ]
    )

    assert resolve_game_selector(client, "LS20") == "ls20-016295f7601e"
    assert resolve_game_selector(client, "G50T") == "g50t-5849a774"


def test_resolve_game_selector_preserves_unknown_selector_with_local_normalization():
    client = DummyGameClient([])

    assert resolve_game_selector(client, " Unknown ") == "Unknown"


def test_handle_list_games_shows_api_id_and_easy_selector():
    client = DummyGameClient(
        [
            {"game_id": "ls20-016295f7601e", "title": "LS20"},
            {"game_id": "ar25", "title": "AR25"},
        ]
    )
    buffer = StringIO()

    with redirect_stdout(buffer):
        handle_list_games(client)

    output = buffer.getvalue()
    assert "Game ID" in output
    assert "Selector" in output
    assert "ls20-016295f7601e" in output
    assert "ls20" in output
