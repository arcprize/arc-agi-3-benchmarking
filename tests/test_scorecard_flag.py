from typing import Any, Dict, List

from arcagi3.arc3tester import ARC3Tester
from arcagi3.schemas import GameResult


class FakeGameClient:
    def __init__(self):
        self.open_calls: List[Dict[str, Any]] = []
        self.close_calls: List[Dict[str, Any]] = []
        self.reset_calls: List[Dict[str, Any]] = []
        self.get_scorecard_calls: List[Dict[str, Any]] = []

    def open_scorecard(self, game_ids, card_id=None, tags=None):
        self.open_calls.append(
            {"game_ids": game_ids, "card_id": card_id, "tags": tags}
        )
        return {"card_id": card_id or "server-card-id"}

    def close_scorecard(self, card_id: str):
        self.close_calls.append({"card_id": card_id})
        return {}

    def get_scorecard(self, card_id: str, game_id=None):
        self.get_scorecard_calls.append({"card_id": card_id, "game_id": game_id})
        return {}

    def reset_game(self, card_id: str, game_id: str, guid=None):
        self.reset_calls.append({"card_id": card_id, "game_id": game_id, "guid": guid})
        return {
            "guid": "fake-guid",
            "score": 0,
            "state": "GAME_OVER",
            "frame": [[[0]]],
        }


def _make_tester(fake_client: FakeGameClient, submit_scorecard: bool) -> ARC3Tester:
    tester = ARC3Tester(
        config="dummy-config",
        save_results_dir=None,
        overwrite_results=False,
        max_actions=1,
        retry_attempts=1,
        api_retries=1,
        num_plays=1,
        show_images=False,
        use_vision=False,
        checkpoint_frequency=0,
        close_on_exit=False,
        memory_word_limit=10,
        submit_scorecard=submit_scorecard,
    )
    # Inject fake client
    tester.game_client = fake_client
    return tester


def test_submit_scorecard_disabled_skips_open_and_close_when_no_card_id():
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=False)

    result: GameResult = tester.play_game("dummy-game", card_id=None, resume_from_checkpoint=False)
    assert result.game_id == "dummy-game"

    # No explicit scorecard open/close calls should have been made.
    assert fake_client.open_calls == []
    assert fake_client.close_calls == []
    # But reset_game should still have been called with some local card_id.
    assert len(fake_client.reset_calls) == 1
    assert fake_client.reset_calls[0]["card_id"].startswith("local-")


def test_resume_from_existing_checkpoint_still_uses_scorecard_apis():
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=False)

    # Even with submit_scorecard=False, when resuming from an existing card_id
    # we should still call get_scorecard but not open a new one.
    result: GameResult = tester.play_game(
        "dummy-game", card_id="existing-card", resume_from_checkpoint=True
    )
    assert result.game_id == "dummy-game"

    assert fake_client.get_scorecard_calls[0]["card_id"] == "existing-card"
    # No new scorecard should be opened in this path.
    assert fake_client.open_calls == []


