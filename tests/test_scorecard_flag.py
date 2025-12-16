import json
import os
from pathlib import Path
from typing import Any, Dict, List

from arcagi3.arc3tester import ARC3Tester
from arcagi3.schemas import GameResult, ModelConfig, ModelPricing


class FakeGameClient:
    ROOT_URL: str = "https://test.example.com"
    
    def __init__(self):
        self.ROOT_URL = "https://test.example.com"
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


def _make_tester(fake_client: FakeGameClient, submit_scorecard: bool, monkeypatch=None) -> ARC3Tester:
    # Mock read_models_config to avoid needing a real config
    if monkeypatch:
        from arcagi3.utils import task_utils
        import arcagi3.arc3tester as arc3tester_module
        import arcagi3.utils as utils_module
        import arcagi3.adapters.provider as provider_module
        dummy_config = ModelConfig(
            name="dummy-config",
            model_name="dummy-model",
            provider="openai",
            is_multimodal=False,
            pricing=ModelPricing(date="2024-01-01", input=0.0, output=0.0),
            kwargs={"memory_word_limit": 100}
        )
        # Patch where it's defined and all places it might be imported
        monkeypatch.setattr(task_utils, "read_models_config", lambda config: dummy_config)
        monkeypatch.setattr(arc3tester_module, "read_models_config", lambda config: dummy_config)
        monkeypatch.setattr(utils_module, "read_models_config", lambda config: dummy_config)
        monkeypatch.setattr(provider_module, "read_models_config", lambda config: dummy_config)
    
    tester = ARC3Tester(
        config="dummy-config",
        save_results_dir=None,
        overwrite_results=False,
        max_actions=1,
        retry_attempts=1,
        api_retries=1,
        num_plays=1,
        max_episode_actions=0,
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


def test_submit_scorecard_disabled_skips_open_and_close_when_no_card_id(monkeypatch):
    # Set dummy API key to avoid GameClient initialization error
    monkeypatch.setenv("ARC_API_KEY", "dummy-key-for-testing")
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=False, monkeypatch=monkeypatch)

    result: GameResult = tester.play_game("dummy-game", card_id=None, resume_from_checkpoint=False)
    assert result.game_id == "dummy-game"

    # No explicit scorecard open/close calls should have been made.
    assert fake_client.open_calls == []
    assert fake_client.close_calls == []
    # But reset_game should still have been called with some local card_id.
    assert len(fake_client.reset_calls) == 1
    assert fake_client.reset_calls[0]["card_id"].startswith("local-")


def test_resume_from_existing_checkpoint_still_uses_scorecard_apis(monkeypatch, tmp_path):
    # Set dummy API key to avoid GameClient initialization error
    monkeypatch.setenv("ARC_API_KEY", "dummy-key-for-testing")
    
    # Create a fake checkpoint directory and metadata file
    checkpoint_dir = tmp_path / ".checkpoint" / "existing-card"
    checkpoint_dir.mkdir(parents=True)
    metadata = {
        "card_id": "existing-card",
        "config": "dummy-config",
        "game_id": "dummy-game",
        "guid": "fake-guid",
        "max_actions": 1,
        "retry_attempts": 1,
        "num_plays": 1,
        "max_episode_actions": 0,
        "action_counter": 0,
        "current_play": 1,
        "play_action_counter": 0,
        "previous_score": 0,
        "use_vision": False,
        "checkpoint_timestamp": "2024-01-01T00:00:00Z",
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Monkeypatch the checkpoint directory to use our temp directory
    from arcagi3.checkpoint import CheckpointManager
    original_checkpoint_dir = CheckpointManager.CHECKPOINT_DIR
    monkeypatch.setattr(CheckpointManager, "CHECKPOINT_DIR", str(tmp_path / ".checkpoint"))
    
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=False, monkeypatch=monkeypatch)

    # Even with submit_scorecard=False, when resuming from an existing card_id
    # we should still call get_scorecard but not open a new one.
    result: GameResult = tester.play_game(
        "dummy-game", card_id="existing-card", resume_from_checkpoint=True
    )
    assert result.game_id == "dummy-game"

    assert fake_client.get_scorecard_calls[0]["card_id"] == "existing-card"
    # No new scorecard should be opened in this path.
    assert fake_client.open_calls == []


