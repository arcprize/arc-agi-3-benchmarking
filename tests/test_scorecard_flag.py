import json
from argparse import Namespace
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from arcagi3.adcr_agent.definition import definition as adcr_definition
from arcagi3.arc3tester import ARC3Tester
from arcagi3.schemas import Cost, GameResult, ModelConfig, ModelPricing, Usage


class FakeGameClient:
    ROOT_URL: str = "https://test.example.com"

    def __init__(self):
        self.ROOT_URL = "https://test.example.com"
        self.open_calls: List[Dict[str, Any]] = []
        self.close_calls: List[Dict[str, Any]] = []
        self.reset_calls: List[Dict[str, Any]] = []
        self.get_scorecard_calls: List[Dict[str, Any]] = []

    def open_scorecard(self, game_ids, card_id=None, tags=None):
        self.open_calls.append({"game_ids": game_ids, "card_id": card_id, "tags": tags})
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


def _make_tester(
    fake_client: FakeGameClient, submit_scorecard: bool, monkeypatch=None
) -> ARC3Tester:
    # Mock read_models_config to avoid needing a real config
    if monkeypatch:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-for-testing")
        import arcagi3.adapters.provider as provider_module
        import arcagi3.arc3tester as arc3tester_module
        import arcagi3.utils as utils_module
        from arcagi3.utils import task_utils

        dummy_config = ModelConfig(
            name="dummy-config",
            model_name="dummy-model",
            provider="openai",
            is_multimodal=False,
            pricing=ModelPricing(date="2024-01-01", input=0.0, output=0.0),
            kwargs={"memory_word_limit": 100},
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


def test_submit_scorecard_disabled_requires_existing_server_card_id(monkeypatch):
    # Set dummy API key to avoid GameClient initialization error
    monkeypatch.setenv("ARC_API_KEY", "dummy-key-for-testing")
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=False, monkeypatch=monkeypatch)

    with pytest.raises(ValueError, match="Fresh offline runs are not supported"):
        tester.play_game("dummy-game", card_id=None, resume_from_checkpoint=False)

    assert fake_client.open_calls == []
    assert fake_client.close_calls == []
    assert fake_client.reset_calls == []


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
        "current_score": 0,
        "current_state": "IN_PROGRESS",
        "previous_score": 0,
        "use_vision": False,
        "checkpoint_timestamp": "2024-01-01T00:00:00Z",
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    costs = {
        "total_cost": {
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "reasoning_cost": 0.0,
            "total_cost": 0.0,
        },
        "total_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
    }
    with open(checkpoint_dir / "costs.json", "w") as f:
        json.dump(costs, f)

    with open(checkpoint_dir / "action_history.json", "w") as f:
        json.dump([], f)

    # Monkeypatch the checkpoint directory to use our temp directory
    from arcagi3.checkpoint import CheckpointManager

    monkeypatch.setattr(CheckpointManager, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / ".checkpoint"))

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


def test_custom_agent_without_adcr_kwargs_still_instantiates(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "dummy-key-for-testing")
    fake_client = FakeGameClient()
    tester = _make_tester(fake_client, submit_scorecard=True, monkeypatch=monkeypatch)

    received: Dict[str, Any] = {}

    class ConstructorStrictAgent:
        def __init__(
            self,
            config: str,
            game_client: FakeGameClient,
            card_id: str,
            max_actions: int,
            num_plays: int,
            max_episode_actions: int,
            checkpoint_frequency: int,
        ) -> None:
            received.update(
                {
                    "config": config,
                    "game_client": game_client,
                    "card_id": card_id,
                    "max_actions": max_actions,
                    "num_plays": num_plays,
                    "max_episode_actions": max_episode_actions,
                    "checkpoint_frequency": checkpoint_frequency,
                }
            )

        def play_game(
            self,
            game_id: str,
            resume_from_checkpoint: bool = False,
            checkpoint_id: str | None = None,
        ) -> GameResult:
            return GameResult(
                game_id=game_id,
                config="dummy-config",
                final_score=0,
                final_state="GAME_OVER",
                actions_taken=0,
                duration_seconds=0.0,
                total_cost=Cost(
                    prompt_cost=0.0,
                    completion_cost=0.0,
                    reasoning_cost=0.0,
                    total_cost=0.0,
                ),
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                actions=[],
                timestamp=datetime.now(timezone.utc),
                card_id=received["card_id"],
            )

    tester.agent_class = ConstructorStrictAgent

    result = tester.play_game("dummy-game", card_id=None, resume_from_checkpoint=False)

    assert result.game_id == "dummy-game"
    assert received["game_client"] is fake_client
    assert received["checkpoint_frequency"] == 0
    assert received["card_id"] == "server-card-id"
    assert len(fake_client.open_calls) == 1
    assert len(fake_client.close_calls) == 1


def test_adcr_definition_maps_runtime_kwargs_from_cli_args():
    kwargs = adcr_definition["get_kwargs"](
        Namespace(use_vision=False, show_images=True, memory_limit=321)
    )

    assert kwargs == {
        "use_vision": False,
        "show_images": True,
        "memory_word_limit": 321,
    }
