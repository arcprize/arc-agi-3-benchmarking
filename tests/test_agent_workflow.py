from typing import Any, Dict, List

import arcagi3.agent as agent_module
from arcagi3.agent import MultimodalAgent
from arcagi3.schemas import GameResult, GameStep
from arcagi3.utils.context import SessionContext


class DummyProvider:
    """Minimal provider stub used for testing workflow hooks."""

    class ModelConfig:
        class Pricing:
            input = 0
            output = 0

        pricing = Pricing()

        class Kwargs:
            memory_word_limit = 100

        kwargs = {"memory_word_limit": 100}

        is_multimodal = False

    model_config = ModelConfig()

    def call_provider(self, messages):
        # Return a dummy response that the agent can parse.
        return {"choices": [{"message": {"content": '{"human_action":"ACTION"}'}}]}

    def extract_usage(self, response):
        return 0, 0, 0

    def extract_content(self, response):
        # Used only in places where JSON is expected.
        return '{"human_action":"ACTION","reasoning":"","expected_result":""}'


class DummyGameClient:
    ROOT_URL: str = "https://test.example.com"

    def __init__(self):
        self.ROOT_URL = "https://test.example.com"
        self.reset_calls = 0
        self.execute_calls: List[Dict[str, Any]] = []

    def _make_64x64_grid(self) -> List[List[int]]:
        """Create a 64x64 grid filled with zeros."""
        return [[0] * 64 for _ in range(64)]

    def reset_game(self, card_id: str, game_id: str, guid=None):
        self.reset_calls += 1
        # Single frame grid with arbitrary score/state.
        return {
            "guid": "dummy-guid",
            "score": 0,
            "state": "IN_PROGRESS",
            "frame": [self._make_64x64_grid()],
            "available_actions": ["1"],
        }

    def execute_action(self, action_name: str, data: Dict[str, Any]):
        self.execute_calls.append({"action": action_name, "data": data})
        # End the game immediately.
        return {
            "guid": data.get("guid", "dummy-guid"),
            "score": 1,
            "state": "GAME_OVER",
            "frame": [self._make_64x64_grid()],
        }


class HookedAgent(MultimodalAgent):
    """Subclass that overrides a specific hook to test customization."""

    def step(self, context: SessionContext) -> GameStep:
        """Return a fixed ACTION1 without consulting the model."""
        return GameStep(
            action={"action": "ACTION1", "x": 0, "y": 0},
            reasoning={"test": "hooked-agent"},
        )


def test_hooked_agent_uses_overridden_convert_to_game_action(monkeypatch):
    # Patch provider factory to return our dummy provider.
    monkeypatch.setattr(agent_module, "create_provider", lambda config: DummyProvider())

    game_client = DummyGameClient()
    agent = HookedAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        checkpoint_frequency=0,
    )

    result: GameResult = agent.play_game("dummy-game", resume_from_checkpoint=False)

    # Ensure the overridden hook was effectively used: the game client should have
    # seen our fixed ACTION1.
    assert game_client.reset_calls == 1
    assert len(game_client.execute_calls) == 1
    assert game_client.execute_calls[0]["action"] == "ACTION1"
    assert result.final_state in ("GAME_OVER", "WIN")
