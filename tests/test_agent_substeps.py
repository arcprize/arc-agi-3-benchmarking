from typing import Any, Dict, List

from PIL import Image

from arcagi3.agent import MultimodalAgent, HUMAN_ACTIONS


class DummyProvider:
    """Stub provider that records messages and returns fixed JSON."""

    class ModelConfig:
        class Pricing:
            input = 0
            output = 0

        pricing = Pricing()

        kwargs: Dict[str, Any] = {"memory_word_limit": 100}
        is_multimodal = False

    model_config = ModelConfig()

    def __init__(self):
        self.last_messages: List[Dict[str, Any]] = []

    def call_provider(self, messages):
        self.last_messages = messages
        # Minimal response body; content will be parsed as JSON by the agent.
        return {"choices": [{"message": {"content": '{"human_action":"ACTION1","action":"ACTION1"}'}}]}

    def extract_usage(self, response):
        # No cost accounting in tests.
        return 0, 0, 0

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


class DummyGameClient:
    """Game client stub that never hits the network."""

    def reset_game(self, card_id: str, game_id: str, guid=None):
        return {
            "guid": "dummy-guid",
            "score": 0,
            "state": "IN_PROGRESS",
            # Single 1x1 grid frame.
            "frame": [[[0]]],
            "available_actions": ["1", "2", "6"],
        }

    def execute_action(self, action_name: str, data: Dict[str, Any]):
        # Immediately end the game.
        return {
            "guid": data.get("guid", "dummy-guid"),
            "score": 1,
            "state": "GAME_OVER",
            "frame": [[[0]]],
        }


def _make_agent(monkeypatch) -> MultimodalAgent:
    import arcagi3.agent as agent_module

    dummy_provider = DummyProvider()
    monkeypatch.setattr(agent_module, "create_provider", lambda config: dummy_provider)

    game_client = DummyGameClient()
    agent = MultimodalAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        retry_attempts=1,
        num_plays=1,
        show_images=False,
        use_vision=False,
        checkpoint_frequency=0,
    )
    # Expose provider for inspection in tests.
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_decide_human_action_step_includes_available_actions_and_memory(monkeypatch):
    agent = _make_agent(monkeypatch)

    # Simulate available action codes and memory text.
    agent._available_actions = ["1", "2", "6"]
    agent._memory_prompt = "Previous memory scratchpad"

    # Simple 1x1 grid frame for text-only path.
    frame_grids = [[[0]]]
    frame_images: List[Image.Image] = []

    analysis = "Some prior analysis"

    result = agent.decide_human_action_step(frame_images, frame_grids, analysis)
    assert result["human_action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    # The last user message should contain our instruction text.
    user_msg = messages[-1]["content"]

    # Ensure bullet list has at least one known action description.
    any_desc = any(desc in str(user_msg) for desc in HUMAN_ACTIONS.values())
    assert any_desc

    # Ensure memory text is present in the prompt.
    assert "Previous memory scratchpad" in str(user_msg)


def test_convert_human_to_game_action_step_includes_valid_actions(monkeypatch):
    agent = _make_agent(monkeypatch)

    agent._available_actions = ["1", "6"]

    human_action = "Click the red square"
    last_frame_grid = [[0]]
    # Dummy 1x1 image for completeness, although we use text-only path.
    last_frame_image = Image.new("RGB", (1, 1))

    result = agent.convert_human_to_game_action_step(
        human_action, last_frame_image, last_frame_grid
    )
    assert result["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]

    # Ensure action list and valid actions hints are present.
    text = str(user_msg)
    assert "ACTION1" in text
    assert "ACTION6" in text


def test_validate_action_matches_available_actions(monkeypatch):
    agent = _make_agent(monkeypatch)
    agent._available_actions = ["1", "6"]

    assert agent._validate_action("ACTION1") is True
    assert agent._validate_action("ACTION6") is True
    assert agent._validate_action("ACTION3") is False


