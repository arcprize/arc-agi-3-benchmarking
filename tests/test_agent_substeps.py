from typing import Any, Dict, List

from PIL import Image

from arcagi3.examples.adcr import ADCRAgent
from arcagi3.agent import HUMAN_ACTIONS
from arcagi3.utils.context import SessionContext


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

    def call_with_tracking(self, context: SessionContext, messages):
        # Tests don't charge cost; just record messages.
        return self.call_provider(messages)

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


def _make_agent(monkeypatch) -> ADCRAgent:
    import arcagi3.agent as agent_module
    import arcagi3.adapters as adapters_module
    from arcagi3.utils import task_utils

    dummy_provider = DummyProvider()
    # Patch create_provider where it's actually referenced by agent.py
    monkeypatch.setattr(agent_module, "create_provider", lambda config: dummy_provider)
    # Also patch adapters module for completeness
    monkeypatch.setattr(adapters_module, "create_provider", lambda config: dummy_provider)
    # Also patch read_models_config to avoid config lookup
    monkeypatch.setattr(
        task_utils,
        "read_models_config",
        lambda config: type('ModelConfig', (), {
            'provider': 'dummy',
            'pricing': type('Pricing', (), {'input': 0, 'output': 0})(),
            'kwargs': {'memory_word_limit': 100},
            'is_multimodal': False,
        })()
    )

    game_client = DummyGameClient()
    agent = ADCRAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        use_vision=False,
        checkpoint_frequency=0,
    )
    # Expose provider for inspection in tests.
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_decide_human_action_step_includes_available_actions_and_memory(monkeypatch):
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1", "2", "6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["memory_prompt"] = "Previous memory scratchpad"

    # Simple 1x1 grid frame for text-only path.
    analysis = "Some prior analysis"

    result = agent.decide_human_action_step(context, analysis)
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

    context = SessionContext()
    context.set_available_actions(["1", "6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    human_action = "Click the red square"
    result = agent.convert_human_to_game_action_step(context, human_action)
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
    context = SessionContext()
    context.set_available_actions(["1", "6"])

    assert agent.validate_action(context, "ACTION1") is True
    assert agent.validate_action(context, "ACTION6") is True
    assert agent.validate_action(context, "ACTION3") is False


