"""Tests for KnowItAllAgent."""

from typing import Any, Dict, List

from PIL import Image

from arcagi3.examples.knowitall import KnowItAllAgent
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
        return {"choices": [{"message": {"content": '{"action":"ACTION1"}'}}]}

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


def _make_agent(monkeypatch, game_rules: str = "Test game rules", use_vision: bool = False) -> KnowItAllAgent:
    import arcagi3.agent as agent_module
    import arcagi3.adapters as adapters_module
    from arcagi3.utils import task_utils

    dummy_provider = DummyProvider()
    dummy_provider.model_config.is_multimodal = use_vision
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
            'is_multimodal': use_vision,
        })()
    )

    game_client = DummyGameClient()
    agent = KnowItAllAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        game_rules=game_rules,
        use_vision=use_vision,
        checkpoint_frequency=0,
    )
    # Expose provider for inspection in tests.
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_game_rules_required_argument(monkeypatch):
    """Test that game_rules is a required parameter."""
    import arcagi3.agent as agent_module
    import arcagi3.adapters as adapters_module
    from arcagi3.utils import task_utils

    dummy_provider = DummyProvider()
    monkeypatch.setattr(agent_module, "create_provider", lambda config: dummy_provider)
    monkeypatch.setattr(adapters_module, "create_provider", lambda config: dummy_provider)
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
    
    # Should raise TypeError if game_rules is missing
    import pytest
    with pytest.raises(TypeError):
        KnowItAllAgent(
            config="dummy-config",
            game_client=game_client,
            card_id="local-test",
            max_actions=5,
            num_plays=1,
            max_episode_actions=0,
            checkpoint_frequency=0,
        )


def test_game_rules_included_in_prompt(monkeypatch):
    """Test that game_rules are included in the step prompt."""
    agent = _make_agent(monkeypatch, game_rules="Move to the goal. Avoid walls.")

    context = SessionContext()
    context.set_available_actions(["1", "2"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    assert step.action["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Ensure game rules are present in the prompt
    assert "Move to the goal. Avoid walls." in str(user_msg)


def test_memory_included_when_present(monkeypatch):
    """Test that memory is included in prompt when present."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["memory_prompt"] = "Previous memory: walls are at (5,5)"

    step = agent.step(context)
    assert step.action["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Ensure memory is present
    assert "Previous memory: walls are at (5,5)" in str(user_msg)
    assert "Memory:" in str(user_msg)


def test_memory_truncated_when_too_long(monkeypatch):
    """Test that memory is truncated to memory_word_limit."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")
    agent.memory_word_limit = 5  # Very small limit for testing

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    # Memory with more than 5 words
    context.datastore["memory_prompt"] = "This is a very long memory that should be truncated to five words"

    step = agent.step(context)
    assert step.action["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Memory should be truncated (only first 5 words)
    memory_section = str(user_msg)
    # Should contain truncated version, not full version
    assert "This is a very long" in memory_section
    # Should not contain the full untruncated text
    assert "truncated to five words" not in memory_section


def test_no_reasoning_in_step_output(monkeypatch):
    """Test that step returns empty reasoning dict."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    
    # KnowItAllAgent should return empty reasoning
    assert step.reasoning == {}


def test_action_validation_against_available_actions(monkeypatch):
    """Test that invalid actions are rejected."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")

    context = SessionContext()
    context.set_available_actions(["1", "6"])  # Only ACTION1 and ACTION6 available

    # Valid actions
    assert agent._validate_action(context, "ACTION1") is True
    assert agent._validate_action(context, "ACTION6") is True
    
    # Invalid actions
    assert agent._validate_action(context, "ACTION3") is False
    assert agent._validate_action(context, "ACTION7") is False


def test_vision_mode_includes_images(monkeypatch):
    """Test that vision mode includes image blocks instead of text grids."""
    agent = _make_agent(monkeypatch, game_rules="Test rules", use_vision=True)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    assert step.action["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should contain image blocks, not text grid representations
    assert any("image_url" in str(block) for block in user_content)
    # Should not contain "Frame 0:" text grid format
    assert "Frame 0:" not in str(user_content)


def test_text_mode_includes_grid_text(monkeypatch):
    """Test that text mode includes grid text representations."""
    agent = _make_agent(monkeypatch, game_rules="Test rules", use_vision=False)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    assert step.action["action"] == "ACTION1"

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should contain text grid format
    assert "Frame 0:" in str(user_content)
    # Should not contain image_url blocks
    assert "image_url" not in str(user_content)


def test_action_with_coordinates(monkeypatch):
    """Test that ACTION6 with x/y coordinates is handled correctly."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")
    
    # Override provider to return ACTION6 with coordinates
    provider = agent._test_provider  # type: ignore[attr-defined]
    original_call = provider.call_provider
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {"choices": [{"message": {"content": '{"action":"ACTION6","x":10,"y":20}'}}]}
    
    provider.call_provider = mock_call_provider

    context = SessionContext()
    context.set_available_actions(["6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    
    assert step.action["action"] == "ACTION6"
    assert step.action.get("x") == 10
    assert step.action.get("y") == 20


def test_action_data_dict_extracted(monkeypatch):
    """Test that action data dict is extracted when present."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")
    
    # Override provider to return action with data dict
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {"choices": [{"message": {"content": '{"action":"ACTION6","data":{"x":15,"y":25}}'}}]}
    
    provider.call_provider = mock_call_provider

    context = SessionContext()
    context.set_available_actions(["6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    
    assert step.action["action"] == "ACTION6"
    assert step.action.get("data") == {"x": 15, "y": 25}


def test_step_raises_error_when_no_action_in_response(monkeypatch):
    """Test that step raises error when no action is returned."""
    agent = _make_agent(monkeypatch, game_rules="Test rules")
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {"choices": [{"message": {"content": '{"reasoning":"test"}'}}]}
    
    provider.call_provider = mock_call_provider

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    import pytest
    with pytest.raises(ValueError, match="No action in response"):
        agent.step(context)

