"""Tests for SwarmAgent."""

from typing import Any, Dict, List

from arcagi3.examples.swarm_agent import SwarmAgent
from arcagi3.agent import HUMAN_ACTIONS
from arcagi3.utils.context import SessionContext, GameProgress


class DummyProvider:
    """Stub provider that records messages and returns fixed JSON."""

    class ModelConfig:
        class Pricing:
            input = 0
            output = 0

        pricing = Pricing()

        kwargs: Dict[str, Any] = {}
        is_multimodal = False

    model_config = ModelConfig()

    def __init__(self):
        self.last_messages: List[Dict[str, Any]] = []

    def call_with_tracking(self, context: SessionContext, messages):
        return self.call_provider(messages)

    def call_provider(self, messages):
        self.last_messages = messages
        # Return swarm coordination response with actions for each game
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"actions":[{"game_id":"game1","action":"ACTION1"},{"game_id":"game2","action":"ACTION2"}]}'
                    }
                }
            ]
        }

    def extract_usage(self, response):
        return 0, 0, 0

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


class DummyGameClient:
    """Game client stub that never hits the network."""

    def reset_game(self, card_id: str, game_id: str, guid=None):
        return {
            "guid": f"dummy-guid-{game_id}",
            "score": 0,
            "state": "IN_PROGRESS",
            "frame": [[[0]]],
            "available_actions": ["1", "2", "6"],
        }

    def execute_action(self, action_name: str, data: Dict[str, Any]):
        return {
            "guid": data.get("guid", "dummy-guid"),
            "score": 1,
            "state": "GAME_OVER",
            "frame": [[[0]]],
        }


def _make_agent(monkeypatch, use_vision: bool = False) -> SwarmAgent:
    import arcagi3.agent as agent_module
    import arcagi3.adapters as adapters_module
    from arcagi3.utils import task_utils

    dummy_provider = DummyProvider()
    dummy_provider.model_config.is_multimodal = use_vision
    monkeypatch.setattr(agent_module, "create_provider", lambda config: dummy_provider)
    monkeypatch.setattr(adapters_module, "create_provider", lambda config: dummy_provider)
    monkeypatch.setattr(
        task_utils,
        "read_models_config",
        lambda config: type('ModelConfig', (), {
            'provider': 'dummy',
            'pricing': type('Pricing', (), {'input': 0, 'output': 0})(),
            'kwargs': {},
            'is_multimodal': use_vision,
        })()
    )

    game_client = DummyGameClient()
    agent = SwarmAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        use_vision=use_vision,
        checkpoint_frequency=0,
    )
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_coordinate_actions_includes_all_game_summaries(monkeypatch):
    """Test that coordinate_actions includes summaries for all games."""
    agent = _make_agent(monkeypatch)

    coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
    game_contexts = {}
    
    for game_id in ["game1", "game2"]:
        ctx = SessionContext(game=GameProgress(game_id=game_id))
        ctx.set_available_actions(["1", "2"])
        ctx.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
        game_contexts[game_id] = ctx

    plan = agent.coordinate_actions(coordinator_context, game_contexts)
    
    assert "actions" in plan
    assert len(plan["actions"]) == 2

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should include both game IDs in the prompt
    assert "game1" in str(user_content)
    assert "game2" in str(user_content)


def test_coordinate_actions_includes_game_state_info(monkeypatch):
    """Test that coordinate_actions includes state, score, and available actions."""
    agent = _make_agent(monkeypatch)

    coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
    ctx = SessionContext(game=GameProgress(game_id="test-game"))
    ctx.set_available_actions(["1", "6"])
    ctx.update(frame_grids=[[[0]]], current_score=5, current_state="IN_PROGRESS")
    game_contexts = {"test-game": ctx}

    plan = agent.coordinate_actions(coordinator_context, game_contexts)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should include game state information
    assert "test-game" in str(user_content)
    assert "Score: 5" in str(user_content)
    assert "IN_PROGRESS" in str(user_content)
    assert "Move Up" in str(user_content)  # ACTION1
    assert "Click object" in str(user_content)  # ACTION6


def test_coordinate_actions_includes_memory_when_present(monkeypatch):
    """Test that coordinate_actions includes memory from game contexts."""
    agent = _make_agent(monkeypatch)

    coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
    ctx = SessionContext(game=GameProgress(game_id="test-game"))
    ctx.set_available_actions(["1"])
    ctx.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    ctx.datastore["memory_prompt"] = "Test memory: walls at (5,5)"
    game_contexts = {"test-game": ctx}

    plan = agent.coordinate_actions(coordinator_context, game_contexts)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    assert "Test memory: walls at (5,5)" in str(user_content)


def test_step_uses_coordinate_actions(monkeypatch):
    """Test that step() calls coordinate_actions and extracts action."""
    agent = _make_agent(monkeypatch)

    context = SessionContext(game=GameProgress(game_id="test-game"))
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    
    assert step.action["action"] == "ACTION1"
    assert "swarm_plan" in step.reasoning


def test_step_validates_action_against_available(monkeypatch):
    """Test that step validates actions against available_actions."""
    agent = _make_agent(monkeypatch)
    
    # Override provider to return invalid action
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"actions":[{"game_id":"test-game","action":"ACTION3"}]}'
                    }
                }
            ]
        }
    
    provider.call_provider = mock_call_provider

    context = SessionContext(game=GameProgress(game_id="test-game"))
    context.set_available_actions(["1", "6"])  # ACTION3 not available
    
    import pytest
    with pytest.raises(ValueError, match="Invalid action"):
        agent.step(context)


def test_play_swarm_initializes_all_games(monkeypatch):
    """Test that play_swarm initializes contexts for all game_ids."""
    agent = _make_agent(monkeypatch)

    results = agent.play_swarm(["game1", "game2"], max_rounds=1)
    
    assert "game1" in results
    assert "game2" in results
    assert results["game1"]["state"] in ("IN_PROGRESS", "GAME_OVER", "WIN")
    assert results["game2"]["state"] in ("IN_PROGRESS", "GAME_OVER", "WIN")


def test_play_swarm_executes_actions_for_all_games(monkeypatch):
    """Test that play_swarm executes actions for all games in plan."""
    agent = _make_agent(monkeypatch)

    results = agent.play_swarm(["game1", "game2"], max_rounds=1)
    
    # Both games should have actions taken
    assert results["game1"]["actions_taken"] > 0
    assert results["game2"]["actions_taken"] > 0


def test_play_swarm_stops_when_all_games_terminate(monkeypatch):
    """Test that play_swarm stops early when all games reach terminal states."""
    agent = _make_agent(monkeypatch)
    
    # Override game client to return WIN/GAME_OVER immediately
    original_execute = agent.game_client.execute_action
    
    def mock_execute(action_name: str, data: Dict[str, Any]):
        return {
            "guid": data.get("guid", "dummy-guid"),
            "score": 10,
            "state": "WIN",
            "frame": [[[0]]],
        }
    
    agent.game_client.execute_action = mock_execute

    results = agent.play_swarm(["game1"], max_rounds=10)
    
    # Should stop early (not use all 10 rounds)
    assert results["game1"]["state"] == "WIN"


def test_play_swarm_skips_invalid_actions(monkeypatch):
    """Test that play_swarm skips actions for invalid game_ids."""
    agent = _make_agent(monkeypatch)
    
    # Override provider to return action for non-existent game
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"actions":[{"game_id":"nonexistent","action":"ACTION1"},{"game_id":"game1","action":"ACTION1"}]}'
                    }
                }
            ]
        }
    
    provider.call_provider = mock_call_provider

    results = agent.play_swarm(["game1"], max_rounds=1)
    
    # Should still work for valid game_id
    assert "game1" in results
    assert results["game1"]["actions_taken"] > 0


def test_vision_mode_includes_images_in_summaries(monkeypatch):
    """Test that vision mode includes image blocks in game summaries."""
    agent = _make_agent(monkeypatch, use_vision=True)

    coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
    ctx = SessionContext(game=GameProgress(game_id="test-game"))
    ctx.set_available_actions(["1"])
    ctx.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    game_contexts = {"test-game": ctx}

    plan = agent.coordinate_actions(coordinator_context, game_contexts)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should contain image blocks, not text grid
    assert any("image_url" in str(block) for block in user_content)
    assert "Frame 0:" not in str(user_content)


def test_text_mode_includes_grid_text_in_summaries(monkeypatch):
    """Test that text mode includes grid text in game summaries."""
    agent = _make_agent(monkeypatch, use_vision=False)

    coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
    ctx = SessionContext(game=GameProgress(game_id="test-game"))
    ctx.set_available_actions(["1"])
    ctx.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    game_contexts = {"test-game": ctx}

    plan = agent.coordinate_actions(coordinator_context, game_contexts)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_content = messages[-1]["content"]
    
    # Should contain text grid format
    assert "Frame 0:" in str(user_content)
    assert "image_url" not in str(user_content)


def test_step_raises_error_when_no_actions_in_response(monkeypatch):
    """Test that step raises error when coordinate_actions returns empty actions list."""
    agent = _make_agent(monkeypatch)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"actions":[]}'
                    }
                }
            ]
        }
    
    provider.call_provider = mock_call_provider

    context = SessionContext(game=GameProgress(game_id="test-game"))
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    
    import pytest
    with pytest.raises(ValueError, match="No action in swarm response"):
        agent.step(context)


def test_step_handles_action_with_coordinates(monkeypatch):
    """Test that step handles ACTION6 with coordinates correctly."""
    agent = _make_agent(monkeypatch)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"actions":[{"game_id":"test-game","action":"ACTION6","x":10,"y":20}]}'
                    }
                }
            ]
        }
    
    provider.call_provider = mock_call_provider

    context = SessionContext(game=GameProgress(game_id="test-game"))
    context.set_available_actions(["6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)
    
    assert step.action["action"] == "ACTION6"
    assert step.action.get("x") == 10
    assert step.action.get("y") == 20

