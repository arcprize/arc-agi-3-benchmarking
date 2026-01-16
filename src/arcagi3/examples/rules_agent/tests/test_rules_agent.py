"""Tests for RulesAgent."""

from typing import Any, Dict, List

from arcagi3.examples.rules_agent import RulesAgent
from arcagi3.agent import HUMAN_ACTIONS
from arcagi3.utils.context import SessionContext
from arcagi3.schemas import GameActionRecord, Cost


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
        self.call_count = 0

    def call_with_tracking(self, context: SessionContext, messages):
        self.call_count += 1
        return self.call_provider(messages)

    def call_provider(self, messages):
        self.last_messages = messages
        # Return different responses based on call count
        if self.call_count == 1:
            # Rules extraction response
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"rules_summary":"Maze navigation game","rules":["Walls block movement","Goal is at (10,10)"],"experiments":["Test diagonal movement"]}'
                        }
                    }
                ]
            }
        elif self.call_count == 2:
            # Analyze response
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Analysis: The action was successful.\n---\nMemory: Walls confirmed"
                        }
                    }
                ]
            }
        elif self.call_count == 3:
            # Decide response
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"human_action":"Move Up following rules"}'
                        }
                    }
                ]
            }
        else:
            # Convert response
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"ACTION1"}'
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
            "guid": "dummy-guid",
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


def _make_agent(monkeypatch, rules_interval: int = 5) -> RulesAgent:
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
    agent = RulesAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        use_vision=False,
        rules_interval=rules_interval,
        checkpoint_frequency=0,
    )
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_extract_rules_step_extracts_rules(monkeypatch):
    """Test that extract_rules_step extracts rules from model response."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    
    # Add some action history
    context.append_action_record(
        GameActionRecord(
            action_num=1,
            action="ACTION1",
            action_data=None,
            reasoning=None,
            result_score=0,
            result_state="IN_PROGRESS",
            cost=Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0),
        )
    )

    summary, rules_list, experiments = agent.extract_rules_step(context)
    
    assert summary
    assert len(rules_list) > 0
    assert len(experiments) > 0


def test_rules_included_in_decide_step(monkeypatch):
    """Test that rules are included in decide_human_action_step prompt."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["rules_summary"] = "Maze navigation game"
    context.datastore["rules_list"] = ["Walls block movement", "Goal is at (10,10)"]

    analysis = "Some analysis"
    result = agent.decide_human_action_step(context, analysis)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Should include rules in prompt
    assert "Maze navigation game" in str(user_msg)
    assert "Walls block movement" in str(user_msg)
    assert "Goal is at (10,10)" in str(user_msg)
    assert "Rules summary:" in str(user_msg)
    assert "Rules:" in str(user_msg)


def test_maybe_extract_rules_respects_interval(monkeypatch):
    """Test that _maybe_extract_rules only runs at specified intervals."""
    agent = _make_agent(monkeypatch, rules_interval=5)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    
    # Action 1 - should not trigger (interval is 5)
    context.set_counters(action_counter=1)
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0
    agent._maybe_extract_rules(context)
    assert provider.call_count == 0  # Should not call
    
    # Action 5 - should trigger
    context.set_counters(action_counter=5)
    agent._maybe_extract_rules(context)
    assert provider.call_count > 0  # Should call


def test_step_includes_rules_in_reasoning(monkeypatch):
    """Test that step includes rules_summary in reasoning."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "Move Up"}
    context.datastore["previous_prompt"] = "Previous prompt"
    context.datastore["rules_summary"] = "Test rules summary"
    
    # Add action history so rules extraction can run
    context.append_action_record(
        GameActionRecord(
            action_num=5,
            action="ACTION1",
            action_data=None,
            reasoning=None,
            result_score=0,
            result_state="IN_PROGRESS",
            cost=Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0),
        )
    )
    context.set_counters(action_counter=5)

    step = agent.step(context)
    
    assert "rules_summary" in step.reasoning
    assert step.reasoning["rules_summary"] == "Test rules summary"


def test_rules_extraction_uses_recent_action_window(monkeypatch):
    """Test that rules extraction uses only recent actions within window."""
    agent = _make_agent(monkeypatch, rules_interval=1)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    
    # Add many actions, but window is 5
    for i in range(1, 10):
        context.append_action_record(
            GameActionRecord(
                action_num=i,
                action="ACTION1",
                action_data=None,
                reasoning=None,
                result_score=0,
                result_state="IN_PROGRESS",
                cost=Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0),
            )
        )
    
    context.set_counters(action_counter=9)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0
    
    agent._maybe_extract_rules(context)
    
    # Should have called provider
    assert provider.call_count > 0
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Should only include recent actions (last 5 by default)
    # Should include action #9, but not necessarily #1
    assert "#9" in str(user_msg)


def test_rules_extraction_includes_previous_rules(monkeypatch):
    """Test that rules extraction prompt includes previous rules for updating."""
    agent = _make_agent(monkeypatch, rules_interval=1)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["rules_summary"] = "Previous summary"
    context.datastore["rules_list"] = ["Previous rule 1", "Previous rule 2"]
    context.datastore["rules_experiments"] = ["Previous experiment"]
    
    context.append_action_record(
        GameActionRecord(
            action_num=1,
            action="ACTION1",
            action_data=None,
            reasoning=None,
            result_score=0,
            result_state="IN_PROGRESS",
            cost=Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0),
        )
    )
    context.set_counters(action_counter=1)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0
    
    agent._maybe_extract_rules(context)
    
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Should include previous rules in prompt
    assert "Previous summary" in str(user_msg)
    assert "Previous rule 1" in str(user_msg)
    assert "Previous experiment" in str(user_msg)


def test_first_step_handles_no_previous_action(monkeypatch):
    """Test that first step handles missing previous_action gracefully."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    # No previous_action set

    # First step should work - analyze returns "no previous action"
    step = agent.step(context)
    
    assert step.action["action"] == "ACTION1"
    assert step.reasoning["analysis"] == "no previous action"


def test_full_step_loop(monkeypatch):
    """Test complete step() method works end-to-end."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "Move Up"}
    context.datastore["previous_prompt"] = "Previous prompt"

    # Reset provider call count
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0

    step = agent.step(context)
    
    # Should have called: analyze, decide, convert (and maybe extract_rules if interval met)
    assert provider.call_count >= 3
    assert step.action["action"] == "ACTION1"
    assert "human_action" in step.reasoning
    assert "rules_summary" in step.reasoning


def test_step_raises_error_when_human_action_missing(monkeypatch):
    """Test that step raises error when human_action is missing from decide step."""
    agent = _make_agent(monkeypatch)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        provider.call_count += 1
        if provider.call_count == 1:
            # Rules extraction (if triggered)
            return {"choices": [{"message": {"content": '{"rules_summary":"","rules":[]}'}}]}
        elif provider.call_count == 2:
            # Analyze
            return {"choices": [{"message": {"content": "Analysis\n---\nMemory"}}]}
        else:
            # Decide - missing human_action
            return {"choices": [{"message": {"content": '{"reasoning":"test"}'}}]}
    
    provider.call_provider = mock_call_provider

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "Move Up"}
    context.datastore["previous_prompt"] = "Previous prompt"

    import pytest
    with pytest.raises(ValueError, match="No human_action in response"):
        agent.step(context)


def test_convert_human_to_game_action_step(monkeypatch):
    """Test that convert_human_to_game_action_step works correctly."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1", "6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    # Override provider to return convert response
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        provider.call_count += 1
        return {"choices": [{"message": {"content": '{"action":"ACTION1"}'}}]}
    
    provider.call_provider = mock_call_provider

    result = agent.convert_human_to_game_action_step(context, "Move Up")
    
    assert result["action"] == "ACTION1"
    assert provider.call_count == 1


def test_step_raises_error_when_action_missing_from_convert(monkeypatch):
    """Test that step raises error when action is missing from convert step."""
    agent = _make_agent(monkeypatch)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        provider.call_count += 1
        if provider.call_count == 1:
            # Rules extraction (if triggered)
            return {"choices": [{"message": {"content": '{"rules_summary":"","rules":[]}'}}]}
        elif provider.call_count == 2:
            # Analyze
            return {"choices": [{"message": {"content": "Analysis\n---\nMemory"}}]}
        elif provider.call_count == 3:
            # Decide
            return {"choices": [{"message": {"content": '{"human_action":"Move Up"}'}}]}
        else:
            # Convert - missing action
            return {"choices": [{"message": {"content": '{"reasoning":"test"}'}}]}
    
    provider.call_provider = mock_call_provider

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "Move Up"}
    context.datastore["previous_prompt"] = "Previous prompt"

    import pytest
    with pytest.raises(ValueError, match="No action in game action response"):
        agent.step(context)

