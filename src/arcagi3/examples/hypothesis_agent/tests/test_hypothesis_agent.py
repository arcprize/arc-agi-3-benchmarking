"""Tests for HypothesisAgent."""

from typing import Any, Dict, List

from arcagi3.examples.hypothesis_agent import HypothesisAgent
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
            # Hypothesis update response
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"hypotheses":[{"hypothesis":"Walls block movement","status":"confirmed","evidence":"Observed in turn 1"}],"active_hypothesis":"Walls block movement","active_experiment":"Test if diagonal movement works"}'
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
                            "content": "Analysis: The action was successful.\n---\nMemory: Walls confirmed at (5,5)"
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
                            "content": '{"human_action":"Move Up to test hypothesis"}'
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


def _make_agent(monkeypatch, hypothesis_interval: int = 3) -> HypothesisAgent:
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
    agent = HypothesisAgent(
        config="dummy-config",
        game_client=game_client,
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        use_vision=False,
        hypothesis_interval=hypothesis_interval,
        checkpoint_frequency=0,
    )
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def test_update_hypotheses_step_extracts_hypotheses(monkeypatch):
    """Test that update_hypotheses_step extracts hypotheses from model response."""
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

    hypotheses, active_hyp, active_exp = agent.update_hypotheses_step(context)
    
    assert len(hypotheses) > 0
    assert "hypothesis" in hypotheses[0]
    assert active_hyp
    assert active_exp


def test_hypotheses_included_in_decide_step(monkeypatch):
    """Test that hypotheses are included in decide_human_action_step prompt."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["hypotheses"] = [
        {"hypothesis": "Walls block movement", "status": "confirmed", "evidence": "Turn 1"}
    ]
    context.datastore["active_hypothesis"] = "Walls block movement"
    context.datastore["active_experiment"] = "Test diagonal movement"

    analysis = "Some analysis"
    result = agent.decide_human_action_step(context, analysis)

    provider = agent._test_provider  # type: ignore[attr-defined]
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Should include hypotheses in prompt
    assert "Walls block movement" in str(user_msg)
    assert "confirmed" in str(user_msg)
    assert "Active hypothesis" in str(user_msg)
    assert "Active experiment" in str(user_msg)


def test_maybe_update_hypotheses_respects_interval(monkeypatch):
    """Test that _maybe_update_hypotheses only runs at specified intervals."""
    agent = _make_agent(monkeypatch, hypothesis_interval=3)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    
    # Action 1 - should not trigger (interval is 3)
    context.set_counters(action_counter=1)
    provider = agent._test_provider  # type: ignore[attr-defined]
    provider.call_count = 0
    agent._maybe_update_hypotheses(context)
    assert provider.call_count == 0  # Should not call
    
    # Action 3 - should trigger
    context.set_counters(action_counter=3)
    agent._maybe_update_hypotheses(context)
    assert provider.call_count > 0  # Should call


def test_step_includes_hypotheses_in_reasoning(monkeypatch):
    """Test that step includes active hypothesis and experiment in reasoning."""
    agent = _make_agent(monkeypatch)

    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "Move Up"}
    context.datastore["previous_prompt"] = "Previous prompt"
    context.datastore["active_hypothesis"] = "Test hypothesis"
    context.datastore["active_experiment"] = "Test experiment"
    
    # Add action history so hypothesis update can run
    context.append_action_record(
        GameActionRecord(
            action_num=3,
            action="ACTION1",
            action_data=None,
            reasoning=None,
            result_score=0,
            result_state="IN_PROGRESS",
            cost=Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0),
        )
    )
    context.set_counters(action_counter=3)

    step = agent.step(context)
    
    assert "active_hypothesis" in step.reasoning
    assert "active_experiment" in step.reasoning


def test_format_hypotheses_formats_correctly(monkeypatch):
    """Test that _format_hypotheses formats hypothesis list correctly."""
    agent = _make_agent(monkeypatch)

    hypotheses = [
        {"hypothesis": "Rule 1", "status": "confirmed", "evidence": "Evidence 1"},
        {"hypothesis": "Rule 2", "status": "candidate", "evidence": ""},
    ]
    
    formatted = agent._format_hypotheses(hypotheses)
    
    assert "Rule 1" in formatted
    assert "Rule 2" in formatted
    assert "[confirmed]" in formatted
    assert "[candidate]" in formatted
    assert "Evidence 1" in formatted


def test_format_hypotheses_handles_empty_list(monkeypatch):
    """Test that _format_hypotheses handles empty list."""
    agent = _make_agent(monkeypatch)

    formatted = agent._format_hypotheses([])
    
    assert formatted == "None"


def test_hypothesis_update_uses_recent_action_window(monkeypatch):
    """Test that hypothesis update uses only recent actions within window."""
    agent = _make_agent(monkeypatch, hypothesis_interval=1)

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
    
    agent._maybe_update_hypotheses(context)
    
    # Should have called provider
    assert provider.call_count > 0
    messages = provider.last_messages
    user_msg = messages[-1]["content"]
    
    # Should only include recent actions (last 5 by default)
    # Should include action #9, but not necessarily #1
    assert "#9" in str(user_msg)


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


def test_full_adcr_loop(monkeypatch):
    """Test complete ADCR loop works end-to-end."""
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
    
    # Should have called: analyze (if previous_action exists), decide, convert
    # At minimum decide and convert should be called
    assert provider.call_count >= 2
    assert step.action["action"] == "ACTION1"
    assert "human_action" in step.reasoning


def test_step_raises_error_when_human_action_missing(monkeypatch):
    """Test that step raises error when human_action is missing from decide step."""
    agent = _make_agent(monkeypatch)
    
    provider = agent._test_provider  # type: ignore[attr-defined]
    
    def mock_call_provider(messages):
        provider.last_messages = messages
        provider.call_count += 1
        if provider.call_count == 1:
            # Hypothesis update (if triggered)
            return {"choices": [{"message": {"content": '{"hypotheses":[]}'}}]}
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
            # Hypothesis update (if triggered)
            return {"choices": [{"message": {"content": '{"hypotheses":[]}'}}]}
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

