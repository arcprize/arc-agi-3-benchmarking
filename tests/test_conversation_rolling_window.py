import copy
import importlib

import pytest

import arcagi3.agent as agent_module
from arcagi3.conversation_rolling_window import ConversationRollingWindow
from arcagi3.runner import _build_default_registry
from arcagi3.utils.context import SessionContext


class DummyGameClient:
    pass


class DummyProvider:
    class ModelConfig:
        class Pricing:
            input = 1_000_000.0
            output = 2_000_000.0

        pricing = Pricing()
        model_name = "dummy-model"

    model_config = ModelConfig()

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_messages = []

    def call_with_tracking(self, context, messages, **kwargs):
        self.last_messages = copy.deepcopy(messages)
        return self.call_provider(messages)

    def call_provider(self, messages):
        return {"choices": [{"message": {"content": self.response_text}}]}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response):
        return 10, 5, 2


def _make_agent(monkeypatch, tmp_path, provider: DummyProvider) -> ConversationRollingWindow:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_module, "create_provider", lambda config: provider)
    return ConversationRollingWindow(
        config="dummy-config",
        game_client=DummyGameClient(),
        card_id="local-test",
        checkpoint_frequency=0,
    )


def test_conversation_rolling_window_agent_imports_into_current_repo():
    importlib.import_module("arcagi3.conversation_rolling_window.agent")


def test_conversation_rolling_window_is_registered():
    runner = _build_default_registry()
    agent_names = [entry["name"] for entry in runner.list_agents()]
    assert "conversation_rolling_window" in agent_names


def test_step_returns_simple_action_and_records_conversation(monkeypatch, tmp_path):
    provider = DummyProvider("I will try ACTION1")
    agent = _make_agent(monkeypatch, tmp_path, provider)

    context = SessionContext(checkpoint_id="test-checkpoint")
    context.set_game_identity(game_id="dummy-game", guid="guid-1")
    context.set_available_actions(["1", "6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)

    assert step.action == {"action": "ACTION1"}
    assert step.reasoning["output"] == "I will try ACTION1"
    conversation = context.datastore["conversation_rolling_window.conversation"]
    assert [msg["role"] for msg in conversation] == ["system", "user", "assistant"]
    assert "Available actions" in provider.last_messages[-1]["content"]


def test_step_parses_click_coordinates(monkeypatch, tmp_path):
    provider = DummyProvider("Maybe ACTION1 first, but final answer is ACTION6 100 42")
    agent = _make_agent(monkeypatch, tmp_path, provider)

    context = SessionContext(checkpoint_id="test-checkpoint")
    context.set_game_identity(game_id="dummy-game", guid="guid-1")
    context.set_available_actions(["6"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")

    step = agent.step(context)

    assert step.action == {"action": "ACTION6", "x": 100, "y": 42}
