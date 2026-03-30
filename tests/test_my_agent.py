import arcagi3.agent as agent_module
from arcagi3.my_agent import MyAgent
from arcagi3.utils.context import SessionContext


class DummyGameClient:
    pass


def test_provider_is_initialized_lazily(monkeypatch):
    provider = object()
    calls = []

    def fake_create_provider(config):
        calls.append(config)
        return provider

    monkeypatch.setattr(agent_module, "create_provider", fake_create_provider)

    agent = MyAgent(
        config="dummy-config",
        game_client=DummyGameClient(),
        card_id="local-test",
        checkpoint_frequency=0,
    )

    assert calls == []
    assert agent.provider is provider
    assert calls == ["dummy-config"]


def test_my_agent_cycles_actions_without_touching_provider(monkeypatch):
    def fail_create_provider(config):
        raise AssertionError("provider should not be created for this test agent")

    monkeypatch.setattr(agent_module, "create_provider", fail_create_provider)

    agent = MyAgent(
        config="dummy-config",
        game_client=DummyGameClient(),
        card_id="local-test",
        checkpoint_frequency=0,
    )
    context = SessionContext()
    context.set_available_actions(["6", "1", "2"])

    first = agent.step(context)
    second = agent.step(context)

    assert first.action == {"action": "ACTION1"}
    assert second.action == {"action": "ACTION2"}
    assert context.datastore["action_history"] == ["ACTION1", "ACTION2"]


def test_my_agent_clicks_center_when_action6_is_only_option(monkeypatch):
    def fail_create_provider(config):
        raise AssertionError("provider should not be created for this test agent")

    monkeypatch.setattr(agent_module, "create_provider", fail_create_provider)

    agent = MyAgent(
        config="dummy-config",
        game_client=DummyGameClient(),
        card_id="local-test",
        checkpoint_frequency=0,
    )
    context = SessionContext()
    context.set_available_actions(["6"])

    step = agent.step(context)

    assert step.action == {"action": "ACTION6", "x": 63, "y": 63}
