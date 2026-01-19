from typing import Any, Dict, List

from PIL import Image

from arcagi3.examples.state_transform_adcr import StateTransformADCRAgent, StateTransformPayload
from arcagi3.utils.context import SessionContext


class DummyProvider:
    """Stub provider that records messages and returns queued responses."""

    class ModelConfig:
        class Pricing:
            input = 0
            output = 0

        pricing = Pricing()
        kwargs: Dict[str, Any] = {"memory_word_limit": 100}
        is_multimodal = True

    model_config = ModelConfig()

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[List[Dict[str, Any]]] = []

    def call_with_tracking(self, context: SessionContext, messages):
        return self.call_provider(messages)

    def call_provider(self, messages):
        self.calls.append(messages)
        content = self.responses.pop(0) if self.responses else "{}"
        return {"choices": [{"message": {"content": content}}]}

    def extract_usage(self, response):
        return 0, 0, 0

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


class DummyGameClient:
    def reset_game(self, card_id: str, game_id: str, guid=None):
        return {
            "guid": "dummy-guid",
            "score": 0,
            "state": "IN_PROGRESS",
            "frame": [[[0]]],
            "available_actions": ["1"],
        }

    def execute_action(self, action_name: str, data: Dict[str, Any]):
        return {
            "guid": data.get("guid", "dummy-guid"),
            "score": 1,
            "state": "GAME_OVER",
            "frame": [[[0]]],
        }


def _make_agent(monkeypatch, transform):
    import arcagi3.agent as agent_module
    import arcagi3.adapters as adapters_module
    from arcagi3.utils import task_utils

    responses = [
        "analysis text\n---\nupdated memory",
        '{"human_action":"ACTION1"}',
        '{"action":"ACTION1"}',
    ]
    dummy_provider = DummyProvider(responses)

    monkeypatch.setattr(agent_module, "create_provider", lambda config: dummy_provider)
    monkeypatch.setattr(adapters_module, "create_provider", lambda config: dummy_provider)
    monkeypatch.setattr(
        task_utils,
        "read_models_config",
        lambda config: type('ModelConfig', (), {
            'provider': 'dummy',
            'pricing': type('Pricing', (), {'input': 0, 'output': 0})(),
            'kwargs': {'memory_word_limit': 100},
            'is_multimodal': True,
        })()
    )

    agent = StateTransformADCRAgent(
        config="dummy-config",
        game_client=DummyGameClient(),
        card_id="local-test",
        max_actions=5,
        num_plays=1,
        max_episode_actions=0,
        use_vision=True,
        checkpoint_frequency=0,
        state_transform=transform,
    )
    agent._test_provider = dummy_provider  # type: ignore[attr-defined]
    return agent


def _make_context():
    context = SessionContext()
    context.set_available_actions(["1"])
    context.update(frame_grids=[[[0]]], current_score=0, current_state="IN_PROGRESS")
    context.datastore["previous_action"] = {"human_action": "ACTION1"}
    return context


def test_state_transform_used_in_all_substeps(monkeypatch):
    call_count = {"count": 0}

    def transform(context: SessionContext) -> StateTransformPayload:
        call_count["count"] += 1
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        return StateTransformPayload(text="TRANSFORMED", images=[img])

    agent = _make_agent(monkeypatch, transform)
    context = _make_context()

    agent.step(context)

    provider = agent._test_provider  # type: ignore[attr-defined]
    assert call_count["count"] == 1
    assert len(provider.calls) == 3

    for messages in provider.calls:
        user_content = messages[-1]["content"]
        content_text = " ".join(
            block.get("text", "") for block in user_content if block.get("type") == "text"
        )
        assert "TRANSFORMED" in content_text
        # The analyze instruction template mentions "Frame"; we only want to ensure the
        # raw frame grid dumps ("Frame 0:", etc.) were not included.
        assert "Frame 0" not in content_text
        assert any(block.get("type") == "image_url" for block in user_content)


def test_state_transform_required_argument():
    try:
        StateTransformADCRAgent(
            config="dummy-config",
            game_client=DummyGameClient(),
            card_id="local-test",
        )
    except TypeError:
        return
    raise AssertionError("StateTransformADCRAgent should require a state_transform argument")


def test_text_only_transform(monkeypatch):
    """Test that text-only transform (no images) works correctly."""
    def transform(context: SessionContext) -> StateTransformPayload:
        return StateTransformPayload(text="Text-only transform", images=None)

    agent = _make_agent(monkeypatch, transform)
    context = _make_context()

    agent.step(context)

    provider = agent._test_provider  # type: ignore[attr-defined]
    # Should have called analyze, decide, convert
    assert len(provider.calls) == 3
    
    # Check that text transform is included
    for messages in provider.calls:
        user_content = messages[-1]["content"]
        content_text = " ".join(
            block.get("text", "") for block in user_content if block.get("type") == "text"
        )
        assert "Text-only transform" in content_text


def test_full_step_loop_with_transform(monkeypatch):
    """Test complete step() method works end-to-end with transform."""
    call_count = {"count": 0}

    def transform(context: SessionContext) -> StateTransformPayload:
        call_count["count"] += 1
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        return StateTransformPayload(text="TRANSFORMED", images=[img])

    agent = _make_agent(monkeypatch, transform)
    context = _make_context()

    step = agent.step(context)

    # Should have called transform once
    assert call_count["count"] == 1
    # Should have called analyze, decide, convert
    provider = agent._test_provider  # type: ignore[attr-defined]
    assert len(provider.calls) == 3
    # Should return valid action
    assert step.action["action"] == "ACTION1"
    assert "analysis" in step.reasoning

