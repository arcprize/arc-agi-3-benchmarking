import json

import pytest

from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import build_stateful_runtime_adapter
from benchmarking.runtime_state import ModelTurnRequest
from tests.unit.test_google_continuation import make_adapter, reply  # noqa: F401


def turn(state):
    return ModelTurnRequest(
        system_prompt="system",
        new_messages=[Message(role="user", content="frame")],
        previous_state=state,
        request_config={
            "model": "gemini-test",
            "max_output_tokens": 1000000,
            "thinking_config": {"include_thoughts": True},
        },
    )


def runtime(model):
    return build_stateful_runtime_adapter(
        model_adapter=model,
        config_id="test",
        runtime_config={
            "sdk": "google-genai",
            "api": "generate_content",
            "state": "continuous_conversation",
        },
    )


def test_native_replay_continuation_and_compaction(make_adapter):  # noqa: F811
    signed = {"text": "think", "thought": True, "thoughtSignature": "opaque-secret"}
    model, calls = make_adapter(
        [
            reply("CONTINUATION", [signed], "token", {"thoughtsTokenCount": 7}),
            reply("STOP", [{"text": "ACTION1"}], usage={"candidatesTokenCount": 2}),
            reply("STOP", [{"text": "ACTION2"}]),
            reply("STOP", [{"text": "ACTION3"}]),
        ]
    )
    adapter = runtime(model)
    initial = adapter.initial_state()
    first = adapter.invoke_turn(turn(initial))
    assert initial.payload == {"contents": []}
    assert first.response.usage.output_tokens == 9
    assert first.response.reasoning_text == "think"
    second = adapter.invoke_turn(turn(first.state))
    body = json.loads(calls[2].content)
    assert body["contents"][1] == {
        "role": "model",
        "parts": [signed, {"text": "ACTION1"}],
    }
    assert "continuationToken" not in body
    assert "opaque-secret" not in json.dumps(second.sanitized_request)
    assert "opaque-secret" not in json.dumps(second.readable_request_messages)
    retained = adapter.unwind_latest_accepted_turn(first.state)
    rebuilt = adapter.rebuild_after_compaction(
        Message(role="user", content="summary"), [retained]
    )
    assert rebuilt.payload["contents"][2]["parts"][0] == signed
    adapter.invoke_turn(turn(rebuilt))
    assert json.loads(calls[3].content)["contents"][0]["parts"] == [{"text": "summary"}]


def test_failed_slice_does_not_mutate_accepted_history(make_adapter):  # noqa: F811
    model, calls = make_adapter(
        [reply("CONTINUATION", token="same"), reply("CONTINUATION", token="same")]
    )
    adapter = runtime(model)
    state = adapter.initial_state()
    with pytest.raises(Exception, match="repeated"):
        adapter.invoke_turn(turn(state))
    assert state.payload == {"contents": []}
    assert state.accepted_turns == []
