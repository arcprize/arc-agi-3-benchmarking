import json

import pytest

from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import build_stateful_runtime_adapter
from benchmarking.runtime_state import ModelTurnRequest, RuntimeState
from tests.unit.test_google_continuation import (  # noqa: F401
    make_adapter,
    make_stream_adapter,
    reply,
    stream_response,
)


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
        model_adapter=build_model_runtime_adapter(
            client=model._client,
            config_id="test",
            runtime_config={
                "sdk": "google-genai",
                "api": "generate_content",
                "state": "continuous_conversation",
            },
        ),
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("call_id", [None, "call-123"])
def test_native_action_is_replayed_as_text_with_ordinary_frame_inputs(
    make_adapter,  # noqa: F811
    make_stream_adapter,  # noqa: F811
    stream,
    call_id,
):
    native_call = {"name": "google:ACTION6", "args": {"x": 22, "y": 29}}
    if call_id is not None:
        native_call["id"] = call_id
    thought = {"thought": True, "text": "think", "thoughtSignature": "dGhvdWdodA=="}
    part = {"functionCall": native_call, "thoughtSignature": "Y2FsbA=="}
    next_call = {"functionCall": {"name": "ACTION3", "args": {}}}
    replies = [
        reply("STOP", [thought, part], usage={"candidatesTokenCount": 2}),
        reply("STOP", [next_call], usage={"candidatesTokenCount": 2}),
        reply("STOP", [{"text": "ACTION4"}], usage={"candidatesTokenCount": 2}),
    ]
    if stream:
        model, calls = make_stream_adapter([stream_response(r) for r in replies])
    else:
        model, calls = make_adapter(replies)
    adapter = runtime(model)
    initial = adapter.initial_state()
    first_request = turn(initial)
    first_request.request_config["stream"] = stream
    first = adapter.invoke_turn(first_request)
    assert json.loads(first.response.output_text) == {
        "actions": [{"action_type": "ACTION6", "x": 22, "y": 29}]
    }
    assert initial.payload == {"contents": []}
    # Retain the true response only for diagnostics; accepted provider history
    # presents the single action as if it had originally arrived as text.
    assert first.response.raw_response["candidates"][0]["content"]["parts"] == [
        thought,
        part,
    ]
    expected_model = {
        "role": "model",
        "parts": [thought, {"text": first.response.output_text}],
    }
    assert first.state.payload["contents"][-1] == expected_model
    persisted_state = first.state.model_dump_json()
    assert "functionCall" not in persisted_state
    assert "call-123" not in persisted_state
    second_request = turn(RuntimeState.model_validate_json(persisted_state))
    second_request.request_config["stream"] = stream
    second = adapter.invoke_turn(second_request)
    contents = json.loads(calls[1].content)["contents"]
    assert contents[-2] == expected_model
    assert contents[-1] == {"role": "user", "parts": [{"text": "frame"}]}
    assert second.readable_request_messages[-2] == {
        "role": "assistant",
        "content": "think\n" + first.response.output_text,
    }
    assert second.readable_request_messages[-1] == {"role": "user", "content": "frame"}
    wire = json.dumps(contents)
    for marker in [
        "functionCall",
        "function_call",
        "functionResponse",
        "call-123",
        "Y2FsbA==",
    ]:
        assert marker not in wire
    assert "dGhvdWdodA==" not in json.dumps(second.readable_request_messages)
    assert "dGhvdWdodA==" not in json.dumps(second.sanitized_request)

    buffered = adapter.buffer_inputs(
        first.state, [Message(role="user", content="frame")]
    )
    assert buffered.payload["contents"][-1] == contents[-1]
    assert first.state.payload["contents"][-1] == expected_model

    retained = adapter.unwind_latest_accepted_turn(second.state)
    assert retained is not None
    rebuilt = adapter.rebuild_after_compaction(
        Message(role="user", content="summary"), [retained]
    )
    assert rebuilt.payload["contents"] == [
        {"role": "user", "parts": [{"text": "summary"}]},
        {"role": "user", "parts": [{"text": "frame"}]},
        {"role": "model", "parts": [{"text": second.response.output_text}]},
    ]
    assert retained.native_items[0]["parts"] == [{"text": "frame"}]
    assert rebuilt.accepted_turns[0].start_item == 1
    assert rebuilt.accepted_turns[0].end_item == 3
    third_request = turn(rebuilt)
    third_request.request_config["stream"] = stream
    adapter.invoke_turn(third_request)
    third_contents = json.loads(calls[2].content)["contents"]
    assert third_contents[-1] == {"role": "user", "parts": [{"text": "frame"}]}
    assert "functionCall" not in json.dumps(third_contents)
    assert "functionResponse" not in json.dumps(third_contents)


def test_multiple_native_calls_reject_without_accepting_state(make_adapter):  # noqa: F811
    from benchmarking.exceptions import InvalidProviderResponseError

    parts = [
        {"functionCall": {"name": "ACTION1", "args": {}}},
        {"functionCall": {"name": "ACTION2", "args": {}}},
        {"text": "ACTION3"},
    ]
    model, _ = make_adapter(
        [reply("STOP", parts, usage={"promptTokenCount": 7, "candidatesTokenCount": 2})]
    )
    adapter = runtime(model)
    initial = adapter.initial_state()
    with pytest.raises(InvalidProviderResponseError) as exc:
        adapter.invoke_turn(turn(initial))
    assert exc.value.usage.total_tokens == 9
    assert initial.payload == {"contents": []}
    assert initial.accepted_turns == []
