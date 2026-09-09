import json
from types import SimpleNamespace

import pytest
from google.genai import errors as google_genai_errors

from benchmarking.exceptions import ContextOverflowError
from benchmarking.google_runtime import (
    GoogleContinuousConversationRuntimeAdapter,
    serialize_interaction_steps,
)
from benchmarking.runtime_adapters import (
    GoogleGenAIInteractionsAdapter,
    build_model_runtime_adapter,
)
from benchmarking.runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
)
from benchmarking.runtime_registry import (
    ADAPTER_DESCRIPTORS,
    build_stateful_runtime_adapter,
)
from benchmarking.runtime_state import ModelTurnRequest


class _FakeInteractions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.interactions = _FakeInteractions(response)


class _FakeModelAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _SDKStep:
    def __init__(self, value):
        self.value = value

    def model_dump(self, *, mode):
        assert mode == "json"
        return dict(self.value)

    def __getattr__(self, key):
        try:
            return self.value[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _steps(turn):
    return [
        {
            "type": "thought",
            "signature": f"opaque-{turn}",
            "summary": [{"type": "text", "text": f"summary {turn}"}],
        },
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "ACTION1"}],
        },
    ]


def _raw_interaction():
    return SimpleNamespace(
        id="interaction-1",
        output_text="ACTION1",
        steps=[_SDKStep(step) for step in _steps(1)],
        usage=SimpleNamespace(
            total_input_tokens=100,
            total_output_tokens=20,
            total_thought_tokens=22,
            total_cached_tokens=15,
            total_tokens=142,
        ),
    )


def _response(turn):
    return ModelResponse(
        output_text="ACTION1",
        reasoning_text=f"summary {turn}",
        usage=NormalizedUsage(total_tokens=10),
        raw_response={"steps": _steps(turn)},
    )


def _turn(adapter, state, content="observation"):
    return ModelTurnRequest(
        system_prompt="system",
        new_messages=[Message(role="user", content=content)],
        request_config={
            "model": "gemini-3.8-flash",
            "store": False,
            "generation_config": {
                "max_output_tokens": 65_536,
                "thinking_level": "low",
                "thinking_summaries": "auto",
            },
        },
        previous_state=state,
    )


@pytest.mark.unit
class TestGoogleInteractionsAdapter:
    def test_builds_stateless_request_and_normalizes_response(self):
        client = _FakeClient(_raw_interaction())
        adapter = GoogleGenAIInteractionsAdapter(client)
        native_input = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "native"}],
            }
        ]

        response = adapter.invoke(
            ModelRequest(
                messages=[
                    Message(role="system", content="system"),
                    Message(role="user", content="normalized"),
                ],
                request_config={
                    "model": "gemini-3.8-flash",
                    "store": False,
                    "generation_config": {"thinking_level": "low"},
                },
                native_input=native_input,
            )
        )

        assert client.interactions.calls == [
            {
                "model": "gemini-3.8-flash",
                "store": False,
                "generation_config": {"thinking_level": "low"},
                "system_instruction": "system",
                "input": native_input,
            }
        ]
        assert response.output_text == "ACTION1"
        assert response.reasoning_text == "summary 1"
        assert response.response_id == "interaction-1"
        assert response.usage == NormalizedUsage(
            input_tokens=100,
            output_tokens=42,
            total_tokens=142,
            reasoning_tokens=22,
            cached_tokens=15,
        )

    def test_maps_normalized_messages_when_native_input_is_absent(self):
        request = ModelRequest(
            messages=[
                Message(role="user", content="one"),
                Message(role="assistant", content="two"),
            ],
            request_config={"model": "gemini-3.8-flash", "store": False},
        )

        kwargs = GoogleGenAIInteractionsAdapter._build_call_kwargs(request)

        assert kwargs["input"] == [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "one"}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "two"}],
            },
        ]

    def test_runtime_builder_selects_interactions_adapter(self):
        adapter = build_model_runtime_adapter(
            client=_FakeClient(_raw_interaction()),
            runtime_config={
                "sdk": "google-genai",
                "api": "interactions",
                "state": "continuous_conversation",
            },
            config_id="google-test",
        )

        assert isinstance(adapter, GoogleGenAIInteractionsAdapter)

    def test_maps_only_recognized_context_limit_errors(self):
        overflow = google_genai_errors.ClientError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "Input token count exceeds the maximum number of tokens",
                }
            },
        )
        adapter = GoogleGenAIInteractionsAdapter(_FakeClient(overflow))

        with pytest.raises(ContextOverflowError, match="maximum number of tokens"):
            adapter.invoke(
                ModelRequest(
                    messages=[Message(role="user", content="large")],
                    request_config={"model": "gemini-3.8-flash", "store": False},
                )
            )

    def test_does_not_map_unrelated_invalid_argument(self):
        invalid = google_genai_errors.ClientError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "Request contains an invalid argument",
                }
            },
        )
        adapter = GoogleGenAIInteractionsAdapter(_FakeClient(invalid))

        with pytest.raises(google_genai_errors.ClientError):
            adapter.invoke(
                ModelRequest(
                    messages=[Message(role="user", content="invalid")],
                    request_config={"model": "gemini-3.8-flash", "store": False},
                )
            )


@pytest.mark.unit
class TestGoogleContinuousConversation:
    def test_first_and_later_turns_replay_every_step_exactly(self):
        low_level = _FakeModelAdapter([_response(1), _response(2)])
        adapter = GoogleContinuousConversationRuntimeAdapter(
            model_adapter=low_level,
            descriptor=ADAPTER_DESCRIPTORS["google.interactions.v1"],
        )

        first = adapter.invoke_turn(_turn(adapter, adapter.initial_state(), "one"))
        second = adapter.invoke_turn(_turn(adapter, first.state, "two"))

        assert low_level.requests[1].native_input == [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "one"}],
            },
            *_steps(1),
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "two"}],
            },
        ]
        assert second.state.payload["steps"][-2:] == _steps(2)
        assert [(turn.start_item, turn.end_item) for turn in second.state.accepted_turns] == [
            (0, 3),
            (3, 6),
        ]
        assert second.state.accepted_turns[-1].messages == [
            Message(role="user", content="two"),
            Message(role="assistant", content="ACTION1"),
        ]
        assert second.state.accepted_turns[-1].reasoning_summary == "summary 2"

        unwind = adapter.unwind_latest_accepted_turn(second.state)

        assert unwind is not None
        assert unwind.state.payload["steps"] == second.state.payload["steps"][:3]
        assert len(unwind.state.accepted_turns) == 1
        assert unwind.removed_items == 3
        assert second.state.payload["steps"][-2:] == _steps(2)

    def test_retry_isolation_reuses_last_accepted_state(self):
        low_level = _FakeModelAdapter([_response(1), _response(2)])
        adapter = GoogleContinuousConversationRuntimeAdapter(
            model_adapter=low_level,
            descriptor=ADAPTER_DESCRIPTORS["google.interactions.v1"],
        )
        request = _turn(adapter, adapter.initial_state())

        orphan = adapter.invoke_turn(request)
        accepted = adapter.invoke_turn(request)

        assert low_level.requests[0].native_input == low_level.requests[1].native_input
        assert orphan.state != accepted.state

    def test_recording_shapes_do_not_contain_signatures(self):
        low_level = _FakeModelAdapter([_response(1)])
        adapter = GoogleContinuousConversationRuntimeAdapter(
            model_adapter=low_level,
            descriptor=ADAPTER_DESCRIPTORS["google.interactions.v1"],
        )

        result = adapter.invoke_turn(_turn(adapter, adapter.initial_state()))
        persisted = json.dumps(
            {
                "request": result.sanitized_request,
                "transition": result.transition.model_dump(),
            }
        )

        assert "opaque-1" not in persisted
        assert "signature" not in persisted
        assert "opaque-1" in result.state.model_dump_json()

    @pytest.mark.parametrize(
        "request_update",
        [
            {"store": True},
            {"background": True},
            {"previous_interaction_id": "interaction"},
            {"generation_config": {}},
        ],
    )
    def test_rejects_incompatible_requests(self, request_update):
        low_level = _FakeModelAdapter([_response(1)])
        adapter = GoogleContinuousConversationRuntimeAdapter(
            model_adapter=low_level,
            descriptor=ADAPTER_DESCRIPTORS["google.interactions.v1"],
        )
        request = _turn(adapter, adapter.initial_state())
        request.request_config.update(request_update)

        with pytest.raises(ValueError, match="Google continuous conversation"):
            adapter.invoke_turn(request)

        assert low_level.requests == []

    def test_registry_builds_google_stateful_adapter(self):
        adapter = build_stateful_runtime_adapter(
            model_adapter=_FakeModelAdapter([]),
            runtime_config={
                "adapter_id": "google.interactions.v1",
                "sdk": "google-genai",
                "api": "interactions",
                "state": "continuous_conversation",
            },
            config_id="google-test",
        )

        assert isinstance(adapter, GoogleContinuousConversationRuntimeAdapter)


@pytest.mark.unit
def test_interaction_step_serialization_fails_closed_on_empty_output():
    response = ModelResponse(
        output_text="ACTION1",
        usage=NormalizedUsage(),
        raw_response={"steps": []},
    )

    with pytest.raises(RuntimeError, match="reusable steps"):
        serialize_interaction_steps(response)
