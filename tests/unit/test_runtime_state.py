import pytest
from pydantic import ValidationError

from benchmarking.runtime_models import Message, ModelResponse, NormalizedUsage
from benchmarking.runtime_registry import (
    ADAPTER_DESCRIPTORS,
    build_stateful_runtime_adapter,
    resolve_adapter_id,
)
from benchmarking.runtime_state import (
    ModelTurnRequest,
    RuntimeState,
    harness_commit_sha,
    sanitize_settings,
    source_permalink,
)


class _FakeModelAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _response(*, text: str = "ACTION1", response_id: str | None = None):
    return ModelResponse(
        output_text=text,
        usage=NormalizedUsage(total_tokens=10),
        response_id=response_id,
    )


def _turn(state: RuntimeState, **overrides):
    values = {
        "system_prompt": "system",
        "new_messages": [Message(role="user", content="frame")],
        "request_config": {"model": "model"},
        "previous_state": state,
    }
    values.update(overrides)
    return ModelTurnRequest(**values)


@pytest.mark.unit
class TestRuntimeStateContract:
    def test_json_round_trip(self):
        state = RuntimeState(
            adapter_id="openai.responses.v1",
            strategy="continuous_conversation",
            payload={"input_items": [{"type": "reasoning", "id": "rs_1"}]},
        )

        restored = RuntimeState.model_validate_json(state.model_dump_json())

        assert restored == state

    def test_rejects_unknown_schema_version(self):
        with pytest.raises(ValidationError, match="schema_version"):
            RuntimeState(
                schema_version=2,
                adapter_id="openai.responses.v1",
                strategy="continuous_conversation",
            )

    def test_rejects_malformed_non_json_payload(self):
        with pytest.raises(ValidationError, match="JSON-serializable"):
            RuntimeState(
                adapter_id="openai.responses.v1",
                strategy="continuous_conversation",
                payload={"bad": {object()}},
            )

    def test_rejects_adapter_mismatch(self):
        state = RuntimeState(
            adapter_id="openai.responses.v1",
            strategy="manual_rolling",
        )

        with pytest.raises(ValueError, match="adapter mismatch"):
            state.validate_for(
                adapter_id="anthropic.messages.v1", strategy="manual_rolling"
            )

    def test_strategy_rejects_malformed_provider_payload(self):
        low_level = _FakeModelAdapter([_response()])
        adapter = build_stateful_runtime_adapter(
            model_adapter=low_level,
            runtime_config={
                "sdk": "openai-python",
                "api": "chat_completions",
                "state": "manual_rolling",
            },
            config_id="manual",
        )
        malformed = RuntimeState(
            adapter_id=adapter.descriptor.adapter_id,
            strategy="manual_rolling",
            payload={"messages": "not-a-list"},
        )

        with pytest.raises(ValueError, match="payload.messages"):
            adapter.invoke_turn(_turn(malformed))


@pytest.mark.unit
class TestCommonStateStrategies:
    def test_manual_rolling_preserves_existing_request_shape(self):
        low_level = _FakeModelAdapter([_response()])
        adapter = build_stateful_runtime_adapter(
            model_adapter=low_level,
            runtime_config={
                "sdk": "openai-python",
                "api": "chat_completions",
                "state": "manual_rolling",
            },
            config_id="legacy-chat",
        )
        state = adapter.buffer_inputs(
            adapter.initial_state(),
            [Message(role="user", content="earlier frame")],
        )

        result = adapter.invoke_turn(_turn(state))

        assert [message.model_dump() for message in low_level.requests[0].messages] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "earlier frame"},
            {"role": "user", "content": "frame"},
        ]
        assert result.state.payload["messages"][-1] == {
            "role": "assistant",
            "content": "ACTION1",
        }

    def test_manual_rolling_uses_existing_oldest_turn_trimming(self):
        low_level = _FakeModelAdapter([_response()])
        adapter = build_stateful_runtime_adapter(
            model_adapter=low_level,
            runtime_config={
                "sdk": "openai-python",
                "api": "chat_completions",
                "state": "manual_rolling",
            },
            config_id="legacy-chat",
        )
        state = RuntimeState(
            adapter_id=adapter.descriptor.adapter_id,
            strategy="manual_rolling",
            payload={
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "ACTION1"},
                ]
            },
        )

        adapter.invoke_turn(_turn(state, max_context_length=5))

        assert [message.content for message in low_level.requests[0].messages] == [
            "system",
            "frame",
        ]

    def test_previous_response_id_preserves_first_and_later_turn_shapes(self):
        low_level = _FakeModelAdapter(
            [_response(response_id="resp_1"), _response(response_id="resp_2")]
        )
        adapter = build_stateful_runtime_adapter(
            model_adapter=low_level,
            runtime_config={
                "sdk": "openai-python",
                "api": "responses",
                "state": "previous_response_id",
            },
            config_id="server-state",
        )

        first = adapter.invoke_turn(_turn(adapter.initial_state()))
        second = adapter.invoke_turn(_turn(first.state))

        assert [m.role for m in low_level.requests[0].messages] == ["system", "user"]
        assert [m.role for m in low_level.requests[1].messages] == ["user"]
        assert low_level.requests[1].request_config["previous_response_id"] == "resp_1"
        assert second.state.payload == {
            "response_id": "resp_2",
            "pending_inputs": [],
        }


@pytest.mark.unit
class TestAdapterRegistryAndProvenance:
    def test_legacy_sdk_and_api_resolve_to_stable_adapter(self):
        assert resolve_adapter_id(
            {"sdk": "openai-python", "api": "responses"}, "legacy"
        ) == "openai.responses.v1"

    def test_explicit_adapter_must_match_legacy_fields(self):
        with pytest.raises(ValueError, match="does not match"):
            resolve_adapter_id(
                {
                    "adapter_id": "anthropic.messages.v1",
                    "sdk": "openai-python",
                    "api": "responses",
                },
                "bad",
            )

    def test_openai_descriptor_is_provider_reference(self):
        descriptor = ADAPTER_DESCRIPTORS["openai.responses.v1"]
        assert descriptor.approval_status == "provider_reference"
        assert descriptor.implementation_path == "benchmarking/openai_runtime.py"

    def test_commit_sha_is_null_when_environment_is_absent(self, monkeypatch):
        monkeypatch.delenv("ARC_HARNESS_COMMIT_SHA", raising=False)
        assert harness_commit_sha() is None
        assert source_permalink(
            repository="https://github.com/arcprize/repo",
            implementation_path="benchmarking/openai_runtime.py",
            commit_sha=None,
        ) is None

    def test_commit_sha_builds_immutable_permalink(self, monkeypatch):
        monkeypatch.setenv("ARC_HARNESS_COMMIT_SHA", "abc123")
        assert harness_commit_sha() == "abc123"
        assert source_permalink(
            repository="https://github.com/arcprize/repo",
            implementation_path="benchmarking/openai_runtime.py",
            commit_sha=harness_commit_sha(),
        ) == (
            "https://github.com/arcprize/repo/blob/abc123/"
            "benchmarking/openai_runtime.py"
        )

    def test_settings_sanitization_keeps_token_limits_and_removes_secrets(self):
        assert sanitize_settings(
            {
                "max_output_tokens": 128_000,
                "api_key": "secret",
                "nested": {"encrypted_content": "opaque"},
            }
        ) == {
            "max_output_tokens": 128_000,
            "api_key": "[redacted]",
            "nested": {},
        }
