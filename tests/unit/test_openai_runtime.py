import json
from types import SimpleNamespace

import pytest

from benchmarking.openai_runtime import (
    OpenAIEncryptedReplayRuntimeAdapter,
    prune_after_latest_compaction,
    serialize_replay_output,
)
from benchmarking.runtime_models import Message, ModelResponse, NormalizedUsage
from benchmarking.runtime_registry import ADAPTER_DESCRIPTORS
from benchmarking.runtime_state import ModelTurnRequest


class _FakeModelAdapter:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _output(turn: int, *, compaction: bool = False):
    items = []
    if compaction:
        items.append(
            {
                "type": "compaction",
                "id": f"cmp_{turn}",
                "encrypted_content": f"compacted-{turn}",
            }
        )
    items.extend(
        [
            {
                "type": "reasoning",
                "id": f"rs_{turn}",
                "encrypted_content": f"encrypted-{turn}",
                "summary": [{"type": "summary_text", "text": "summary"}],
            },
            {
                "type": "message",
                "id": f"msg_{turn}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ACTION1"}],
            },
        ]
    )
    return items


def _response(turn: int, *, output=None):
    return ModelResponse(
        output_text="ACTION1",
        reasoning_text="summary",
        usage=NormalizedUsage(total_tokens=10),
        raw_response={"output": output if output is not None else _output(turn)},
    )


def _adapter(responses):
    low_level = _FakeModelAdapter(responses)
    adapter = OpenAIEncryptedReplayRuntimeAdapter(
        model_adapter=low_level,
        descriptor=ADAPTER_DESCRIPTORS["openai.responses.v1"],
    )
    return adapter, low_level


def _turn(adapter, state, content="frame"):
    return ModelTurnRequest(
        system_prompt="system",
        new_messages=[Message(role="user", content=content)],
        request_config={
            "model": "gpt-5.6-sol",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"context": "all_turns", "summary": "auto"},
        },
        previous_state=state,
    )


@pytest.mark.unit
class TestOpenAIEncryptedReplay:
    def test_first_and_later_turns_preserve_all_native_output_items(self):
        adapter, low_level = _adapter([_response(1), _response(2)])

        first = adapter.invoke_turn(_turn(adapter, adapter.initial_state(), "frame 1"))
        second = adapter.invoke_turn(_turn(adapter, first.state, "frame 2"))

        assert low_level.requests[0].native_input == [
            {"role": "user", "content": "frame 1"}
        ]
        assert low_level.requests[1].native_input == [
            {"role": "user", "content": "frame 1"},
            *_output(1),
            {"role": "user", "content": "frame 2"},
        ]
        assert second.state.payload["input_items"][-2:] == _output(2)

    def test_retry_isolation_reuses_last_accepted_state(self):
        adapter, low_level = _adapter([_response(1), _response(2)])
        state = adapter.initial_state()
        request = _turn(adapter, state)

        orphan = adapter.invoke_turn(request)
        accepted = adapter.invoke_turn(request)

        assert low_level.requests[0].native_input == low_level.requests[1].native_input
        assert orphan.state != accepted.state
        assert accepted.state.payload["input_items"][-2:] == _output(2)

    def test_buffered_forced_reset_input_precedes_next_frame(self):
        adapter, low_level = _adapter([_response(1)])
        state = adapter.buffer_inputs(
            adapter.initial_state(),
            [Message(role="user", content="game over")],
        )

        adapter.invoke_turn(_turn(adapter, state, "next frame"))

        assert low_level.requests[0].native_input == [
            {"role": "user", "content": "game over"},
            {"role": "user", "content": "next frame"},
        ]

    def test_compaction_keeps_latest_compaction_and_later_items(self):
        output = [
            {"type": "compaction", "id": "old", "encrypted_content": "old"},
            {"type": "message", "id": "between"},
            {"type": "compaction", "id": "latest", "encrypted_content": "new"},
            {"type": "message", "id": "after"},
        ]
        adapter, _ = _adapter([_response(1, output=output)])

        result = adapter.invoke_turn(_turn(adapter, adapter.initial_state()))

        assert result.state.payload["input_items"] == output[2:]
        assert result.transition.compaction_items_returned == 2
        assert result.transition.history_items_before_prune == 5
        assert result.transition.history_items_after_prune == 2

    def test_recording_surfaces_contain_no_encrypted_payload(self):
        adapter, _ = _adapter([_response(1)])

        result = adapter.invoke_turn(_turn(adapter, adapter.initial_state()))
        persisted_shape = json.dumps(
            {
                "request": result.sanitized_request,
                "transition": result.transition.model_dump(),
            }
        )

        assert "encrypted-1" not in persisted_shape
        assert '"encrypted_content":' not in persisted_shape
        assert result.transition.sanitized_items == [
            {"type": "message", "role": "user"}
        ]

    def test_malformed_empty_output_fails_closed(self):
        adapter, _ = _adapter([_response(1, output=[])])

        with pytest.raises(RuntimeError, match="replayable output items"):
            adapter.invoke_turn(_turn(adapter, adapter.initial_state()))

    @pytest.mark.parametrize(
        "request_update",
        [
            {"store": True},
            {"background": True},
            {"previous_response_id": "resp"},
            {"conversation": "conv"},
            {"include": []},
            {"reasoning": {"context": "auto", "summary": "auto"}},
            {"reasoning": {"context": "all_turns", "summary": "detailed"}},
        ],
    )
    def test_rejects_invalid_request_at_provider_boundary(self, request_update):
        adapter, low_level = _adapter([_response(1)])
        request = _turn(adapter, adapter.initial_state())
        request.request_config.update(request_update)

        with pytest.raises(ValueError, match="OpenAI encrypted replay"):
            adapter.invoke_turn(request)

        assert low_level.requests == []


@pytest.mark.unit
class TestOpenAIReplaySerialization:
    def test_sdk_response_only_fields_are_removed_without_dropping_nulls(self):
        class _SDKItem:
            def __init__(self, value):
                self.value = value

            def model_dump(self, *, mode):
                assert mode == "json"
                return dict(self.value)

        response = ModelResponse(
            output_text="ACTION1",
            usage=NormalizedUsage(),
            raw_response=SimpleNamespace(
                output=[
                    _SDKItem(
                        {
                            "type": "reasoning",
                            "id": "rs",
                            "encrypted_content": "opaque",
                            "summary": None,
                            "status": "completed",
                        }
                    ),
                    _SDKItem(
                        {
                            "type": "compaction",
                            "id": "cmp",
                            "encrypted_content": "opaque-compact",
                            "created_by": "server",
                        }
                    ),
                ]
            ),
        )

        assert serialize_replay_output(response) == [
            {
                "type": "reasoning",
                "id": "rs",
                "encrypted_content": "opaque",
                "summary": None,
            },
            {
                "type": "compaction",
                "id": "cmp",
                "encrypted_content": "opaque-compact",
            },
        ]

    def test_pruning_without_compaction_preserves_complete_history(self):
        items = [{"type": "message", "id": "one"}]
        assert prune_after_latest_compaction(items) == items
