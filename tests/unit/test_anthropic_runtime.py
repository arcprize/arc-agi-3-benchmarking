import json
from copy import deepcopy
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from benchmarking.anthropic_runtime import (
    AnthropicContinuousConversationRuntimeAdapter,
    normalize_native_response,
    normalize_native_usage,
    safe_provider_error_metadata,
    serialize_replay_content,
    validate_continuous_conversation_request,
)
from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.exceptions import EmptyResponseError, InvalidProviderResponseError
from benchmarking.model_config import _validate_model_config_entry, get_model_config
from benchmarking.runtime_adapters import (
    AnthropicMessagesAdapter,
    build_model_runtime_adapter,
)
from benchmarking.runtime_models import Message, ModelRequest, NormalizedUsage
from benchmarking.runtime_registry import (
    ADAPTER_DESCRIPTORS,
    build_stateful_runtime_adapter,
)
from benchmarking.runtime_state import ModelTurnRequest, RuntimeState, sanitize_settings

CONFIG_ID = "anthropic-opus-5-low-provider-adapter"


def _request_config():
    return deepcopy(get_model_config(CONFIG_ID)["request"])


def _raw_response(text="ACTION1", *, blocks=None, stop_reason="end_turn", usage=None):
    return {
        "model": "claude-opus-5",
        "role": "assistant",
        "content": blocks
        if blocks is not None
        else [
            {
                "type": "thinking",
                "thinking": "Visible summary",
                "signature": "secret-signature",
            },
            {
                "type": "thinking",
                "thinking": "",
                "signature": "empty-thinking-signature",
            },
            {"type": "redacted_thinking", "data": "secret-ciphertext"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }


def _compaction_response(summary="summary", *, signature="summary-signature"):
    return _raw_response(
        blocks=[{"type": "compaction", "content": summary, "signature": signature}],
        stop_reason="compaction",
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": [
                {"type": "compaction", "input_tokens": 100, "output_tokens": 20}
            ],
        },
    )


class _FakeModelAdapter:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, request):
        self.requests.append(deepcopy(request))
        raw = next(self.responses)
        if isinstance(raw, Exception):
            raise raw
        return normalize_native_response(raw, request.request_config)


def _adapter(responses, *, trigger_tokens=175_000):
    low_level = _FakeModelAdapter(responses)
    runtime = deepcopy(get_model_config(CONFIG_ID)["runtime"])
    runtime["compaction"]["trigger_tokens"] = trigger_tokens
    adapter = build_stateful_runtime_adapter(
        model_adapter=low_level,
        runtime_config=runtime,
        config_id=CONFIG_ID,
    )
    return adapter, low_level


def _turn(adapter, state, text="frame"):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="system",
            new_messages=[Message(role="user", content=text)],
            request_config=_request_config(),
            previous_state=state,
            max_context_length=1_000_000,
        )
    )


@pytest.mark.unit
class TestAnthropicRuntime:
    @pytest.mark.parametrize("sdk_blocks", [False, True])
    def test_replay_removes_only_response_only_sdk_fields(self, sdk_blocks):
        blocks = [
            {"type": "thinking", "thinking": "", "signature": "opaque-thinking"},
            {"type": "redacted_thinking", "data": "opaque-redacted"},
            {"type": "text", "text": "ACTION1", "parsed_output": {"action": 1}},
            {"type": "compaction", "content": "summary", "encrypted_content": None},
            {
                "type": "compaction",
                "content": "later summary",
                "encrypted_content": "opaque-compaction",
                "signature": "opaque-compaction-signature",
                "future_field": {"version": 1},
            },
        ]
        snapshot = deepcopy(blocks)
        if sdk_blocks:
            message = anthropic.types.beta.BetaMessage.model_validate(
                {"id": "msg_test", "type": "message", **_raw_response(blocks=blocks)}
            )
            serialized = serialize_replay_content(message.content)
        else:
            serialized = serialize_replay_content(blocks)
        expected = deepcopy(snapshot)
        expected[2].pop("parsed_output")
        expected[3].pop("encrypted_content")
        assert serialized == expected
        assert blocks == snapshot

    @pytest.mark.parametrize(
        "details",
        [
            None,
            {},
            {"thinking_tokens": None},
            {"thinking_tokens": 0},
            {"thinking_tokens": 3},
        ],
    )
    def test_optional_thinking_breakdown_does_not_increase_output_or_cost(
        self, details
    ):
        usage = normalize_native_usage(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": details,
            }
        )
        assert usage.reasoning_tokens == ((details or {}).get("thinking_tokens") or 0)
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15
        assert usage.cost == 0

    @pytest.mark.parametrize("top_thinking,expected", [(None, 5), (0, 2), (3, 5)])
    def test_iteration_thinking_is_not_added_to_top_level_breakdown_twice(
        self, top_thinking, expected
    ):
        usage = normalize_native_usage(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": {"thinking_tokens": top_thinking},
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "output_tokens_details": {"thinking_tokens": 2},
                    },
                    {
                        "type": "message",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "output_tokens_details": {"thinking_tokens": 3},
                    },
                ],
            }
        )
        assert usage.reasoning_tokens == expected
        assert usage.input_tokens == 110
        assert usage.output_tokens == 15
        assert usage.total_tokens == 125

    def test_exact_replay_and_readable_projection(self):
        raw = _raw_response()
        raw["content"].append({"type": "future_block", "opaque": {"version": 1}})
        adapter, low_level = _adapter([raw, _raw_response()])
        initial = adapter.initial_state()
        first = _turn(adapter, initial)
        snapshot = first.state.model_dump_json()
        second = _turn(adapter, first.state, "next")
        assert initial.payload == {"messages": []}
        assert first.state.model_dump_json() == snapshot
        assert low_level.requests[1].native_input == [
            {"role": "user", "content": "frame"},
            {"role": "assistant", "content": raw["content"]},
            {"role": "user", "content": "next"},
        ]
        assert second.response.reasoning_text == "Visible summary"
        readable = str(second.readable_request_messages)
        assert "messages" not in second.sanitized_request
        assert "Visible summary" in readable
        assert "secret-signature" not in readable
        assert "secret-ciphertext" not in readable
        assert "empty-thinking-signature" not in readable
        second.state.payload["messages"][1]["content"][0]["thinking"] = "changed"
        assert first.state.model_dump_json() == snapshot

    def test_multiple_compactions_and_buffered_inputs(self):
        adapter, low_level = _adapter(
            [
                _raw_response(),
                _compaction_response("summary one"),
                _raw_response(),
                _compaction_response("summary two"),
                _raw_response(),
            ],
            trigger_tokens=15,
        )
        first = _turn(adapter, adapter.initial_state(), "old frame")
        buffered = adapter.buffer_inputs(
            first.state, [Message(role="user", content="GAME_OVER")]
        )
        second = _turn(adapter, buffered, "reset frame")
        assert low_level.requests[1].native_input == first.state.payload["messages"]
        assert low_level.requests[1].request_config["compaction"] == {
            "type": "summarize"
        }
        assert low_level.requests[1].request_config["cache_control"] == {
            "type": "ephemeral"
        }
        assert [
            message["content"] for message in low_level.requests[2].native_input[-2:]
        ] == ["GAME_OVER", "reset frame"]
        assert "compaction" not in low_level.requests[2].request_config
        assert "context_management" not in low_level.requests[2].request_config
        assert low_level.requests[2].request_config["cache_control"] == {
            "type": "ephemeral"
        }
        assert second.transition.history_items_before_prune == 5
        assert second.transition.history_items_after_prune == 4
        assert second.response.usage.total_tokens == 135
        assert second.state.payload["context_tokens"] == 15
        assert "reset frame" in str(second.readable_request_messages)
        third = _turn(adapter, second.state, "post-compaction")
        readable = str(third.readable_request_messages)
        assert "summary two" in readable
        assert "summary one" not in readable
        assert "old frame" not in readable
        assert "post-compaction" not in str(low_level.requests[3].native_input)
        assert "reset frame" in str(low_level.requests[3].native_input)
        assert (
            third.state.payload["messages"][0]["content"]
            == _compaction_response("summary two")["content"]
        )

    @pytest.mark.parametrize("summary", [None, "", " "])
    def test_null_compaction_preserves_history(self, summary):
        adapter, low_level = _adapter(
            [_raw_response(), _compaction_response(summary)], trigger_tokens=15
        )
        state = _turn(adapter, adapter.initial_state()).state
        snapshot = state.model_dump_json()
        with pytest.raises(InvalidProviderResponseError) as raised:
            _turn(adapter, state, "newest")
        assert raised.value.usage.total_tokens == 120
        assert state.model_dump_json() == snapshot
        assert "newest" not in str(low_level.requests[-1].native_input)

    def test_inline_compaction_is_rejected_even_with_valid_action(self):
        raw = _raw_response()
        raw["content"].insert(0, {"type": "compaction", "content": None})
        adapter, _ = _adapter([raw])
        state = adapter.initial_state()
        with pytest.raises(InvalidProviderResponseError):
            _turn(adapter, state)
        assert state.payload == {"messages": []}

    @pytest.mark.parametrize(
        "field,value", [("model", "different-model"), ("system_prompt", "changed")]
    )
    def test_rejects_session_identity_changes(self, field, value):
        adapter, _ = _adapter([_raw_response()])
        state = _turn(adapter, adapter.initial_state()).state
        state.payload[field] = value
        with pytest.raises(ValueError, match=f"cannot change {field}"):
            _turn(adapter, state)

    def test_rejects_mismatched_state(self):
        adapter, _ = _adapter([])
        state = RuntimeState(
            adapter_id="openai.responses.v1", strategy="continuous_conversation"
        )
        with pytest.raises(ValueError, match="adapter mismatch"):
            _turn(adapter, state)

    def test_harness_compaction_is_rejected_before_any_provider_call(self):
        adapter, low_level = _adapter([])
        state = adapter.initial_state()
        snapshot = state.model_dump_json()
        compactor = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary", trigger_tokens=175_000)
        )
        with pytest.raises(ValueError, match="does not support harness summary"):
            compactor.compact(
                adapter=adapter,
                state=state,
                request_config=_request_config(),
                trigger_tokens=175_000,
                max_context_length=1_000_000,
                max_retries=2,
            )
        assert low_level.requests == []
        assert state.model_dump_json() == snapshot

    def test_provider_error_does_not_expose_native_state(self):
        adapter, _ = _adapter([RuntimeError("secret-signature")])
        with pytest.raises(InvalidProviderResponseError) as raised:
            _turn(adapter, adapter.initial_state())
        assert "secret-signature" not in str(raised.value)

    @pytest.mark.parametrize(
        "stop_reason",
        [
            "refusal",
            "max_tokens",
            "pause_turn",
            "tool_use",
            "compaction",
            "model_context_window_exceeded",
            None,
        ],
    )
    def test_incomplete_response_cannot_supply_an_action(self, stop_reason):
        raw = _raw_response(stop_reason=stop_reason)
        with pytest.raises(InvalidProviderResponseError) as raised:
            normalize_native_response(raw, _request_config())
        assert raised.value.usage.total_tokens == 15
        assert "secret-signature" not in str(raised.value.response)

    def test_stop_sequence_must_be_configured(self):
        raw = _raw_response(stop_reason="stop_sequence")
        raw["stop_sequence"] = "END"
        with pytest.raises(InvalidProviderResponseError):
            normalize_native_response(raw, _request_config())
        config = {**_request_config(), "stop_sequences": ["END"]}
        assert normalize_native_response(raw, config).output_text == "ACTION1"

    @pytest.mark.parametrize(
        "blocks",
        [
            [],
            [{"type": "thinking", "thinking": "summary", "signature": "secret"}],
            [{"type": "text", "text": " "}],
        ],
    )
    def test_empty_or_thought_only_responses_retain_usage(self, blocks):
        with pytest.raises(EmptyResponseError) as raised:
            normalize_native_response(_raw_response(blocks=blocks), _request_config())
        assert raised.value.usage.total_tokens == 15
        assert "secret" not in str(raised.value.response)

    def test_usage_iterations_include_compaction_and_cache_without_double_counting(
        self,
    ):
        usage = normalize_native_usage(
            {
                "input_tokens": 99999,
                "output_tokens": 99999,
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 30,
                    },
                    {
                        "type": "message",
                        "input_tokens": 40,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 10,
                    },
                ],
            }
        )
        assert usage == NormalizedUsage(
            input_tokens=200,
            output_tokens=15,
            total_tokens=215,
            cached_tokens=30,
            cache_write_tokens=30,
        )
        assert usage.reasoning_tokens == 0
        assert usage.cost == 0
        assert (
            normalize_native_usage(
                SimpleNamespace(input_tokens=4, output_tokens=3)
            ).total_tokens
            == 7
        )

    def test_nested_sanitization_preserves_readable_and_nonopaque_data(self):
        raw = {
            "response": _raw_response(),
            "nested": [{"signature": "hidden", "data": {"x": 1}}],
        }
        safe = sanitize_settings(raw)
        assert "secret-" not in str(safe)
        assert "empty-thinking-signature" not in str(safe)
        assert "Visible summary" in str(safe)
        assert safe["nested"] == [{"data": {"x": 1}}]
        assert raw["response"]["content"][0]["signature"] == "secret-signature"


@pytest.mark.unit
class TestAnthropicConfiguration:
    @pytest.mark.parametrize(
        "policy",
        [
            {},
            {"strategy": "harness_summary", "trigger_tokens": 175_000},
            {"strategy": "native", "trigger_tokens": 0},
            {"strategy": "native", "trigger_tokens": True},
            {"strategy": "native", "trigger_tokens": 1_000_000},
            {
                "strategy": "native",
                "trigger_tokens": 175_000,
                "summary_max_output_tokens": 0,
            },
            {
                "strategy": "native",
                "trigger_tokens": 175_000,
                "pause_after_compaction": True,
            },
        ],
    )
    def test_invalid_native_policy(self, policy):
        config = deepcopy(get_model_config(CONFIG_ID))
        config["runtime"]["compaction"] = policy
        with pytest.raises(ValueError):
            _validate_model_config_entry(config, 1, set())

    def test_native_policy_requires_beta(self):
        config = deepcopy(get_model_config(CONFIG_ID))
        config["request"]["betas"] = []
        with pytest.raises(ValueError, match="compact-2026-09-04"):
            _validate_model_config_entry(config, 1, set())

    @pytest.mark.parametrize(
        "update",
        [
            {"compaction": {"type": "summarize"}},
            {"betas": ["compact-2026-01-12"]},
            {"output_config": {"task_budget": {"remaining": 10}}},
        ],
    )
    def test_rejects_unsafe_action_request_settings(self, update):
        config = _request_config()
        config.update(update)
        with pytest.raises(ValueError):
            validate_continuous_conversation_request(config)

    @pytest.mark.parametrize(
        "cache_control",
        [
            "ephemeral",
            {},
            {"type": "persistent"},
            {"type": "ephemeral", "ttl": "2h"},
            {"type": "ephemeral", "scope": "conversation"},
        ],
    )
    def test_rejects_invalid_cache_control(self, cache_control):
        config = _request_config()
        config["cache_control"] = cache_control
        with pytest.raises(ValueError, match="cache_control"):
            validate_continuous_conversation_request(config)

    @pytest.mark.parametrize("ttl", [None, "5m", "1h"])
    def test_accepts_supported_cache_control(self, ttl):
        config = _request_config()
        config["cache_control"] = {"type": "ephemeral"}
        if ttl is not None:
            config["cache_control"]["ttl"] = ttl
        validate_continuous_conversation_request(config)

    @pytest.mark.parametrize(
        "model", ["claude-opus-5", "claude-fable-5", "claude-fable-5-1"]
    )
    def test_native_policy_does_not_hardcode_the_model(self, model):
        config = deepcopy(get_model_config(CONFIG_ID))
        config["request"]["model"] = model
        _validate_model_config_entry(config, 1, set())

    def test_thinking_telemetry_beta_is_optional(self):
        config = deepcopy(get_model_config(CONFIG_ID))
        config["request"]["betas"].remove("thinking-token-count-2026-05-13")
        _validate_model_config_entry(config, 1, set())

    def test_checked_in_profile_and_registry(self):
        config = get_model_config(CONFIG_ID)
        assert config["request"]["model"] == "claude-opus-5"
        assert config["request"]["output_config"] == {"effort": "low"}
        assert config["request"]["max_tokens"] == 128_000
        assert config["request"]["cache_control"] == {"type": "ephemeral"}
        assert "thinking-token-count-2026-05-13" in config["request"]["betas"]
        assert config["agent"]["MAX_CONTEXT_LENGTH"] == 1_000_000
        assert config["pricing"] == {"input": 5, "output": 25}
        assert "store" not in config["request"]
        assert isinstance(
            _adapter([])[0], AnthropicContinuousConversationRuntimeAdapter
        )
        assert (
            ADAPTER_DESCRIPTORS["anthropic.messages.v1"].approval_status == "unreviewed"
        )

    @pytest.mark.parametrize(
        "update",
        [
            {"store": False},
            {"previous_response_id": "response"},
            {"include": []},
            {"reasoning": {}},
            {"fallbacks": "default"},
            {"tools": []},
            {"max_tokens": True},
            {"betas": "compact-2026-01-12"},
            {"thinking": {"type": "adaptive"}},
            {"thinking": {"type": "disabled", "display": "summarized"}},
            {"context_management": {}},
        ],
    )
    def test_request_validation_cannot_be_bypassed_by_legacy_adapter_selection(
        self, update
    ):
        config = deepcopy(get_model_config(CONFIG_ID))
        config["runtime"].pop("adapter_id")
        config["request"].update(update)
        with pytest.raises(ValueError):
            _validate_model_config_entry(config, 1, set())
        with pytest.raises(ValueError):
            validate_continuous_conversation_request(config["request"])

    @pytest.mark.parametrize(
        "edit",
        [
            {"type": "clear_thinking_20251015"},
            {"type": "compact_20260112", "pause_after_compaction": True},
            {"type": "compact_20260112", "pause_after_compaction": 0},
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 49999},
            },
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": True},
            },
            {
                "type": "compact_20260112",
                "trigger": {"type": "output_tokens", "value": 50000},
            },
        ],
    )
    def test_invalid_compaction_edit(self, edit):
        request = _request_config()
        request["context_management"] = {"edits": [edit]}
        with pytest.raises(ValueError):
            validate_continuous_conversation_request(request)

    @pytest.mark.parametrize("explicit_adapter", [False, True])
    def test_compaction_is_optional_but_harness_compaction_is_rejected(
        self, explicit_adapter
    ):
        config = deepcopy(get_model_config(CONFIG_ID))
        if not explicit_adapter:
            config["runtime"].pop("adapter_id")
        config["runtime"].pop("compaction")
        config["request"].pop("betas")
        _validate_model_config_entry(config, 1, set())
        config["runtime"]["compaction"] = {
            "strategy": "harness_summary",
            "trigger_tokens": 175_000,
        }
        with pytest.raises(ValueError, match="native"):
            _validate_model_config_entry(config, 1, set())


def _stream_events(*, stop_reason="end_turn", complete=True, sdk_extras=False):
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    events = [{"type": "message_start", "message": message}]
    blocks_and_deltas = [
        (
            {"type": "thinking", "thinking": "", "signature": ""},
            [
                {"type": "thinking_delta", "thinking": "SDK thinking"},
                {"type": "signature_delta", "signature": "sdk-signature"},
            ],
        ),
        (
            {"type": "thinking", "thinking": "", "signature": ""},
            [
                {
                    "type": "signature_delta",
                    "signature": "sdk-empty-thinking-signature",
                },
            ],
        ),
        ({"type": "redacted_thinking", "data": "sdk-ciphertext"}, []),
        ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "ACTION1"}]),
    ]
    for index, (block, deltas) in enumerate(blocks_and_deltas):
        if sdk_extras and block["type"] == "text":
            block["parsed_output"] = None
        events.append(
            {"type": "content_block_start", "index": index, "content_block": block}
        )
        events.extend(
            {"type": "content_block_delta", "index": index, "delta": delta}
            for delta in deltas
        )
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": {"category": "synthetic-category"}
                if stop_reason == "refusal"
                else None,
            },
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": {"thinking_tokens": 3},
                "iterations": [
                    {"type": "message", "input_tokens": 10, "output_tokens": 5},
                ],
            },
        }
    )
    if complete:
        events.append({"type": "message_stop"})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _summary_stream_events(raw, *, complete=True):
    message = {
        "id": "msg_summary",
        "type": "message",
        **raw,
        "content": [],
        "stop_reason": None,
    }
    events = [{"type": "message_start", "message": message}, {"type": "ping"}]
    for index, block in enumerate(raw["content"]):
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": block},
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": raw["stop_reason"],
                "stop_sequence": None,
                "stop_details": raw.get("stop_details"),
            },
            "usage": raw["usage"],
        }
    )
    if complete:
        events.append({"type": "message_stop"})
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


class _InterruptedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield _stream_events(complete=False)
        raise httpx.ReadError("failure mentioning sdk-signature")


def _sdk_adapter(client):
    config = get_model_config(CONFIG_ID)
    return build_stateful_runtime_adapter(
        model_adapter=build_model_runtime_adapter(
            client=client, runtime_config=config["runtime"], config_id=CONFIG_ID
        ),
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )


@pytest.mark.unit
class TestAnthropicOnDemandCompaction:
    @pytest.mark.parametrize(
        "content", [[None], ["invalid block"], {"type": "compaction"}]
    )
    def test_malformed_summary_keeps_returned_usage(self, content):
        summary = _compaction_response()
        summary["content"] = content
        adapter, _ = _adapter([_raw_response(), summary], trigger_tokens=15)
        state = _turn(adapter, adapter.initial_state()).state
        snapshot = state.model_dump_json()
        with pytest.raises(InvalidProviderResponseError) as raised:
            _turn(adapter, state)
        assert raised.value.usage.total_tokens == 120
        assert state.model_dump_json() == snapshot

    def test_compaction_usage_does_not_schedule_another_compaction(self):
        summary = _compaction_response()
        summary["usage"]["iterations"][0]["input_tokens"] = 200_000
        adapter, low_level = _adapter(
            [
                _raw_response(usage={"input_tokens": 175_000, "output_tokens": 5}),
                summary,
                _raw_response(),
                _raw_response(),
            ]
        )
        first = _turn(adapter, adapter.initial_state())
        second = _turn(adapter, first.state, "fresh")
        assert second.response.usage.total_tokens > 175_000
        assert second.state.payload["context_tokens"] == 15
        third = _turn(adapter, second.state, "another fresh frame")
        assert third.transition.compaction_items_returned == 0
        assert len(low_level.requests) == 4

    @pytest.mark.parametrize(
        "stop_reason",
        [
            "max_tokens",
            "model_context_window_exceeded",
            "refusal",
            "tool_use",
            "end_turn",
            "pause_turn",
        ],
    )
    def test_failed_summary_cannot_be_accepted_as_action(self, stop_reason):
        raw = _compaction_response()
        raw["stop_reason"] = stop_reason
        raw["content"].append({"type": "text", "text": "ACTION1"})
        adapter, low_level = _adapter([_raw_response(), raw], trigger_tokens=15)
        state = _turn(adapter, adapter.initial_state()).state
        snapshot = state.model_dump_json()
        with pytest.raises(InvalidProviderResponseError) as raised:
            _turn(adapter, state, "untouched next frame")
        assert state.model_dump_json() == snapshot
        assert raised.value.usage.total_tokens == 120
        assert len(low_level.requests) == 2
        assert "untouched next frame" not in str(low_level.requests[-1].native_input)
        assert "summary-signature" not in str(raised.value.response)

    @pytest.mark.parametrize("signature", [None, "", " "])
    def test_unsigned_summary_does_not_replace_history(self, signature):
        adapter, _ = _adapter(
            [_raw_response(), _compaction_response(signature=signature)],
            trigger_tokens=15,
        )
        state = _turn(adapter, adapter.initial_state()).state
        snapshot = state.model_dump_json()
        with pytest.raises(InvalidProviderResponseError):
            _turn(adapter, state)
        assert state.model_dump_json() == snapshot

    @pytest.mark.parametrize(
        "failure",
        [
            _raw_response(stop_reason="refusal"),
            _raw_response(stop_reason="max_tokens"),
            _raw_response(
                blocks=[
                    {"type": "thinking", "thinking": "only", "signature": "private"}
                ]
            ),
            RuntimeError("private-summary-signature"),
        ],
    )
    def test_failed_action_discards_provisional_summary_but_keeps_usage(self, failure):
        adapter, low_level = _adapter(
            [_raw_response(), _compaction_response(), failure], trigger_tokens=15
        )
        state = _turn(adapter, adapter.initial_state()).state
        snapshot = state.model_dump_json()
        with pytest.raises(InvalidProviderResponseError) as raised:
            _turn(adapter, state, "latest untouched frame")
        assert state.model_dump_json() == snapshot
        assert raised.value.usage.total_tokens == (
            120 if isinstance(failure, Exception) else 135
        )
        assert "private-summary-signature" not in str(raised.value)
        assert "latest untouched frame" not in str(low_level.requests[1].native_input)
        assert (
            low_level.requests[2].native_input[-1]["content"]
            == "latest untouched frame"
        )

    def test_first_frame_is_never_compacted_even_if_large(self):
        adapter, low_level = _adapter([_raw_response()], trigger_tokens=1)
        _turn(adapter, adapter.initial_state(), "large frame " * 1000)
        assert len(low_level.requests) == 1
        assert "compaction" not in low_level.requests[0].request_config

    def test_summary_strips_action_only_constraints_without_mutating_config(self):
        adapter, low_level = _adapter(
            [_raw_response(), _compaction_response(), _raw_response()],
            trigger_tokens=15,
        )
        state = _turn(adapter, adapter.initial_state()).state
        config = _request_config()
        config["stop_sequences"] = ["END"]
        config["output_config"]["format"] = {
            "type": "json_schema",
            "schema": {"type": "object"},
        }
        snapshot = deepcopy(config)
        result = adapter.invoke_turn(
            ModelTurnRequest(
                system_prompt="system",
                new_messages=[Message(role="user", content="latest")],
                request_config=config,
                previous_state=state,
            )
        )
        summary_config = low_level.requests[1].request_config
        assert "stop_sequences" not in summary_config
        assert summary_config["output_config"] == {"effort": "low"}
        assert summary_config["thinking"] == config["thinking"]
        assert summary_config["max_tokens"] == 8192
        assert low_level.requests[2].request_config == snapshot
        assert config == snapshot
        assert result.state.payload["context_tokens"] == 15

    @pytest.mark.parametrize("placement", ["misplaced", "duplicate", "unsigned"])
    def test_invalid_summary_replay_is_rejected_locally(self, placement):
        block = _compaction_response()["content"]
        messages = [{"role": "assistant", "content": block}]
        if placement == "misplaced":
            messages.insert(0, {"role": "user", "content": "summarized history"})
        elif placement == "duplicate":
            messages.append(deepcopy(messages[0]))
        else:
            block[0].pop("signature")
        adapter, low_level = _adapter([])
        state = adapter.initial_state()
        state.payload["messages"] = messages
        with pytest.raises(ValueError, match="signed compaction block first"):
            _turn(adapter, state)
        assert low_level.requests == []

    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("failed_summary", [False, True])
    def test_pinned_sdk_sends_on_demand_and_replays_signed_summary(
        self, streaming, failed_summary
    ):
        captured = []
        summary = _compaction_response("SDK signed summary")
        summary["content"][0]["encrypted_content"] = None
        if failed_summary:
            summary["content"] = []
            summary["stop_reason"] = "refusal"
            summary["stop_details"] = {"category": "synthetic-category"}

        def handle(request):
            body = json.loads(request.content)
            captured.append(body)
            assert "compact-2026-09-04" in request.headers["anthropic-beta"]
            assert "context_management" not in body
            assert "extra_body" not in body
            if "compaction" in body:
                assert body["compaction"] == {"type": "summarize"}
                assert "fresh frame" not in str(body["messages"])
                raw = summary
            else:
                raw = _raw_response(
                    usage={"input_tokens": 175_000, "output_tokens": 5}
                    if len(captured) == 1
                    else None
                )
            if streaming:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_summary_stream_events(raw),
                )
            return httpx.Response(
                200, json={"id": "msg_test", "type": "message", **raw}
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            config = _request_config()
            config.update(stream=streaming, max_tokens=4096)
            request = ModelTurnRequest(
                system_prompt="system",
                new_messages=[Message(role="user", content="old frame")],
                request_config=config,
                previous_state=adapter.initial_state(),
            )
            first = adapter.invoke_turn(request)
            request.previous_state = first.state
            request.new_messages = [Message(role="user", content="fresh frame")]
            snapshot = first.state.model_dump_json()
            if failed_summary:
                with pytest.raises(InvalidProviderResponseError) as raised:
                    adapter.invoke_turn(request)
                assert raised.value.usage.total_tokens == 120
                assert (
                    raised.value.response["stop_details"]["category"]
                    == "synthetic-category"
                )
                assert len(captured) == 2
            else:
                second = adapter.invoke_turn(request)
                assert captured[2]["messages"][0][
                    "content"
                ] == serialize_replay_content(summary["content"])
                assert captured[2]["messages"][-1] == {
                    "role": "user",
                    "content": "fresh frame",
                }
                assert "compaction" not in captured[2]
                assert second.response.usage.total_tokens == 135
                assert second.transition.compaction_items_returned == 1
                assert "summary-signature" not in str(second.readable_request_messages)
                assert "summary-signature" not in str(second.action_state)
            assert first.state.model_dump_json() == snapshot
            assert captured[1]["messages"] == first.state.payload["messages"]
            assert all(
                body["model"] == "claude-opus-5"
                and body["output_config"]["effort"] == "low"
                for body in captured
            )

    def test_interrupted_summary_stream_preserves_history_and_billed_usage(self):
        def handle(_request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_summary_stream_events(_compaction_response(), complete=False),
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            state = adapter.initial_state()
            state.payload.update(
                messages=[
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "ACTION1"},
                ],
                context_tokens=175_000,
            )
            snapshot = state.model_dump_json()
            with pytest.raises(InvalidProviderResponseError) as raised:
                _turn(adapter, state, "fresh frame")
        assert raised.value.usage.total_tokens == 120
        assert state.model_dump_json() == snapshot


@pytest.mark.unit
class TestAnthropicSDKBoundary:
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("sdk_extras", [False, True])
    def test_real_sdk_roundtrips_native_content(self, streaming, sdk_extras):
        captured = []
        raw = {"id": "msg_test", "type": "message", **_raw_response()}
        raw["usage"]["output_tokens_details"] = {"thinking_tokens": 3}
        if sdk_extras:
            raw["content"][-1]["parsed_output"] = None

        def handle(request):
            captured.append(json.loads(request.content))
            assert "compact-2026-09-04" in request.headers["anthropic-beta"]
            assert (
                "thinking-token-count-2026-05-13" in request.headers["anthropic-beta"]
            )
            if streaming:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_stream_events(sdk_extras=sdk_extras),
                )
            return httpx.Response(200, json=raw)

        with anthropic.Anthropic(
            api_key="synthetic-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            config = _request_config()
            config["stream"] = streaming
            config["max_tokens"] = 4096
            request = ModelTurnRequest(
                system_prompt="system",
                new_messages=[Message(role="user", content="frame")],
                previous_state=adapter.initial_state(),
                request_config=config,
            )
            first = adapter.invoke_turn(request)
            request.previous_state = first.state
            request.new_messages = [Message(role="user", content="next frame")]
            adapter.invoke_turn(request)
        native = first.response.raw_response["content"]
        replay = captured[1]["messages"][-2]["content"]
        assert replay == serialize_replay_content(native)
        assert first.response.usage.reasoning_tokens == 3
        assert captured[0]["system"] == captured[1]["system"] == "system"
        assert all("store" not in body for body in captured)
        assert next(block for block in replay if block["type"] == "redacted_thinking")[
            "data"
        ]
        assert any(
            block["type"] == "thinking"
            and block["thinking"] == ""
            and block["signature"]
            for block in replay
        )
        if streaming:
            assert first.response.usage.total_tokens == 15
            assert first.response.reasoning_text == "SDK thinking"
        else:
            assert replay == serialize_replay_content(raw["content"])

    def test_stream_refusal_details_survive_sdk_accumulation(self):
        def handle(_request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_stream_events(stop_reason="refusal"),
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            with pytest.raises(InvalidProviderResponseError) as raised:
                _turn(adapter, adapter.initial_state())
        assert raised.value.response["stop_details"]["category"] == "synthetic-category"
        assert raised.value.usage.total_tokens == 15
        assert raised.value.usage.reasoning_tokens == 3
        assert "sdk-signature" not in str(raised.value.response)
        assert "sdk-ciphertext" not in str(raised.value.response)

    @pytest.mark.parametrize("transport_error", [False, True])
    def test_interrupted_stream_never_returns_an_action_and_retains_usage(
        self, transport_error
    ):
        def handle(_request):
            kwargs = (
                {"stream": _InterruptedStream()}
                if transport_error
                else {"content": _stream_events(complete=False)}
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, **kwargs
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            state = adapter.initial_state()
            with pytest.raises(InvalidProviderResponseError) as raised:
                _turn(adapter, state)
        assert state.payload == {"messages": []}
        assert raised.value.usage.total_tokens == 15
        assert raised.value.usage.reasoning_tokens == 3
        assert "sdk-signature" not in str(raised.value)

    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("request_id_source", ["header", "body"])
    def test_real_sdk_error_preserves_only_safe_metadata(
        self, streaming, request_id_source
    ):
        def handle(_request):
            return httpx.Response(
                400,
                headers=(
                    {"request-id": "req_test_failure"}
                    if request_id_source == "header"
                    else {}
                ),
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "sdk-signature",
                    },
                    "request_id": (
                        "req_unused_body"
                        if request_id_source == "header"
                        else "req_test_failure"
                    ),
                    "signature": "sdk-signature",
                },
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            config = _request_config()
            config.update(stream=streaming, max_tokens=4096)
            with pytest.raises(InvalidProviderResponseError) as raised:
                adapter.invoke_turn(
                    ModelTurnRequest(
                        system_prompt="system",
                        new_messages=[Message(role="user", content="frame")],
                        request_config=config,
                        previous_state=adapter.initial_state(),
                        max_context_length=1_000_000,
                    )
                )
        assert raised.value.response == {
            "provider_error": {
                "exception_class": "BadRequestError",
                "provider_error_type": "invalid_request_error",
                "http_status": 400,
                "request_id": "req_test_failure",
            }
        }
        assert "sdk-signature" not in str(raised.value)
        assert "sdk-signature" not in str(raised.value.response)

    def test_stream_error_event_preserves_metadata_and_partial_usage(self):
        body = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "private-error-body"},
        }

        def handle(_request):
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "request-id": "req_stream_failure",
                },
                content=_stream_events(complete=False)
                + f"event: error\ndata: {json.dumps(body)}\n\n".encode(),
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            state = adapter.initial_state()
            with pytest.raises(InvalidProviderResponseError) as raised:
                _turn(adapter, state)
        assert raised.value.response["provider_error"] == {
            "exception_class": "APIStatusError",
            "provider_error_type": "overloaded_error",
            "http_status": 200,
            "request_id": "req_stream_failure",
        }
        assert raised.value.usage.total_tokens == 15
        assert raised.value.usage.reasoning_tokens == 3
        assert state.payload == {"messages": []}
        assert "private-error-body" not in str(raised.value.response)

    @pytest.mark.parametrize(
        "body",
        [
            "private-error-body",
            {"error": "private-error-body"},
            {
                "error": {"type": "private error body"},
                "request_id": "private request body\nwith control characters",
            },
            {"error": {"type": "x" * 81}, "request_id": "x" * 201},
        ],
    )
    def test_error_metadata_drops_malformed_or_unbounded_values(self, body):
        response = httpx.Response(
            400, request=httpx.Request("POST", "https://example.com")
        )
        error = anthropic.BadRequestError(
            "private-error-body", response=response, body=body
        )
        assert safe_provider_error_metadata(error) == {
            "exception_class": "BadRequestError",
            "http_status": 400,
        }

    def test_native_request_builder_does_not_mutate_inputs(self):
        native = [{"role": "assistant", "content": _raw_response()["content"]}]
        request = ModelRequest(
            messages=[Message(role="system", content="system")],
            request_config=_request_config(),
            native_input=native,
        )
        kwargs = AnthropicMessagesAdapter._build_request_kwargs(request)
        kwargs["messages"][0]["content"][0]["signature"] = "changed"
        assert request.native_input[0]["content"][0]["signature"] == "secret-signature"
