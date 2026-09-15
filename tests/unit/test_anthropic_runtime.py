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
    prune_after_latest_compaction,
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


def _adapter(responses):
    low_level = _FakeModelAdapter(responses)
    adapter = build_stateful_runtime_adapter(
        model_adapter=low_level,
        runtime_config=get_model_config(CONFIG_ID)["runtime"],
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
        def compacted(summary):
            return _raw_response(
                blocks=[
                    {"type": "compaction", "content": summary},
                    {
                        "type": "thinking",
                        "thinking": "after",
                        "signature": "new-signature",
                    },
                    {"type": "text", "text": "ACTION1"},
                ]
            )

        adapter, low_level = _adapter(
            [
                _raw_response(),
                compacted("summary one"),
                _raw_response(),
                compacted("summary two"),
                _raw_response(),
            ]
        )
        first = _turn(adapter, adapter.initial_state(), "old frame")
        buffered = adapter.buffer_inputs(
            first.state, [Message(role="user", content="GAME_OVER")]
        )
        second = _turn(adapter, buffered, "reset frame")
        assert [
            message["content"] for message in low_level.requests[1].native_input[-2:]
        ] == ["GAME_OVER", "reset frame"]
        assert second.transition.history_items_before_prune == 5
        assert second.transition.history_items_after_prune == 1
        third = _turn(adapter, second.state, "post-compaction")
        readable = str(third.readable_request_messages)
        assert "summary one" in readable
        assert "old frame" not in readable
        assert "reset frame" not in readable
        fourth = _turn(adapter, third.state, "another")
        fifth = _turn(adapter, fourth.state, "final")
        assert "summary two" in str(fifth.readable_request_messages)
        assert "summary one" not in str(fifth.readable_request_messages)
        assert (
            fifth.state.payload["messages"][0]["content"][1]["signature"]
            == "new-signature"
        )

    def test_prunes_at_latest_successful_block_not_failed_block(self):
        messages = [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "content": "earlier"},
                    {"type": "compaction", "content": "latest"},
                    {"type": "compaction", "content": None},
                    {"type": "text", "text": "ACTION1"},
                ],
            },
            {"role": "user", "content": "pending"},
        ]
        snapshot = deepcopy(messages)
        pruned = prune_after_latest_compaction(messages)
        assert messages == snapshot
        assert pruned[0]["content"] == messages[1]["content"][1:]
        assert pruned[1]["content"] == "pending"

    def test_null_compaction_preserves_history(self):
        raw = _raw_response()
        raw["content"].insert(0, {"type": "compaction", "content": None})
        adapter, _ = _adapter([raw])
        result = _turn(adapter, adapter.initial_state())
        assert result.state.payload["messages"][0]["content"] == "frame"
        assert (
            result.transition.history_items_before_prune
            == result.transition.history_items_after_prune
        )

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
    def test_checked_in_profile_and_registry(self):
        config = get_model_config(CONFIG_ID)
        assert config["request"]["model"] == "claude-opus-5"
        assert config["request"]["output_config"] == {"effort": "low"}
        assert config["request"]["max_tokens"] == 128_000
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
            {"betas": []},
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
        request["context_management"]["edits"] = [edit]
        with pytest.raises(ValueError):
            validate_continuous_conversation_request(request)

    @pytest.mark.parametrize("explicit_adapter", [False, True])
    def test_compaction_is_optional_but_harness_compaction_is_rejected(
        self, explicit_adapter
    ):
        config = deepcopy(get_model_config(CONFIG_ID))
        if not explicit_adapter:
            config["runtime"].pop("adapter_id")
        config["request"].pop("context_management")
        config["request"].pop("betas")
        _validate_model_config_entry(config, 1, set())
        config["runtime"]["compaction"] = {
            "strategy": "harness_summary",
            "trigger_tokens": 175_000,
        }
        with pytest.raises(ValueError, match="native request compaction only"):
            _validate_model_config_entry(config, 1, set())


def _stream_events(*, stop_reason="end_turn", complete=True):
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
            {"type": "compaction", "content": None},
            [{"type": "compaction_delta", "content": "SDK summary"}],
        ),
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
                "iterations": [
                    {"type": "compaction", "input_tokens": 50000, "output_tokens": 100},
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
class TestAnthropicSDKBoundary:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_real_sdk_roundtrips_native_content(self, streaming):
        captured = []
        raw = {"id": "msg_test", "type": "message", **_raw_response()}

        def handle(request):
            captured.append(json.loads(request.content))
            assert "compact-2026-01-12" in request.headers["anthropic-beta"]
            if streaming:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_stream_events(),
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
        assert replay == native
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
            assert first.response.usage.total_tokens == 50115
            assert first.response.reasoning_text == "SDK thinking"
            assert replay[0] == {"type": "compaction", "content": "SDK summary"}
        else:
            assert replay == raw["content"]

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
        assert raised.value.usage.total_tokens == 50115
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
        assert raised.value.usage.total_tokens == 50115
        assert "sdk-signature" not in str(raised.value)

    def test_real_sdk_error_body_does_not_leak_to_logs(self):
        def handle(_request):
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "sdk-signature",
                    },
                },
            )

        with anthropic.Anthropic(
            api_key="synthetic-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            adapter = _sdk_adapter(client)
            with pytest.raises(InvalidProviderResponseError) as raised:
                _turn(adapter, adapter.initial_state())
        assert "sdk-signature" not in str(raised.value)

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
