import json
from copy import deepcopy
from datetime import datetime, timezone

import httpx
import pytest
import yaml
from arcengine import GameAction
from openai import OpenAI

import benchmarking.model_config as model_config
from benchmarking.agent import BenchmarkingAgent
from benchmarking.compaction import (
    SUMMARY_REQUEST_PROMPT,
    SummaryCompactionPolicy,
    SummaryCompactor,
)
from benchmarking.deepseek_runtime import (
    ACTION_TOOL_NAME,
    DEEPSEEK_ADAPTER_ID,
    SUMMARY_TOOL_NAME,
    DeepSeekChatCompletionsAdapter,
    DeepSeekContinuousConversationRuntimeAdapter,
    normalize_deepseek_response,
    validate_deepseek_request,
)
from benchmarking.exceptions import (
    ContextOverflowError,
    InvalidProviderResponseError,
    TransientProviderError,
)
from benchmarking.recording import RunRecord
from benchmarking.runtime_adapters import (
    OpenAIChatCompletionsAdapter,
    build_model_runtime_adapter,
)
from benchmarking.runtime_models import Message, ModelRequest
from benchmarking.runtime_registry import (
    ADAPTER_DESCRIPTORS,
    build_stateful_runtime_adapter,
    resolve_adapter_id,
)
from benchmarking.runtime_state import ModelTurnRequest, RuntimeState

pytestmark = pytest.mark.unit


def _tool_raw(
    arguments: dict,
    reasoning: str | None = "  Keep the key.\n",
    *,
    name: str = ACTION_TOOL_NAME,
    content=None,
    finish: str | None = "tool_calls",
):
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_action",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        ],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chat-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _text_raw(content="ACTION1", *, finish="stop"):
    raw = _tool_raw({"action_type": "ACTION1"}, finish=finish)
    raw["choices"][0]["message"] = {
        "role": "assistant",
        "content": content,
        "reasoning_content": "thought",
    }
    return raw


def _config():
    return {
        "id": "deepseek-test",
        "runtime": {
            "sdk": "openai-python",
            "api": "chat_completions",
            "state": "continuous_conversation",
            "adapter_id": DEEPSEEK_ADAPTER_ID,
            "compaction": {
                "strategy": "harness_summary",
                "trigger_tokens": 1_000,
                "summary_max_output_tokens": 128,
                "summary_input_headroom_tokens": 128,
            },
        },
        "client": {
            "base_url": "https://example.test/v1",
            "api_key_env": "TEST_KEY",
        },
        "request": {
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "max_tokens": 256,
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "low",
                }
            },
        },
        "agent": {"MAX_CONTEXT_LENGTH": 10_000},
    }


def _adapter(responses):
    calls = []
    remaining = iter(responses)

    def handle(request):
        calls.append(json.loads(request.content))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    client = OpenAI(
        api_key="test",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    runtime = _config()["runtime"]
    transport = build_model_runtime_adapter(
        client=client,
        runtime_config=runtime,
        config_id="test",
    )
    adapter = build_stateful_runtime_adapter(
        model_adapter=transport,
        runtime_config=runtime,
        config_id="test",
    )
    return adapter, calls


def _turn(adapter, state=None, content="fresh observation", request_config=None):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="Choose one action.",
            new_messages=[Message(role="user", content=content)],
            request_config=request_config or _config()["request"],
            previous_state=state or adapter.initial_state(),
        )
    )


def _compact(adapter, state, *, request_config=None):
    return SummaryCompactor(
        SummaryCompactionPolicy.model_validate(_config()["runtime"]["compaction"])
    ).compact(
        adapter=adapter,
        state=state,
        request_config=request_config or _config()["request"],
        trigger_tokens=1_000,
        max_context_length=10_000,
        max_retries=1,
    )


def _stream(parts, *, include_usage=True, done=True):
    events = [
        {
            "id": "stream-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        for delta, finish in parts
    ]
    if include_usage:
        events.append(
            {
                "id": "stream-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [],
                "usage": _tool_raw({"action_type": "ACTION1"})["usage"],
            }
        )
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if done:
        content += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=content,
    )


def _stream_error(error):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n",
    )


def test_replays_exact_reasoning_and_closes_tool_call():
    reasoning = "  Move down, then inspect.  "
    adapter, calls = _adapter(
        [
            _tool_raw({"action_type": "ACTION2"}, reasoning),
            _tool_raw({"action_type": "ACTION1"}),
        ]
    )
    first = _turn(adapter)
    restored = RuntimeState.model_validate_json(first.state.model_dump_json())
    second = _turn(adapter, restored)

    assert first.response.output_text == '{"actions":[{"action_type":"ACTION2"}]}'
    assert second.response.output_text == '{"actions":[{"action_type":"ACTION1"}]}'
    assert calls[0]["tools"][0]["function"]["name"] == ACTION_TOOL_NAME
    assert "tool_choice" not in calls[0]
    assert calls[1]["messages"][2] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": "call_action",
                "type": "function",
                "function": {
                    "name": ACTION_TOOL_NAME,
                    "arguments": '{"action_type":"ACTION2"}',
                },
            }
        ],
    }
    assert calls[1]["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_action",
        "content": (
            "The action was accepted. The resulting game state will be provided "
            "in the next user message."
        ),
    }
    assert calls[1]["messages"][4]["content"] == "fresh observation"
    assert len(first.state.accepted_turns) == 1
    assert first.state.accepted_turns[0].end_item == 3


def test_plain_text_action_is_not_a_fallback():
    adapter, _ = _adapter([_text_raw()])
    state = adapter.initial_state()
    with pytest.raises(InvalidProviderResponseError) as captured:
        _turn(adapter, state)
    assert captured.value.usage.total_tokens == 120
    assert not state.accepted_turns


@pytest.mark.parametrize(
    "raw",
    [
        _tool_raw({"action_type": "ACTION1"}, name="other_tool"),
        _tool_raw({"x": 1, "y": 2}),
        _tool_raw({"action_type": "ACTION1"}, finish="stop"),
    ],
)
def test_rejects_invalid_tool_protocol(raw):
    with pytest.raises(InvalidProviderResponseError) as captured:
        normalize_deepseek_response(raw)
    assert captured.value.usage.total_tokens == 120


@pytest.mark.parametrize(
    "arguments",
    [
        {"action_type": "ACTION6", "x": 1},
        {"action_type": "ACTION6", "x": -1, "y": 2},
        {"action_type": "ACTION1", "extra": True},
    ],
)
def test_rejects_invalid_action_arguments(arguments):
    with pytest.raises(InvalidProviderResponseError):
        normalize_deepseek_response(_tool_raw(arguments))


def test_rejects_missing_tool_call_as_provider_error():
    raw = _tool_raw({"action_type": "ACTION1"})
    raw["choices"][0]["message"].pop("tool_calls")
    with pytest.raises(
        InvalidProviderResponseError, match="did not return a tool call"
    ):
        normalize_deepseek_response(raw)


def test_allows_nullable_content_and_empty_reasoning():
    response = normalize_deepseek_response(
        _tool_raw({"action_type": "ACTION1"}, reasoning="", content=None)
    )
    assert response.output_text == '{"actions":[{"action_type":"ACTION1"}]}'
    assert response.reasoning_text == ""


def test_usage_preserves_reasoning_and_cache_without_double_counting():
    raw = _tool_raw({"action_type": "ACTION1"})
    raw["usage"].pop("total_tokens")
    raw["usage"]["prompt_tokens_details"] = {
        "cached_tokens": 10,
        "cache_write_tokens": 4,
    }
    usage = normalize_deepseek_response(raw).usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
        100,
        20,
        120,
    )
    assert (usage.cached_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (
        10,
        4,
        12,
    )


def test_stream_accumulates_reasoning_tool_call_and_usage():
    stream = _stream(
        [
            ({"reasoning_content": "think "}, None),
            (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_stream",
                            "type": "function",
                            "function": {
                                "name": ACTION_TOOL_NAME,
                                "arguments": '{"action_type":',
                            },
                        }
                    ]
                },
                None,
            ),
            (
                {
                    "reasoning_content": "more",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"ACTION2"}'},
                        }
                    ],
                },
                "tool_calls",
            ),
        ]
    )
    adapter, calls = _adapter([stream])
    result = _turn(
        adapter,
        request_config={**_config()["request"], "stream": True},
    )
    assert result.response.output_text == '{"actions":[{"action_type":"ACTION2"}]}'
    assert result.response.reasoning_text == "think more"
    assert result.response.usage.total_tokens == 120
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert result.state.payload["messages"][1]["tool_calls"][0]["id"] == "call_stream"


def test_interrupted_stream_preserves_usage_and_closes():
    class BrokenStream:
        closed = False

        def __iter__(self):
            yield {
                "choices": [],
                "usage": _tool_raw({"action_type": "ACTION1"})["usage"],
            }
            raise httpx.ReadError("disconnected")

        def close(self):
            self.closed = True

    stream = BrokenStream()
    with pytest.raises(InvalidProviderResponseError) as captured:
        DeepSeekChatCompletionsAdapter._consume_stream(stream)
    assert captured.value.usage.total_tokens == 120
    assert stream.closed


def test_streamed_context_error_allows_summary_compaction_to_unwind():
    summary_stream = _stream(
        [
            (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_summary",
                            "type": "function",
                            "function": {
                                "name": SUMMARY_TOOL_NAME,
                                "arguments": '{"summary":"Keep moving east."}',
                            },
                        }
                    ]
                },
                "tool_calls",
            )
        ]
    )
    adapter, calls = _adapter(
        [
            _tool_raw({"action_type": "ACTION1"}),
            _stream_error(
                {
                    "code": "context_length_exceeded",
                    "message": "maximum context length exceeded",
                }
            ),
            summary_stream,
        ]
    )
    accepted = _turn(adapter).state
    result = _compact(
        adapter,
        accepted,
        request_config={**_config()["request"], "stream": True},
    )

    assert result.summary == "Keep moving east."
    assert result.excluded_turns == 1
    assert len(calls[2]["messages"]) < len(calls[1]["messages"])


def test_streaming_rejects_disabled_usage():
    request = {
        **_config()["request"],
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    with pytest.raises(ValueError, match="include_usage=true"):
        validate_deepseek_request(request)


@pytest.mark.parametrize(
    "status,body,error",
    [
        (400, {"code": "context_length_exceeded"}, ContextOverflowError),
        (413, {"message": "maximum context length"}, ContextOverflowError),
        (429, {"message": "rate limited"}, TransientProviderError),
        (503, {"message": "unavailable"}, TransientProviderError),
    ],
)
def test_sdk_error_classification(status, body, error):
    adapter, _ = _adapter([httpx.Response(status, json={"error": body})])
    with pytest.raises(error):
        _turn(adapter)


def test_summary_uses_tool_and_excludes_pending_observations():
    adapter, calls = _adapter(
        [
            _tool_raw({"action_type": "ACTION1"}, reasoning="exact thought"),
            _tool_raw(
                {"summary": "Keep moving east."},
                name=SUMMARY_TOOL_NAME,
            ),
            _tool_raw({"action_type": "ACTION2"}),
        ]
    )
    accepted = _turn(adapter).state
    state = adapter.buffer_inputs(
        accepted,
        [Message(role="user", content="GAME_OVER buffered observation")],
    )
    snapshot = state.model_dump()
    result = _compact(adapter, state)

    assert result.summary == "Keep moving east."
    assert calls[1]["tools"][0]["function"]["name"] == SUMMARY_TOOL_NAME
    assert calls[1]["messages"][2]["reasoning_content"] == "exact thought"
    assert calls[1]["messages"][3]["role"] == "tool"
    assert calls[1]["messages"][-1]["content"] == SUMMARY_REQUEST_PROMPT
    assert "GAME_OVER" not in json.dumps(calls[1])
    assert state.model_dump() == snapshot
    _turn(adapter, result.state, "fresh reset frame")
    assert [message["content"] for message in calls[2]["messages"][-2:]] == [
        "GAME_OVER buffered observation",
        "fresh reset frame",
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"tools": []},
        {"tool_choice": "auto"},
        {"response_format": {"type": "json_object"}},
        {"stop": "ACTION"},
        {"max_output_tokens": 1},
        {"max_completion_tokens": 1},
        {"max_tokens": 0},
        {"store": True},
        {"n": 2},
        {"stream": "true"},
        {"extra_body": {"messages": []}},
        {"extra_body": {"tools": []}},
        {"extra_body": {"max_tokens": 9}},
        {"extra_body": []},
    ],
)
def test_request_contract_rejects_overrides(override):
    with pytest.raises(ValueError):
        validate_deepseek_request({**_config()["request"], **override})


def test_registry_and_config_preserve_standard_chat_completions(tmp_path, monkeypatch):
    config = _config()
    path = tmp_path / "model_configs.yaml"
    path.write_text(yaml.safe_dump([config]))
    monkeypatch.setattr(model_config, "MODEL_CONFIG_PATH", path)
    assert model_config.load_model_configs()[0] == config
    runtime = config["runtime"]
    assert resolve_adapter_id(runtime, "test") == DEEPSEEK_ADAPTER_ID
    assert (
        resolve_adapter_id(
            {key: value for key, value in runtime.items() if key != "adapter_id"},
            "test",
        )
        == DEEPSEEK_ADAPTER_ID
    )
    assert isinstance(
        build_model_runtime_adapter(
            client=None,
            runtime_config=runtime,
            config_id="test",
        ),
        DeepSeekChatCompletionsAdapter,
    )

    standard = {
        "sdk": "openai-python",
        "api": "chat_completions",
        "state": "manual_rolling",
    }
    assert resolve_adapter_id(standard, "test") == "openai.chat_completions.v1"
    assert isinstance(
        build_model_runtime_adapter(
            client=None,
            runtime_config=standard,
            config_id="test",
        ),
        OpenAIChatCompletionsAdapter,
    )


@pytest.mark.parametrize("field", ["reasoning_replay", "tool_calling"])
def test_config_rejects_removed_generic_runtime_modes(tmp_path, monkeypatch, field):
    config = _config()
    config["runtime"][field] = True
    path = tmp_path / "model_configs.yaml"
    path.write_text(yaml.safe_dump([config]))
    monkeypatch.setattr(model_config, "MODEL_CONFIG_PATH", path)
    with pytest.raises(ValueError, match="DeepSeek tool behavior is fixed"):
        model_config.load_model_configs()


def test_checked_in_config_is_the_single_deepseek_profile():
    configs = model_config.load_model_configs()
    deepseek = [config for config in configs if "deepseek" in config["id"]]
    assert [config["id"] for config in deepseek] == [
        "deepseek-v4-1-flash-low-provider-adapter"
    ]
    assert deepseek[0]["runtime"]["adapter_id"] == DEEPSEEK_ADAPTER_ID


def test_input_mutation_cannot_modify_accepted_state():
    class MutatingTransport:
        def invoke(self, request: ModelRequest):
            request.native_input[1]["content"] = "mutated"
            return normalize_deepseek_response(_tool_raw({"action_type": "ACTION1"}))

    adapter = DeepSeekContinuousConversationRuntimeAdapter(
        model_adapter=MutatingTransport(),
        descriptor=ADAPTER_DESCRIPTORS[DEEPSEEK_ADAPTER_ID],
    )
    first = _turn(adapter).state
    snapshot = first.model_dump()
    _turn(adapter, first)
    assert first.model_dump() == snapshot


def _agent(adapter, tmp_path):
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent._stateful_adapter = adapter
    agent._runtime_state = adapter.initial_state()
    agent._pending_turn_messages = [Message(role="user", content="newest frame")]
    agent._request_kwargs = _config()["request"]
    agent.MAX_RETRIES = 2
    agent.MAX_CONTEXT_LENGTH = 10_000
    agent.ESTIMATED_CHARS_PER_TOKEN = 1.0
    agent.analysis_mode = False
    agent.token_counter = 0
    agent.conversation = []
    agent._summary_compactor = SummaryCompactor(
        SummaryCompactionPolicy.model_validate(_config()["runtime"]["compaction"])
    )
    agent._pending_compaction_trigger_tokens = None
    agent._compaction_counter = 0
    agent._pricing = {}
    agent.MODEL = "test-model"
    agent.step_counter = 0
    agent.run_dir = str(tmp_path)
    agent.run_record = RunRecord(
        run_id="test",
        game_id="test",
        agent_name="test",
        model="test-model",
        started_at=datetime.now(timezone.utc),
        run_dir=str(tmp_path),
        runtime={},
    )
    return agent


def test_agent_retries_invalid_tool_action_from_last_accepted_state(tmp_path):
    adapter, calls = _adapter(
        [
            _tool_raw({"action_type": "ACTION7"}, reasoning="rejected"),
            _tool_raw({"action_type": "ACTION1"}, reasoning="accepted"),
        ]
    )
    agent = _agent(adapter, tmp_path)
    response, action, retries, _ = agent._request_with_retries([GameAction.ACTION1])
    assert action == GameAction.ACTION1
    assert retries == 1
    assert response.usage.total_tokens == agent.token_counter == 240
    assert calls[0]["messages"] == calls[1]["messages"]
    assert "rejected" not in agent._runtime_state.model_dump_json()
    assert len(agent._runtime_state.accepted_turns) == 1


def test_summary_preserves_deepseek_settings_and_replaces_output_limit():
    request = deepcopy(_config()["request"])
    adapter, calls = _adapter(
        [
            _tool_raw({"action_type": "ACTION1"}),
            _tool_raw({"summary": "summary"}, name=SUMMARY_TOOL_NAME),
        ]
    )
    accepted = _turn(adapter, request_config=request).state
    _compact(adapter, accepted, request_config=request)
    assert calls[0]["max_tokens"] == 256
    assert calls[1]["max_tokens"] == 128
    assert (
        calls[1]["chat_template_kwargs"]
        == request["extra_body"]["chat_template_kwargs"]
    )
    assert request["max_tokens"] == 256
