import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
from benchmarking.exceptions import (
    CompactionContextOverflowError,
    CompactionFailureError,
    ContextOverflowError,
    InvalidProviderResponseError,
    TransientProviderError,
)
from benchmarking.open_source_runtime import (
    OPEN_SOURCE_ADAPTER_ID,
    OpenSourceChatCompletionsAdapter,
    OpenSourceContinuousConversationRuntimeAdapter,
    normalize_open_source_response,
    validate_open_source_request,
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
from benchmarking.runtime_state import RuntimeState

pytestmark = pytest.mark.unit


def _raw(
    content="ACTION1",
    reasoning="  Keep the key.\n",
    *,
    field="reasoning_content",
    finish="stop",
):
    return {
        "id": "chat-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    field: reasoning,
                },
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


def _config(replay="reasoning_content"):
    return {
        "id": "open-source-test",
        "runtime": {
            "sdk": "openai-python",
            "api": "chat_completions",
            "state": "continuous_conversation",
            "adapter_id": OPEN_SOURCE_ADAPTER_ID,
            "reasoning_replay": replay,
            "compaction": {
                "strategy": "harness_summary",
                "trigger_tokens": 1_000,
                "summary_max_output_tokens": 128,
                "summary_input_headroom_tokens": 128,
            },
        },
        "client": {"base_url": "https://example.test/v1", "api_key_env": "TEST_KEY"},
        "request": {"model": "test-model", "max_tokens": 256},
        "agent": {"MAX_CONTEXT_LENGTH": 10_000},
    }


def _adapter(responses, replay="reasoning_content"):
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
    runtime = _config(replay)["runtime"]
    transport = build_model_runtime_adapter(
        client=client, runtime_config=runtime, config_id="test"
    )
    return build_stateful_runtime_adapter(
        model_adapter=transport, runtime_config=runtime, config_id="test"
    ), calls


def _turn(adapter, state=None, content="fresh observation", request_config=None):
    from benchmarking.runtime_state import ModelTurnRequest

    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="Choose one action.",
            new_messages=[Message(role="user", content=content)],
            request_config=request_config or _config()["request"],
            previous_state=state or adapter.initial_state(),
        )
    )


def _compact(adapter, state, *, retries=1, capacity=10_000, request_config=None):
    return SummaryCompactor(
        SummaryCompactionPolicy.model_validate(_config()["runtime"]["compaction"])
    ).compact(
        adapter=adapter,
        state=state,
        request_config=request_config or _config()["request"],
        trigger_tokens=1_000,
        max_context_length=capacity,
        max_retries=retries,
    )


@pytest.mark.parametrize("source_field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize(
    "replay", ["reasoning_content", "reasoning", "reasoning_aliases"]
)
def test_sdk_round_trip_preserves_native_reasoning_exactly(source_field, replay):
    reasoning = "  ACTION7 is a bad idea.\nTry the key.  "
    adapter, calls = _adapter(
        [_raw(reasoning=reasoning, field=source_field), _raw()], replay
    )
    initial = adapter.initial_state()
    first = _turn(adapter, initial)
    snapshot = first.state.model_dump()
    restored = RuntimeState.model_validate_json(first.state.model_dump_json())
    second = _turn(adapter, restored)
    assistant = calls[1]["messages"][2]
    assert assistant["role"] == "assistant"
    if replay == "reasoning_aliases":
        assert assistant == {
            "role": "assistant",
            "content": "ACTION1",
            "reasoning_content": reasoning,
            "reasoning": reasoning,
        }
    else:
        assert assistant == {
            "role": "assistant",
            "content": "ACTION1",
            replay: reasoning,
        }
    assert initial.payload["messages"] == []
    assert first.state.model_dump() == snapshot
    assert first.response.reasoning_text == reasoning
    assert first.response.usage.cached_tokens == 40
    assert first.response.usage.reasoning_tokens == 12
    assert first.response.usage.total_tokens == 120
    assert second.readable_request_messages == calls[1]["messages"]
    assert "store" not in calls[0]


@pytest.mark.parametrize("reasoning", [None, ""])
def test_non_thinking_or_empty_thinking_turns(reasoning):
    adapter, calls = _adapter([_raw(reasoning=reasoning), _raw()], "reasoning_content")
    first = _turn(adapter)
    _turn(adapter, first.state)
    expected = {"role": "assistant", "content": "ACTION1"}
    if reasoning is not None:
        expected["reasoning_content"] = reasoning
    assert calls[1]["messages"][2] == expected


@pytest.mark.parametrize(
    "finish", ["length", "content_filter", "tool_calls", None, "unknown"]
)
def test_rejects_valid_looking_unfinished_actions_with_usage(finish):
    adapter, _ = _adapter([_raw(finish=finish)])
    state = adapter.initial_state()
    with pytest.raises(InvalidProviderResponseError) as captured:
        _turn(adapter, state)
    assert captured.value.usage.total_tokens == 120
    assert not state.accepted_turns


@pytest.mark.parametrize(
    "message",
    [
        {"content": None},
        {"content": ""},
        {"content": " "},
        {"refusal": "no"},
        {"tool_calls": [{"id": "tool"}]},
        {"function_call": {"name": "tool"}},
        {"reasoning_content": ["not text"]},
        {"reasoning": "conflicting"},
        {
            "reasoning_details": [
                {"type": "reasoning.encrypted", "encrypted_content": "OPAQUE"}
            ]
        },
    ],
)
def test_rejects_empty_malformed_or_opaque_outputs(message):
    raw = _raw()
    raw["choices"][0]["message"].update(message)
    with pytest.raises(InvalidProviderResponseError) as captured:
        normalize_open_source_response(raw)
    assert captured.value.usage.total_tokens == 120
    assert "OPAQUE" not in json.dumps(captured.value.response)


def test_empty_choices_keep_usage():
    raw = _raw()
    raw["choices"] = []
    with pytest.raises(InvalidProviderResponseError) as captured:
        normalize_open_source_response(raw)
    assert captured.value.usage.total_tokens == 120


def test_usage_fallback_and_no_double_counting():
    raw = _raw()
    raw["usage"].pop("total_tokens")
    raw["usage"]["prompt_tokens_details"] = {
        "cached_tokens": 10,
        "cache_write_tokens": 4,
    }
    raw["usage"]["cost"] = 0.1
    raw["usage"]["cost_details"] = {"upstream_inference_cost": 0.1}
    usage = normalize_open_source_response(raw).usage
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
    assert usage.cost == 0.1


def _stream(parts, *, usage=True, done=True):
    events = []
    for delta, finish in parts:
        events.append(
            {
                "id": "stream-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )
    if usage:
        events.append(
            {
                "id": "stream-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [],
                "usage": _raw()["usage"],
            }
        )
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if done:
        content += "data: [DONE]\n\n"
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=content
    )


def _stream_error(error):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n",
    )


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_sdk_stream_accumulates_reasoning_and_usage_trailer(field):
    stream = _stream(
        [
            ({field: "think "}, None),
            ({field: "more"}, None),
            ({"content": "ACTION"}, None),
            ({"content": "1"}, "stop"),
        ]
    )
    adapter, calls = _adapter([stream, _raw()], field)
    first = _turn(adapter, request_config={**_config()["request"], "stream": True})
    assert first.response.reasoning_text == "think more"
    assert first.response.usage.total_tokens == 120
    assert calls[0]["stream_options"] == {"include_usage": True}
    _turn(adapter, first.state)
    assert calls[1]["messages"][2][field] == "think more"


@pytest.mark.parametrize("finish", [None, "length", "content_filter", "tool_calls"])
def test_incomplete_stream_never_accepts_action(finish):
    adapter, _ = _adapter([_stream([({"content": "ACTION1"}, finish)])])
    with pytest.raises(InvalidProviderResponseError) as captured:
        _turn(adapter, request_config={**_config()["request"], "stream": True})
    assert captured.value.usage.total_tokens == 120


def test_interrupted_stream_preserves_observed_usage_and_closes():
    class BrokenStream:
        closed = False

        def __iter__(self):
            yield {"choices": [], "usage": _raw()["usage"]}
            yield {"choices": [{"index": 0, "delta": {"content": "ACTION1"}}]}
            raise httpx.ReadError("disconnected")

        def close(self):
            self.closed = True

    stream = BrokenStream()
    with pytest.raises(InvalidProviderResponseError) as captured:
        OpenSourceChatCompletionsAdapter._consume_stream(stream)
    assert captured.value.usage.total_tokens == 120
    assert stream.closed


@pytest.mark.parametrize(
    "status,body,error",
    [
        (
            400,
            {"code": "context_length_exceeded", "message": "too long"},
            ContextOverflowError,
        ),
        (413, {"message": "maximum context length"}, ContextOverflowError),
        (422, {"message": "input exceeds max_seq_len"}, ContextOverflowError),
        (429, {"message": "rate limited"}, TransientProviderError),
        (503, {"message": "unavailable"}, TransientProviderError),
    ],
)
def test_real_sdk_error_classification(status, body, error):
    adapter, _ = _adapter([httpx.Response(status, json={"error": body})])
    with pytest.raises(error):
        _turn(adapter)


def test_connection_error_classification():
    adapter, _ = _adapter([httpx.ConnectError("offline")])
    with pytest.raises(TransientProviderError):
        _turn(adapter)


def test_streamed_context_error_allows_summary_compaction_to_unwind():
    adapter, calls = _adapter(
        [
            _raw(),
            _stream_error(
                {
                    "code": "context_length_exceeded",
                    "message": "maximum context length exceeded",
                }
            ),
            _stream([({"content": "Keep moving east."}, "stop")]),
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
        validate_open_source_request(request)


@pytest.mark.parametrize(
    "replay", ["reasoning_content", "reasoning", "reasoning_aliases"]
)
def test_summary_sees_reasoning_but_not_pending_observations(replay):
    adapter, calls = _adapter(
        [_raw(), _raw("Keep the key; the door is east."), _raw()], replay
    )
    accepted = _turn(adapter).state
    state = adapter.buffer_inputs(
        accepted, [Message(role="user", content="GAME_OVER buffered observation")]
    )
    snapshot = state.model_dump()
    result = _compact(adapter, state)
    assert "Keep the key" in json.dumps(calls[1]["messages"])
    assert "GAME_OVER" not in json.dumps(calls[1])
    assert "GAME_OVER" not in json.dumps(result.prompt)
    assert calls[1]["messages"][-1]["content"] == SUMMARY_REQUEST_PROMPT
    assert calls[1]["max_tokens"] == 128
    assert result.history_items_to_compact == 2
    assert result.state.payload["messages"][0]["content"].endswith(result.summary)
    assert state.model_dump() == snapshot
    _turn(adapter, result.state, "fresh reset frame")
    assert [message["content"] for message in calls[2]["messages"][-2:]] == [
        "GAME_OVER buffered observation",
        "fresh reset frame",
    ]


def test_summary_overflow_restores_exact_recent_turns_and_pending_inputs():
    overflow = httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
    adapter, calls = _adapter(
        [
            _raw(reasoning="old"),
            _raw(reasoning="  recent exact  "),
            overflow,
            _raw("Old discovery summary"),
            _raw(),
        ],
        "reasoning",
    )
    first = _turn(adapter).state
    second = _turn(adapter, first, "recent observation").state
    state = adapter.buffer_inputs(second, [Message(role="user", content="pending")])
    result = _compact(adapter, state)
    assert result.overflow_recoveries == 1
    assert result.excluded_turns == 1
    assert result.state.payload["messages"][1:] == second.payload["messages"][2:]
    assert "recent exact" not in json.dumps(calls[3])
    _turn(adapter, result.state)
    assert calls[4]["messages"][3]["reasoning"] == "  recent exact  "
    assert calls[4]["messages"][-2]["content"] == "pending"


def test_repeated_summaries_retain_previous_summary_and_new_reasoning():
    adapter, calls = _adapter(
        [
            _raw(),
            _raw("summary one"),
            _raw(reasoning="second discovery"),
            _raw("summary two"),
        ]
    )
    first = _compact(adapter, _turn(adapter).state)
    second = _compact(adapter, _turn(adapter, first.state).state)
    assert "summary one" in json.dumps(calls[3])
    assert "second discovery" in json.dumps(calls[3])
    assert len(second.state.payload["messages"]) == 1
    assert second.summary == "summary two"


def test_failed_summary_does_not_commit_and_counts_all_attempts():
    adapter, _ = _adapter([_raw(), _raw("partial", finish="length"), _raw("")])
    state = adapter.buffer_inputs(
        _turn(adapter).state, [Message(role="user", content="pending")]
    )
    snapshot = state.model_dump()
    with pytest.raises(CompactionFailureError) as captured:
        _compact(adapter, state)
    assert captured.value.usage.total_tokens == 240
    assert state.model_dump() == snapshot


def test_summary_retries_keep_failed_usage_and_stop_at_protected_boundary():
    overflow = httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
    adapter, calls = _adapter(
        [_raw(), _raw("partial", finish="length"), overflow, overflow]
    )
    with pytest.raises(CompactionContextOverflowError) as captured:
        _compact(adapter, _turn(adapter).state)
    assert captured.value.usage.total_tokens == 120
    assert len(calls) == 4


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


def test_agent_retries_from_accepted_state_and_parses_only_final_answer(tmp_path):
    adapter, calls = _adapter(
        [
            _raw("not an action", reasoning="BAD1"),
            _raw("ACTION1", reasoning="BAD2", finish="length"),
            _raw("ACTION1", reasoning="ACTION7"),
        ]
    )
    agent = _agent(adapter, tmp_path)
    response, action, retries, _ = agent._request_with_retries(
        [GameAction.ACTION1, GameAction.ACTION7]
    )
    assert action == GameAction.ACTION1
    assert retries == 2
    assert response.usage.total_tokens == agent.token_counter == 360
    assert calls[0]["messages"] == calls[1]["messages"] == calls[2]["messages"]
    assert "BAD" not in agent._runtime_state.model_dump_json()
    assert len(agent._runtime_state.accepted_turns) == 1


def test_agent_action_overflow_compacts_before_retrying_fresh_frame(tmp_path):
    overflow = httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
    adapter, calls = _adapter([_raw(), overflow, _raw("summary"), _raw()])
    agent = _agent(adapter, tmp_path)
    agent._runtime_state = _turn(adapter, content="old frame").state
    agent._runtime_state = adapter.buffer_inputs(
        agent._runtime_state, [Message(role="user", content="buffered reset")]
    )
    response, action, _, _ = agent._request_with_retries([GameAction.ACTION1])
    assert action == GameAction.ACTION1
    assert "newest frame" not in json.dumps(calls[2])
    assert "buffered reset" not in json.dumps(calls[2])
    assert [message["content"] for message in calls[3]["messages"][-2:]] == [
        "buffered reset",
        "newest frame",
    ]
    assert agent.token_counter == 240
    assert response.usage.total_tokens == 120
    assert agent._pending_compaction_usage.total_tokens == 120
    artifact = json.loads((tmp_path / "compaction_001.json").read_text())
    assert artifact["summary"] == "summary"
    assert "Keep the key" in json.dumps(artifact["prompt"])
    assert "buffered reset" not in json.dumps(artifact)


@pytest.mark.parametrize(
    "override",
    [
        {"tools": []},
        {"response_format": {"type": "json_object"}},
        {"stop": "ACTION"},
        {"max_output_tokens": 1},
        {"max_completion_tokens": 1},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"store": True},
        {"n": 2},
        {"stream": "true"},
        {"extra_body": {"messages": []}},
        {"extra_body": {"max_tokens": 9}},
        {"extra_body": {"context_management": []}},
        {"extra_body": {"store": True}},
        {"extra_body": []},
    ],
)
def test_request_contract_rejects_history_overrides_and_incompatible_options(override):
    with pytest.raises(ValueError):
        validate_open_source_request({**_config()["request"], **override})


def test_registry_and_config_validation_preserve_standard_runtime(
    tmp_path, monkeypatch
):
    config = _config()
    path = tmp_path / "model_configs.yaml"
    path.write_text(yaml.safe_dump([config]))
    monkeypatch.setattr(model_config, "MODEL_CONFIG_PATH", path)
    assert model_config.load_model_configs()[0] == config
    runtime = config["runtime"]
    with pytest.raises(ValueError, match="requires runtime.adapter_id"):
        resolve_adapter_id(
            {key: value for key, value in runtime.items() if key != "adapter_id"},
            "test",
        )
    legacy = {
        "sdk": "openai-python",
        "api": "chat_completions",
        "state": "manual_rolling",
    }
    assert resolve_adapter_id(legacy, "test") == "openai.chat_completions.v1"
    assert isinstance(
        build_model_runtime_adapter(
            client=None, runtime_config=legacy, config_id="test"
        ),
        OpenAIChatCompletionsAdapter,
    )
    assert isinstance(
        build_model_runtime_adapter(
            client=None, runtime_config=runtime, config_id="test"
        ),
        OpenSourceChatCompletionsAdapter,
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_adapter_id({**legacy, "adapter_id": OPEN_SOURCE_ADAPTER_ID}, "test")


@pytest.mark.parametrize(
    "change",
    [
        {"reasoning_replay": "auto"},
        {"reasoning_replay": []},
        {"state": "manual_rolling"},
        {"compaction": {"strategy": "native"}},
        {"compaction": {"strategy": "harness_summary", "trigger_tokens": 20_000}},
    ],
)
def test_config_rejects_invalid_replay_and_compaction(tmp_path, monkeypatch, change):
    config = _config()
    config["runtime"].update(change)
    path = tmp_path / "model_configs.yaml"
    path.write_text(yaml.safe_dump([config]))
    monkeypatch.setattr(model_config, "MODEL_CONFIG_PATH", path)
    with pytest.raises(ValueError):
        model_config.load_model_configs()


def test_state_rejects_different_replay_mode():
    adapter, _ = _adapter([])
    other, _ = _adapter([], "reasoning")
    with pytest.raises(ValueError, match="reasoning_replay"):
        other.buffer_inputs(adapter.initial_state(), [])


def test_run_metadata_records_replay_mode(tmp_path, monkeypatch):
    config = _config("reasoning_content")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmarking.agent.get_model_config", lambda _: deepcopy(config)
    )
    monkeypatch.setattr(
        "benchmarking.agent.build_model_runtime_client", lambda **_: None
    )
    agent = BenchmarkingAgent(
        card_id="test",
        game_id="test",
        agent_name="test",
        ROOT_URL="http://test",
        record=False,
        arc_env=SimpleNamespace(info=SimpleNamespace(baseline_actions=[])),
        config=config["id"],
    )
    metadata = json.loads((Path(agent.run_dir) / "run_meta.json").read_text())
    assert metadata["runtime"]["reasoning_replay"] == "reasoning_content"
    assert metadata["runtime"]["adapter_id"] == OPEN_SOURCE_ADAPTER_ID
    assert "carry forward" not in agent._build_system_prompt()


def test_input_mutation_cannot_modify_accepted_state():
    class MutatingTransport:
        def invoke(self, request: ModelRequest):
            request.native_input[1]["content"] = "mutated"
            return normalize_open_source_response(_raw())

    adapter = OpenSourceContinuousConversationRuntimeAdapter(
        model_adapter=MutatingTransport(),
        descriptor=ADAPTER_DESCRIPTORS[OPEN_SOURCE_ADAPTER_ID],
    )
    first = _turn(adapter).state
    snapshot = first.model_dump()
    _turn(adapter, first)
    assert first.model_dump() == snapshot


def test_checked_in_open_source_low_profiles_use_generic_adapter():
    configs = model_config.load_model_configs()
    profiles = [
        config
        for config in configs
        if config["id"]
        in {
            "zai-glm-5-3-flash-low-provider-adapter",
            "alibaba-qwen3-8-27b-low-provider-adapter",
        }
    ]
    assert {
        resolve_adapter_id(config["runtime"], config["id"]) for config in profiles
    } == {OPEN_SOURCE_ADAPTER_ID}
    assert [config["id"] for config in profiles] == [
        "zai-glm-5-3-flash-low-provider-adapter",
        "alibaba-qwen3-8-27b-low-provider-adapter",
    ]
    assert profiles[0]["runtime"]["reasoning_replay"] == "reasoning_content"
    assert profiles[0]["request"]["reasoning_effort"] == "low"
    assert "temperature" not in profiles[0]["request"]
    assert "top_p" not in profiles[0]["request"]
    assert "extra_body" not in profiles[0]["request"]
    assert profiles[1]["runtime"]["reasoning_replay"] == "reasoning_aliases"
    assert profiles[1]["client"] == {
        "base_url": "https://model-wglyjv63.api.baseten.co/environments/production/sync/v1",
        "api_key_env": "BASETEN_API_KEY",
    }
    assert profiles[1]["agent"]["MAX_RETRIES"] == 7
    assert profiles[1]["request"]["reasoning_effort"] == "low"
    assert "temperature" not in profiles[1]["request"]
    assert "top_p" not in profiles[1]["request"]
    assert "presence_penalty" not in profiles[1]["request"]
    assert "extra_body" not in profiles[1]["request"]


@pytest.mark.parametrize("output_key", ["max_tokens", "max_completion_tokens"])
def test_summary_preserves_provider_toggles_and_replaces_correct_output_limit(
    output_key,
):
    request = {
        "model": "test-model",
        output_key: 256,
        "extra_body": {
            "thinking": {"type": "enabled", "clear_thinking": False},
        },
    }
    adapter, calls = _adapter([_raw(), _raw("summary")], "reasoning_content")
    accepted = _turn(adapter, request_config=request).state
    _compact(adapter, accepted, request_config=request)
    assert calls[0][output_key] == 256
    assert calls[1][output_key] == 128
    assert "max_output_tokens" not in calls[1]
    assert calls[1]["thinking"] == request["extra_body"]["thinking"]
    assert request[output_key] == 256


@pytest.mark.parametrize(
    "field,value",
    [
        ("refusal", "No"),
        ("tool_calls", [{"id": "tool"}]),
        ("reasoning_details", [{"type": "reasoning.encrypted", "data": "OPAQUE"}]),
    ],
)
def test_stream_rejections_keep_trailer_usage_without_opaque_diagnostics(field, value):
    adapter, _ = _adapter(
        [_stream([({field: value}, None), ({"content": "ACTION1"}, "stop")])]
    )
    with pytest.raises(InvalidProviderResponseError) as captured:
        _turn(adapter, request_config={**_config()["request"], "stream": True})
    assert captured.value.usage.total_tokens == 120
    assert "OPAQUE" not in json.dumps(captured.value.response)


def test_summary_transient_retry_and_billed_failure_usage(monkeypatch):
    monkeypatch.setattr("benchmarking.compaction.time.sleep", lambda _: None)
    adapter, calls = _adapter(
        [
            _raw(),
            httpx.Response(429, json={"error": {"message": "busy"}}),
            _raw("partial", finish="length"),
            _raw("summary"),
        ]
    )
    result = _compact(adapter, _turn(adapter).state, retries=2)
    assert result.attempts == 3
    assert result.usage.total_tokens == 240
    assert calls[1]["messages"] == calls[2]["messages"] == calls[3]["messages"]


def test_oversized_summary_preserves_usage_and_original_state():
    adapter, _ = _adapter([_raw(), _raw("summary " * 1_000)])
    state = _turn(adapter).state
    original = state.model_dump()
    with pytest.raises(CompactionContextOverflowError) as captured:
        _compact(adapter, state, capacity=1_000)
    assert captured.value.usage.total_tokens == 120
    assert state.model_dump() == original
