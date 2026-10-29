import json
from copy import deepcopy
from datetime import datetime, timezone

import httpx
import pytest
from arcengine import GameAction
from openai import OpenAI

from benchmarking.agent import BenchmarkingAgent
from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.exceptions import (
    CompactionFailureError,
    ContextOverflowError,
    InvalidProviderResponseError,
    TransientProviderError,
)
from benchmarking.inception_runtime import (
    INCEPTION_ADAPTER_ID,
    InceptionChatCompletionsAdapter,
    InceptionContinuousConversationRuntimeAdapter,
    normalize_inception_response,
)
from benchmarking.model_config import _validate_model_config_entry
from benchmarking.recording import RunRecord
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import build_stateful_runtime_adapter
from benchmarking.runtime_state import ModelTurnRequest, RuntimeState, sanitize_settings

pytestmark = pytest.mark.unit


def _detail(data="OPAQUE-SECRET", index=0):
    return {
        "type": "reasoning.encrypted",
        "format": "inception-v1",
        "index": index,
        "data": data,
    }


def _raw(content="ACTION1", *, data="OPAQUE-SECRET", finish="stop"):
    return {
        "id": "inception-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mercury-3",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_details": [_detail(data)],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }


def _config():
    return {
        "id": "inception-test",
        "agent": {"MAX_CONTEXT_LENGTH": 10000},
        "runtime": {
            "sdk": "openai-python",
            "api": "chat_completions",
            "state": "continuous_conversation",
            "adapter_id": INCEPTION_ADAPTER_ID,
            "compaction": {"strategy": "harness_summary", "trigger_tokens": 1000},
        },
        "client": {"api_key_env": "TEST_KEY"},
        "request": {
            "model": "mercury-3",
            "max_tokens": 256,
            "reasoning_effort": "high",
        },
    }


@pytest.fixture
def runtime():
    clients = []

    def make(responses):
        calls, remaining = [], iter(responses)

        def handle(request):
            calls.append(json.loads(request.content))
            raw = next(remaining)
            if isinstance(raw, Exception):
                raise raw
            return (
                raw
                if isinstance(raw, httpx.Response)
                else httpx.Response(200, json=raw)
            )

        client = OpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        clients.append(client)
        cfg = _config()
        transport = build_model_runtime_adapter(
            client=client, runtime_config=cfg["runtime"], config_id=cfg["id"]
        )
        assert isinstance(transport, InceptionChatCompletionsAdapter)
        adapter = build_stateful_runtime_adapter(
            model_adapter=transport, runtime_config=cfg["runtime"], config_id=cfg["id"]
        )
        assert isinstance(adapter, InceptionContinuousConversationRuntimeAdapter)
        return adapter, calls

    yield make
    for client in clients:
        client.close()


def _turn(adapter, state=None, content="frame", *, stream=False):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="Choose an action.",
            new_messages=[Message(role="user", content=content)],
            request_config={**_config()["request"], "stream": stream},
            previous_state=state if state is not None else adapter.initial_state(),
        )
    )


def _compact(adapter, state, retries=0):
    return SummaryCompactor(
        SummaryCompactionPolicy(strategy="harness_summary", trigger_tokens=1000)
    ).compact(
        adapter=adapter,
        state=state,
        request_config=_config()["request"],
        trigger_tokens=1000,
        max_context_length=10000,
        max_retries=retries,
    )


def _stream(parts):
    events = [
        {
            "id": "inception-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mercury-3",
            "choices": [{"index": 0, "delta": d, "finish_reason": f}],
        }
        for d, f in parts
    ]
    events.append(
        {
            "id": "inception-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mercury-3",
            "choices": [],
            "usage": _raw()["usage"],
        }
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(e)}\n\n" for e in events)
        + "data: [DONE]\n\n",
    )


def test_native_replay_is_lossless_and_readable_surfaces_are_redacted(runtime):
    first_raw = _raw()
    first_raw["choices"][0]["message"]["reasoning_details"].append(
        _detail("SECOND-SECRET", 1)
    )
    adapter, calls = runtime([first_raw, _raw()])
    first = _turn(adapter)
    snapshot = first.state.model_dump()
    restored = RuntimeState.model_validate_json(first.state.model_dump_json())
    second = _turn(adapter, restored, "next frame")
    assert calls[1]["messages"][2] == first_raw["choices"][0]["message"]
    assert first.state.model_dump() == snapshot
    assert first.response.reasoning_text is None
    assert first.response.usage.reasoning_tokens == 12
    assert first.response.usage.cached_tokens == 40
    assert first.response.usage.total_tokens == 120
    surfaces = [
        second.readable_request_messages,
        second.sanitized_request,
        second.transition.model_dump(),
        second.action_state,
        sanitize_settings(first.response.raw_response),
    ]
    assert "OPAQUE-SECRET" not in json.dumps(surfaces)
    assert "SECOND-SECRET" not in json.dumps(surfaces)
    assert second.transition.sanitized_items[-2]["reasoning_present"]


def test_stream_fragments_and_multiple_blocks_replay_without_loss(runtime):
    stream = _stream(
        [
            ({"reasoning_details": [_detail("AAA", 0), _detail("CCC", 1)]}, None),
            (
                {
                    "reasoning_details": [
                        {"index": 0, "data": "BBB"},
                        {"index": 1, "data": "DDD"},
                    ]
                },
                None,
            ),
            ({"content": "ACTION1"}, "stop"),
        ]
    )
    adapter, calls = runtime([stream, _raw()])
    first = _turn(adapter, stream=True)
    _turn(adapter, first.state)
    assert calls[0]["stream_options"]["include_usage"] is True
    assert calls[1]["messages"][2]["reasoning_details"] == [
        _detail("AAABBB", 0),
        _detail("CCCDDD", 1),
    ]
    assert first.response.usage.total_tokens == 120


@pytest.mark.parametrize(
    "detail",
    [
        {
            "type": "reasoning.encrypted",
            "format": "foreign-v1",
            "index": 0,
            "data": "SECRET",
        },
        {"type": "reasoning.text", "index": 0, "data": "SECRET"},
        {
            "type": "reasoning.encrypted",
            "format": "inception-v1",
            "index": True,
            "data": "SECRET",
        },
        {
            "type": "reasoning.encrypted",
            "format": "inception-v1",
            "index": 0,
            "data": "",
        },
        {
            "type": "reasoning.encrypted",
            "format": "inception-v1",
            "index": 0,
            "data": {},
        },
        {
            "type": "reasoning.encrypted",
            "format": "inception-v1",
            "index": 0,
            "data": "SECRET",
            "unknown": "SECRET",
        },
    ],
)
def test_invalid_details_fail_closed_with_usage(detail):
    raw = _raw()
    raw["choices"][0]["message"]["reasoning_details"] = [detail]
    with pytest.raises(InvalidProviderResponseError) as error:
        normalize_inception_response(raw)
    assert error.value.usage.total_tokens == 120
    assert "SECRET" not in json.dumps(error.value.response)


def test_conflicting_stream_metadata_is_rejected(runtime):
    adapter, _ = runtime(
        [
            _stream(
                [
                    ({"reasoning_details": [_detail()]}, None),
                    ({"reasoning_details": [{"index": 0, "format": "other"}]}, None),
                    ({"content": "ACTION1"}, "stop"),
                ]
            )
        ]
    )
    with pytest.raises(InvalidProviderResponseError):
        _turn(adapter, stream=True)


@pytest.mark.parametrize("finish", ["length", None, "tool_calls", "content_filter"])
def test_unfinished_response_does_not_mutate_state(runtime, finish):
    adapter, _ = runtime([_raw(finish=finish)])
    state = adapter.initial_state()
    before = state.model_dump()
    with pytest.raises(InvalidProviderResponseError) as error:
        _turn(adapter, state)
    assert state.model_dump() == before
    assert error.value.usage.total_tokens == 120


@pytest.mark.parametrize("content", ["", " ", None])
def test_thought_only_response_is_rejected(content):
    with pytest.raises(InvalidProviderResponseError):
        normalize_inception_response(_raw(content))


def test_usage_aliases_do_not_double_count():
    raw = _raw()
    raw["usage"].update(reasoning_tokens=99, cached_input_tokens=80)
    response = normalize_inception_response(raw)
    assert response.usage.reasoning_tokens == 12
    assert response.usage.cached_tokens == 40
    raw["usage"].pop("completion_tokens_details")
    raw["usage"].pop("prompt_tokens_details")
    raw["usage"]["reasoning_tokens"] = 12
    response = normalize_inception_response(raw)
    assert response.usage.reasoning_tokens == 12
    assert response.usage.cached_tokens == 80
    assert response.usage.output_tokens == 20


def test_instant_response_without_reasoning_is_supported(runtime):
    raw = _raw()
    raw["choices"][0]["message"].pop("reasoning_details")
    adapter, calls = runtime([raw, raw])
    first = _turn(adapter)
    _turn(adapter, first.state)
    assert "reasoning_details" not in calls[1]["messages"][2]


def test_compaction_preserves_pending_frames_and_redacts_prompt(runtime):
    adapter, calls = runtime([_raw(), _raw("Remember the key."), _raw()])
    state = adapter.buffer_inputs(
        _turn(adapter).state, [Message(role="user", content="pending RESET frame")]
    )
    snapshot = state.model_dump()
    compacted = _compact(adapter, state)
    assert calls[1]["messages"][2]["reasoning_details"] == [_detail()]
    assert "pending RESET" not in json.dumps(calls[1])
    assert "OPAQUE-SECRET" not in json.dumps(compacted.prompt)
    assert state.model_dump() == snapshot
    _turn(adapter, compacted.state, "fresh observation")
    assert [m["content"] for m in calls[2]["messages"][-2:]] == [
        "pending RESET frame",
        "fresh observation",
    ]
    assert "reasoning_details" not in json.dumps(calls[2])


def test_overflow_unwind_restores_exact_encrypted_recent_turn(runtime):
    overflow = httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
    adapter, calls = runtime(
        [_raw(data="OLD"), _raw(data="RECENT"), overflow, _raw("summary"), _raw()]
    )
    first = _turn(adapter).state
    state = _turn(adapter, first, "second frame").state
    snapshot = state.model_dump()
    compacted = _compact(adapter, state)
    assert compacted.excluded_turns == 1
    assert state.model_dump() == snapshot
    _turn(adapter, compacted.state)
    assert calls[-1]["messages"][3]["reasoning_details"] == [_detail("RECENT")]
    assert "RECENT" not in json.dumps(calls[-2])


def test_failed_compaction_keeps_accepted_state_and_usage(runtime):
    adapter, _ = runtime([_raw(), _raw("partial", finish="length")])
    state = _turn(adapter).state
    snapshot = state.model_dump()
    with pytest.raises(CompactionFailureError) as error:
        _compact(adapter, state)
    assert error.value.usage.total_tokens == 120
    assert state.model_dump() == snapshot


@pytest.mark.parametrize(
    "status,body,error",
    [
        (400, {"code": "context_length_exceeded"}, ContextOverflowError),
        (429, {"message": "rate limited"}, TransientProviderError),
        (503, {"message": "unavailable"}, TransientProviderError),
    ],
)
def test_real_sdk_error_classification(runtime, status, body, error):
    adapter, _ = runtime([httpx.Response(status, json={"error": body})])
    with pytest.raises(error):
        _turn(adapter)


def test_registry_and_config_contract():
    cfg = _config()
    assert _validate_model_config_entry(cfg, 0, set()) == cfg
    for overrides in [
        {"reasoning_replay": "reasoning_content"},
        {"tool_calling": True},
        {"state": "manual_rolling"},
    ]:
        invalid = deepcopy(cfg)
        invalid["runtime"].update(overrides)
        with pytest.raises(ValueError):
            _validate_model_config_entry(invalid, 0, set())


def test_agent_rejects_bad_actions_without_replaying_their_reasoning(runtime, tmp_path):
    adapter, calls = runtime(
        [_raw("invalid action", data="REJECTED"), _raw(data="ACCEPTED")]
    )
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent._stateful_adapter = adapter
    agent._runtime_state = adapter.initial_state()
    agent._pending_turn_messages = [Message(role="user", content="newest frame")]
    agent._request_kwargs = _config()["request"]
    agent.MAX_RETRIES = 2
    agent.MAX_CONTEXT_LENGTH = 10000
    agent.ESTIMATED_CHARS_PER_TOKEN = 1.0
    agent.analysis_mode = False
    agent.token_counter = 0
    agent.conversation = []
    agent._summary_compactor = None
    agent._pending_compaction_trigger_tokens = None
    agent._compaction_counter = 0
    agent._pricing = {}
    agent.MODEL = "test"
    agent.step_counter = 0
    agent.run_dir = str(tmp_path)
    agent.run_record = RunRecord(
        run_id="test",
        game_id="test",
        agent_name="test",
        model="test",
        started_at=datetime.now(timezone.utc),
        run_dir=str(tmp_path),
        runtime={},
    )
    response, action, retries, _ = agent._request_with_retries([GameAction.ACTION1])
    assert action == GameAction.ACTION1 and retries == 1
    assert response.usage.total_tokens == agent.token_counter == 240
    assert calls[0]["messages"] == calls[1]["messages"]
    assert "REJECTED" not in agent._runtime_state.model_dump_json()
    assert "ACCEPTED" in agent._runtime_state.model_dump_json()
    for artifact in tmp_path.glob("*.json"):
        assert "REJECTED" not in artifact.read_text()
        assert "ACCEPTED" not in artifact.read_text()
