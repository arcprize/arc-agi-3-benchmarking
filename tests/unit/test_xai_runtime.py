import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from arcengine import GameAction
from openai import OpenAI

from benchmarking import model_config, runtime_clients
from benchmarking.agent import BenchmarkingAgent
from benchmarking.exceptions import EmptyResponseError
from benchmarking.recording import RunRecord
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_models import Message, ModelRequest
from benchmarking.runtime_registry import (
    build_stateful_runtime_adapter,
    resolve_adapter_id,
)
from benchmarking.runtime_state import ModelTurnRequest
from benchmarking.xai_runtime import (
    XAICompactionPolicy,
    XAIResponsesAdapter,
    normalize_xai_response,
    validate_continuous_conversation_request,
)

CONFIG_ID = "xai-grok-4-6-low-provider-adapter"
RUNTIME = {
    "sdk": "openai-python",
    "api": "responses",
    "adapter_id": "xai.responses.v1",
    "state": "continuous_conversation",
}
REQUEST = {
    "model": "grok-4.6",
    "store": False,
    "include": ["reasoning.encrypted_content"],
    "reasoning": {"effort": "low"},
}


def _usage(total=100):
    return {
        "input_tokens": total - 10,
        "output_tokens": 10,
        "total_tokens": total,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens_details": {"reasoning_tokens": 6},
    }


def _response(turn=1, text="ACTION1", total=100):
    return {
        "id": f"resp_{turn}",
        "object": "response",
        "created_at": 1,
        "model": "grok-4.6",
        "status": "completed",
        "usage": _usage(total),
        "output": [
            {
                "type": "reasoning",
                "id": f"rs_{turn}",
                "status": "completed",
                "encrypted_content": f"opaque-reasoning-{turn}",
                "summary": [{"type": "summary_text", "text": "visible summary"}],
            },
            {
                "type": "message",
                "id": f"msg_{turn}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
    }


def _compaction(turn=1):
    return {
        "id": f"cmp_{turn}",
        "object": "response.compaction",
        "created_at": 1,
        "model": "grok-4.6",
        "usage": {**_usage(40), "dropped_message_count": 3},
        "output": [
            {
                "type": "compaction",
                "id": f"cmp_{turn}",
                "encrypted_content": f"opaque-compaction-{turn}",
            }
        ],
    }


def _adapter(responses, *, compact=False):
    calls = []

    def handle(request):
        calls.append((request.url.path, json.loads(request.content)))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            return httpx.Response(response[0], json=response[1])
        return httpx.Response(200, json=response)

    client = OpenAI(
        api_key="test-key",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    runtime = deepcopy(RUNTIME)
    if compact:
        runtime["compaction"] = {"strategy": "native", "trigger_tokens": 50}
    low_level = build_model_runtime_adapter(
        client=client, runtime_config=runtime, config_id="test"
    )
    adapter = build_stateful_runtime_adapter(
        model_adapter=low_level, runtime_config=runtime, config_id="test"
    )
    return adapter, calls


def _turn(adapter, state=None, content="frame"):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="system",
            new_messages=[Message(role="user", content=content)],
            request_config=REQUEST,
            previous_state=state or adapter.initial_state(),
        )
    )


@pytest.mark.unit
class TestXAIReplay:
    def test_action_text_comes_from_the_replayed_output_not_a_helper_field(self):
        raw = _response(text="not an action")
        raw["output_text"] = "ACTION1"
        assert normalize_xai_response(raw).output_text == "not an action"

    def test_sdk_wire_requests_replay_every_returned_field(self):
        adapter, calls = _adapter([_response(), _response(2)])
        first = _turn(adapter, content="first")
        original = first.state.model_dump()
        second = _turn(adapter, first.state, "next")
        assert calls[0] == (
            "/v1/responses",
            {
                **REQUEST,
                "input": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "first"},
                ],
            },
        )
        assert calls[1][1]["input"] == [
            *calls[0][1]["input"],
            *_response()["output"],
            {"role": "user", "content": "next"},
        ]
        assert first.state.model_dump() == original
        assert second.response.reasoning_text == "visible summary"
        assert second.response.usage.reasoning_tokens == 6
        assert second.response.usage.cached_tokens == 5

    def test_compaction_excludes_fresh_and_buffered_observations(self):
        adapter, calls = _adapter(
            [_response(), _compaction(), _response(2)], compact=True
        )
        first = _turn(adapter)
        buffered = adapter.buffer_inputs(
            first.state, [Message(role="user", content="reset")]
        )
        original = buffered.model_dump()
        result = _turn(adapter, buffered, "new frame")
        assert calls[1] == (
            "/v1/responses/compact",
            {
                "model": "grok-4.6",
                "input": first.state.payload["input_items"],
            },
        )
        assert calls[2][1]["input"] == [
            *_compaction()["output"],
            {"role": "user", "content": "reset"},
            {"role": "user", "content": "new frame"},
        ]
        assert result.state.payload["input_items"] == [
            *calls[2][1]["input"],
            *_response(2)["output"],
        ]
        assert buffered.model_dump() == original
        assert result.response.usage.total_tokens == 140
        assert result.state.payload["context_tokens"] == 100
        assert result.transition.compaction_items_returned == 1
        assert result.action_state["native_compaction"]["usage"]["total_tokens"] == 40
        assert len(result.state.accepted_turns) == 1

    def test_repeated_compaction_replays_previous_blob(self):
        adapter, calls = _adapter(
            [
                _response(),
                _compaction(),
                _response(2),
                _compaction(2),
                _response(3),
            ],
            compact=True,
        )
        first = _turn(adapter)
        second = _turn(adapter, first.state, "second")
        third = _turn(adapter, second.state, "third")
        assert calls[3][1]["input"] == second.state.payload["input_items"]
        assert calls[3][1]["input"][0] == _compaction()["output"][0]
        assert third.state.payload["input_items"][0] == _compaction(2)["output"][0]
        assert "opaque-compaction-1" not in json.dumps(third.state.payload)

    def test_compaction_cost_does_not_retrigger_compaction(self):
        adapter, calls = _adapter(
            [
                _response(),
                _compaction(),
                _response(2, total=20),
                _response(3),
            ],
            compact=True,
        )
        first = _turn(adapter)
        second = _turn(adapter, first.state)
        _turn(adapter, second.state)
        assert second.response.usage.total_tokens == 60
        assert second.state.payload["context_tokens"] == 20
        assert [path for path, _ in calls].count("/v1/responses/compact") == 1

    def test_buffered_first_observation_is_not_compacted(self):
        adapter, calls = _adapter([_response()], compact=True)
        state = adapter.buffer_inputs(
            adapter.initial_state(), [Message(role="user", content="reset")]
        )
        _turn(adapter, state)
        assert calls[0][1]["input"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "reset"},
            {"role": "user", "content": "frame"},
        ]

    def test_readable_artifacts_never_include_opaque_state_or_old_frames(self):
        adapter, _ = _adapter([_response(), _compaction(), _response(2)], compact=True)
        first = _turn(adapter, content="old frame")
        result = _turn(adapter, first.state, "current frame")
        artifact = json.dumps(
            {
                "request": result.sanitized_request,
                "transition": result.transition.model_dump(),
                "action": result.action_state,
                "messages": result.readable_request_messages,
            }
        )
        assert "opaque-" not in artifact
        assert "old frame" not in artifact
        assert "current frame" in artifact

    @pytest.mark.parametrize("field", ["model", "system_prompt"])
    def test_model_and_system_cannot_change(self, field):
        adapter, calls = _adapter([_response()])
        first = _turn(adapter)
        state = first.state.model_copy(deep=True)
        state.payload[field] = "different"
        with pytest.raises(ValueError, match="cannot change"):
            _turn(adapter, state)
        assert len(calls) == 1

    def test_no_harness_summary_fallback(self):
        adapter, _ = _adapter([])
        with pytest.raises(ValueError, match="native compaction"):
            adapter.rebuild_after_compaction(
                Message(role="user", content="summary"), []
            )


@pytest.mark.unit
class TestXAIFailures:
    def test_response_error_diagnostics_do_not_echo_provider_error_messages(self):
        raw = _response()
        raw["error"] = {"code": "invalid_request", "message": "opaque-secret"}
        adapter, _ = _adapter([raw])
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter)
        assert "opaque-secret" not in json.dumps(error.value.response)

    def test_malformed_sdk_output_does_not_leak_through_serialization_warnings(
        self, recwarn
    ):
        raw = _response()
        raw["output"].append({"type": "unknown", "encrypted_content": "opaque-secret"})
        adapter, _ = _adapter([raw])
        with pytest.raises(EmptyResponseError):
            _turn(adapter)
        assert not recwarn.list

    @pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", None])
    def test_valid_looking_action_requires_completed_status(self, status):
        raw = _response()
        raw["status"] = status
        adapter, _ = _adapter([raw])
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter)
        assert error.value.usage.total_tokens == 100
        assert "opaque-reasoning" not in json.dumps(error.value.response)

    @pytest.mark.parametrize(
        "mutation",
        [
            "missing_blob",
            "blank_blob",
            "unfinished_item",
            "refusal",
            "tool",
            "thought_only",
            "empty",
            "unknown",
            "wrong_role",
            "error",
        ],
    )
    def test_rejects_unusable_native_output_with_usage(self, mutation):
        raw = _response()
        if mutation == "missing_blob":
            del raw["output"][0]["encrypted_content"]
        elif mutation == "blank_blob":
            raw["output"][0]["encrypted_content"] = " "
        elif mutation == "unfinished_item":
            raw["output"][1]["status"] = "incomplete"
        elif mutation == "refusal":
            raw["output"][1]["content"] = [{"type": "refusal", "refusal": "No"}]
        elif mutation == "tool":
            raw["output"].append({"type": "function_call", "name": "ACTION1"})
        elif mutation == "thought_only":
            raw["output"].pop()
        elif mutation == "empty":
            raw["output"][1]["content"][0]["text"] = " "
        elif mutation == "unknown":
            raw["output"].append({"type": "unknown"})
        elif mutation == "wrong_role":
            raw["output"][1]["role"] = "user"
        else:
            raw["error"] = {"code": "failed"}
        adapter, _ = _adapter([raw])
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter)
        assert error.value.usage.total_tokens == 100

    @pytest.mark.parametrize(
        "output",
        [
            [],
            [{}],
            ["bad"],
            [{"type": "compaction", "encrypted_content": ""}],
            _response()["output"],
            [*_compaction()["output"], *_compaction(2)["output"]],
        ],
    )
    def test_bad_compaction_is_atomic_and_billed(self, output):
        compacted = _compaction()
        compacted["output"] = output
        adapter, calls = _adapter([_response(), compacted], compact=True)
        first = _turn(adapter)
        original = first.state.model_dump()
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter, first.state)
        assert error.value.usage.total_tokens == 40
        assert first.state.model_dump() == original
        assert len(calls) == 2

    @pytest.mark.parametrize(
        "failure",
        [
            (429, {"error": {"message": "opaque-secret", "type": "rate_limit"}}),
            (400, {"error": {"message": "context_length_exceeded opaque-secret"}}),
            httpx.ReadTimeout("opaque-secret"),
        ],
    )
    def test_action_transport_failure_keeps_compaction_usage_and_redacts_errors(
        self, failure
    ):
        adapter, _ = _adapter([_response(), _compaction(), failure], compact=True)
        first = _turn(adapter)
        original = first.state.model_dump()
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter, first.state)
        assert error.value.usage.total_tokens == 40
        assert first.state.model_dump() == original
        assert "opaque-secret" not in str(error.value)
        assert "opaque-secret" not in json.dumps(error.value.response)

    def test_failed_action_bills_both_calls_and_retry_reuses_accepted_history(self):
        failed = _response(2)
        failed["status"] = "incomplete"
        adapter, calls = _adapter(
            [
                _response(),
                _compaction(),
                failed,
                _compaction(2),
                _response(3),
            ],
            compact=True,
        )
        first = _turn(adapter)
        with pytest.raises(EmptyResponseError) as error:
            _turn(adapter, first.state)
        assert error.value.usage.total_tokens == 140
        accepted = _turn(adapter, first.state)
        assert calls[1] == calls[3]
        assert "opaque-reasoning-2" not in json.dumps(accepted.state.payload)


@pytest.mark.unit
class TestXAIConfiguration:
    @pytest.mark.parametrize("explicit_strategy", [True, False])
    def test_agent_records_native_policy_without_creating_summary_compactor(
        self, monkeypatch, tmp_path, explicit_strategy
    ):
        config = deepcopy(model_config.get_model_config(CONFIG_ID))
        if not explicit_strategy:
            config["runtime"]["compaction"].pop("strategy")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("benchmarking.agent.get_model_config", lambda _: config)
        monkeypatch.setattr(
            "benchmarking.agent.build_model_runtime_client", lambda **_: object()
        )
        agent = BenchmarkingAgent(
            card_id="card-id",
            game_id="game-id",
            agent_name="test",
            ROOT_URL="https://arcprize.org",
            record=False,
            arc_env=SimpleNamespace(info=SimpleNamespace(baseline_actions=[])),
            config=CONFIG_ID,
        )
        assert agent._summary_compactor is None
        assert agent.run_record.runtime["adapter_id"] == "xai.responses.v1"
        assert agent.run_record.runtime["compaction"] == {
            "strategy": "native",
            "trigger_tokens": 200_000,
            "context_limit_tokens": 500_000,
        }

    @pytest.mark.parametrize(
        "update",
        [
            {"store": True},
            {"store": None},
            {"stream": True},
            {"background": True},
            {"previous_response_id": None},
            {"conversation": "id"},
            {"include": []},
            {"reasoning": {"context": "auto"}},
            {"context_management": []},
            {"compact_threshold": 100},
            {"extra_body": {"store": True}},
            {"extra_query": {"store": True}},
            {"input": []},
            {"instructions": "override"},
            {"tools": []},
            {"tool_choice": "none"},
            {"truncation": "auto"},
        ],
    )
    def test_rejects_conflicting_settings_before_io(self, update):
        request = {**REQUEST, **update}
        with pytest.raises(ValueError):
            validate_continuous_conversation_request(request)
        with pytest.raises(ValueError):
            XAIResponsesAdapter(None).invoke(
                ModelRequest(messages=[], request_config=request)
            )

    @pytest.mark.parametrize(
        "update",
        [
            {"state": "manual_rolling"},
            {"state": "previous_response_id"},
            {"api": "chat_completions"},
            {"sdk": "google-genai"},
        ],
    )
    def test_rejects_mismatched_registration(self, update):
        with pytest.raises(ValueError, match="requires"):
            resolve_adapter_id({**RUNTIME, **update}, "test")

    def test_checked_in_profile_and_native_policy(self):
        config = model_config.get_model_config(CONFIG_ID)
        assert config["request"]["model"] == "grok-4.6"
        assert config["request"]["reasoning"] == {"effort": "low"}
        assert config["runtime"]["compaction"] == {
            "strategy": "native",
            "trigger_tokens": 200_000,
        }
        assert config["agent"]["MAX_CONTEXT_LENGTH"] == 500_000
        assert resolve_adapter_id(config["runtime"], CONFIG_ID) == "xai.responses.v1"

    @pytest.mark.parametrize(
        "compaction",
        [
            {"strategy": "harness_summary"},
            {"trigger_tokens": True},
            {"trigger_tokens": 0},
            {"trigger_tokens": "10"},
            {"unknown": 1},
        ],
    )
    def test_rejects_invalid_native_compaction_policy(self, compaction):
        with pytest.raises(ValueError):
            XAICompactionPolicy.model_validate(compaction)

    def test_config_rejects_trigger_without_output_headroom(self):
        config = deepcopy(model_config.get_model_config(CONFIG_ID))
        config["runtime"]["compaction"]["trigger_tokens"] = 400_000
        with pytest.raises(ValueError, match="context limit"):
            model_config._validate_model_config_entry(config, 1, set())

    def test_xai_client_defaults_never_use_openrouter(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        monkeypatch.setattr(runtime_clients, "OpenAIClient", lambda **kwargs: kwargs)
        client = runtime_clients.build_model_runtime_client(
            runtime_config=RUNTIME, client_config={}, config_id="test"
        )
        assert client == {"base_url": "https://api.x.ai/v1", "api_key": "xai-test-key"}

    def test_provider_reported_cost_survives_sdk_serialization(self):
        raw = _response()
        raw["usage"].update(cost=0.02, cost_details={"inference": 0.02})
        adapter, _ = _adapter([raw])
        result = _turn(adapter)
        assert result.response.usage.cost == 0.02
        assert normalize_xai_response(raw).usage.cost_details == {"inference": 0.02}


def _agent(adapter, state=None):
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    agent._stateful_adapter = adapter
    agent._runtime_state = state or adapter.initial_state()
    agent._pending_turn_messages = [Message(role="user", content="frame")]
    agent._request_kwargs = REQUEST
    agent._continuous_conversation = True
    agent._summary_compactor = None
    agent._server_state = False
    agent.MAX_CONTEXT_LENGTH = 500_000
    agent.ESTIMATED_CHARS_PER_TOKEN = 1.0
    agent.MAX_RETRIES = 1
    agent.analysis_mode = False
    agent.token_counter = 0
    agent.conversation = []
    agent._build_system_prompt = lambda: "system"
    agent.diagnostics = []
    agent._save_diagnostic = agent.diagnostics.append
    return agent


@pytest.mark.unit
class TestXAIAgentIntegration:
    def test_only_parseable_actions_commit_compacted_state(self):
        adapter, calls = _adapter(
            [
                _response(),
                _compaction(),
                _response(2, text="not an action"),
                _compaction(2),
                _response(3),
            ],
            compact=True,
        )
        first = _turn(adapter)
        agent = _agent(adapter, first.state)
        response, action, retries, _ = agent._request_with_retries([GameAction.ACTION1])
        assert action == GameAction.ACTION1
        assert retries == 1
        assert response.usage.total_tokens == 280
        assert agent.token_counter == 280
        assert calls[1] == calls[3]
        assert "opaque-reasoning-2" not in json.dumps(agent._runtime_state.payload)
        assert "opaque-reasoning-3" in json.dumps(agent._runtime_state.payload)

    def test_incomplete_response_never_becomes_an_action(self):
        failed = _response()
        failed["status"] = "incomplete"
        adapter, calls = _adapter([failed, _response(2)])
        agent = _agent(adapter)
        response, _, retries, _ = agent._request_with_retries([GameAction.ACTION1])
        assert retries == 1
        assert calls[0] == calls[1]
        assert response.usage.total_tokens == 200
        assert "opaque-reasoning-1" not in json.dumps(agent._runtime_state.payload)
        assert "opaque-" not in json.dumps(agent.diagnostics)

    def test_exhausted_retries_persist_billed_usage_without_advancing_state(self):
        adapter, _ = _adapter(
            [_response(text="not an action"), _response(2, text="no action")]
        )
        agent = _agent(adapter)
        initial = agent._runtime_state.model_dump()
        agent.run_record = RunRecord(
            run_id="test",
            game_id="test",
            agent_name="test",
            model="grok-4.6",
            started_at=datetime.now(timezone.utc),
            run_dir="test",
        )
        saved = []
        agent._write_run_meta = lambda: saved.append(agent.run_record.model_dump())
        with pytest.raises(RuntimeError, match="valid action"):
            agent._request_with_retries([GameAction.ACTION1])
        assert agent.run_record.total_usage.total_tokens == 200
        assert len(saved) == 1
        assert agent._runtime_state.model_dump() == initial
