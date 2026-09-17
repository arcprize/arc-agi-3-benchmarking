"""Explicitly opt-in paid Opus 5 low replay and native compaction tests."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
from dotenv import load_dotenv

from benchmarking.model_config import get_model_config
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_clients import build_model_runtime_client
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import build_stateful_runtime_adapter
from benchmarking.runtime_state import ModelTurnRequest

CONFIG_ID = "anthropic-opus-5-low-provider-adapter"
MEMORY_TOKEN = "ARC-ANTHROPIC-CONTINUITY-7Q"
SECOND_MEMORY_TOKEN = "ARC-ANTHROPIC-CONTINUITY-9R"
SYSTEM_PROMPT = (
    "Follow the requested response format and remember explicit instructions."
)


def _require_paid_test(variable):
    if os.environ.get(variable) != "1":
        pytest.skip(f"Set {variable}=1 to run this paid Anthropic test.")
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.fail(f"ANTHROPIC_API_KEY is required when {variable}=1.", pytrace=False)


class _CapturingAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.requests = []

    def invoke(self, request):
        self.requests.append(deepcopy(request))
        return self.adapter.invoke(request)


def _build_live_adapter(*, trigger_tokens=175_000, max_output_tokens=4096):
    config = deepcopy(get_model_config(CONFIG_ID))
    config["request"]["max_tokens"] = max_output_tokens
    config["runtime"]["compaction"]["trigger_tokens"] = trigger_tokens
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=CONFIG_ID,
    ).with_options(timeout=240.0, max_retries=0)
    captured = _CapturingAdapter(
        build_model_runtime_adapter(
            client=client,
            runtime_config=config["runtime"],
            config_id=CONFIG_ID,
        )
    )
    adapter = build_stateful_runtime_adapter(
        model_adapter=captured,
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )
    return client, adapter, config["request"], captured.requests


def _turn(adapter, state, config, content):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt=SYSTEM_PROMPT,
            new_messages=[Message(role="user", content=content)],
            request_config=config,
            previous_state=state,
            max_context_length=1_000_000,
        )
    )


@pytest.mark.integration
@pytest.mark.slow
def test_anthropic_continuous_conversation_two_turn_live():
    _require_paid_test("RUN_ANTHROPIC_LIVE_TESTS")
    client, adapter, config, _ = _build_live_adapter()
    with client:
        first = _turn(
            adapter,
            adapter.initial_state(),
            config,
            f"Remember {MEMORY_TOKEN} for later. Calculate 97 multiplied by 89 and reply with the product.",
        )
        second = _turn(
            adapter,
            first.state,
            config,
            "Reply only with the exact token I asked you to remember.",
        )
    assert "8633" in first.response.output_text
    assert MEMORY_TOKEN in second.response.output_text
    assert first.response.usage.total_tokens > 0
    assert config["output_config"]["effort"] == "low"
    assert any(
        block["type"] in {"thinking", "redacted_thinking"}
        for block in first.response.raw_response["content"]
    )


@pytest.mark.integration
@pytest.mark.slow
def test_anthropic_continuous_conversation_compaction_live():
    _require_paid_test("RUN_ANTHROPIC_COMPACTION_LIVE_TESTS")
    client, adapter, config, requests = _build_live_adapter(trigger_tokens=1)
    with client:
        first = _turn(
            adapter,
            adapter.initial_state(),
            config,
            f"Remember the exact token {MEMORY_TOKEN} for the final check, including across summaries. Reply FIRST_OK.",
        )
        fresh_observation = "Fresh observation marker: NOT_IN_SUMMARY_4R. Reply only with the exact token you remembered earlier."
        compacted = _turn(
            adapter,
            first.state,
            config,
            fresh_observation,
        )
    assert MEMORY_TOKEN in compacted.response.output_text
    assert compacted.transition.compaction_items_returned == 1
    block = compacted.state.payload["messages"][0]["content"][0]
    assert (
        block["type"] == "compaction"
        and block["content"].strip()
        and block["signature"]
    )
    assert requests[1].native_input == first.state.payload["messages"]
    assert "NOT_IN_SUMMARY_4R" not in str(requests[1].native_input)
    assert requests[2].native_input[-1] == {
        "role": "user",
        "content": fresh_observation,
    }
    assert "compaction" not in requests[2].request_config
    assert compacted.action_state["native_compaction"]["usage"]["total_tokens"] > 0


@pytest.mark.integration
@pytest.mark.slow
def test_anthropic_continuous_conversation_two_compactions_live():
    _require_paid_test("RUN_ANTHROPIC_MULTI_COMPACTION_LIVE_TESTS")
    client, adapter, config, requests = _build_live_adapter(
        trigger_tokens=1, max_output_tokens=8192
    )
    state = adapter.initial_state()
    memories = [(MEMORY_TOKEN, 987_654_321), (SECOND_MEMORY_TOKEN, 314_159_265)]
    with client:
        for index, (memory, answer) in enumerate(memories):
            constraints = "; ".join(
                f"x mod {modulus} = {answer % modulus}"
                for modulus in (97, 101, 103, 107, 109)
            )
            snapshot = state.model_dump_json()
            solved = _turn(
                adapter,
                state,
                config,
                "Find the smallest nonnegative integer x satisfying "
                f"{constraints}. Think carefully and verify every congruence. "
                f"Remember the token {memory} and the answer, preserving any "
                "previously remembered tokens and results across summaries. "
                "Reply with the numeric answer.",
            )
            assert state.model_dump_json() == snapshot
            assert str(answer) in solved.response.output_text
            thinking = [
                block
                for block in solved.response.raw_response["content"]
                if block["type"] == "thinking"
            ]
            assert thinking and all(block.get("signature") for block in thinking)
            assert solved.transition.compaction_items_returned == index
            if index:
                assert requests[-2].native_input == state.payload["messages"]
                assert SECOND_MEMORY_TOKEN not in str(requests[-2].native_input)
            state = solved.state
        final = _turn(
            adapter,
            state,
            config,
            "Reply with both exact remembered tokens and their associated numeric answers.",
        )
    assert final.transition.compaction_items_returned == 1
    summaries = [
        request for request in requests if "compaction" in request.request_config
    ]
    assert len(summaries) == 2
    assert summaries[-1].native_input == state.payload["messages"]
    assert all(
        "context_management" not in request.request_config for request in requests
    )
    head = final.state.payload["messages"][0]["content"][0]
    assert head["type"] == "compaction" and head["signature"]
    for memory, answer in memories:
        assert memory in final.response.output_text
        assert str(answer) in final.response.output_text
    assert config["model"] == "claude-opus-5"
    assert config["output_config"]["effort"] == "low"
