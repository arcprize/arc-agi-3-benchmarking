"""Opt-in paid tests for Google continuous conversation and compaction."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
from dotenv import load_dotenv

from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.model_config import get_model_config
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_clients import build_model_runtime_client
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import build_stateful_runtime_adapter
from benchmarking.runtime_state import ModelTurnRequest

CONFIG_ID = "google-gemini-3-8-flash-low-provider-adapter"
MEMORY_TOKEN = "GEMINI-CONTINUOUS-CONVERSATION-7Q"


def _require_paid_test(environment_variable: str) -> None:
    if os.environ.get(environment_variable) != "1":
        pytest.skip(f"Set {environment_variable}=1 to run this paid Google test.")
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        pytest.fail(
            f"GOOGLE_API_KEY is required when {environment_variable}=1.",
            pytrace=False,
        )


def _build_live_adapter():
    config = deepcopy(get_model_config(CONFIG_ID))
    config["request"]["generation_config"]["max_output_tokens"] = 4_096
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=CONFIG_ID,
    )
    low_level = build_model_runtime_adapter(
        client=client,
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )
    stateful = build_stateful_runtime_adapter(
        model_adapter=low_level,
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )
    return stateful, config["request"]


def _turn(adapter, state, request_config, content):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="Follow the requested response format exactly.",
            new_messages=[Message(role="user", content=content)],
            request_config=request_config,
            previous_state=state,
        )
    )


@pytest.mark.integration
@pytest.mark.slow
def test_google_continuous_conversation_two_turn_live() -> None:
    _require_paid_test("RUN_GOOGLE_LIVE_TESTS")
    adapter, request_config = _build_live_adapter()

    first = _turn(
        adapter,
        adapter.initial_state(),
        request_config,
        f"Remember the token {MEMORY_TOKEN}. Reply only FIRST_OK.",
    )
    second = _turn(
        adapter,
        first.state,
        request_config,
        "Reply with only the token I asked you to remember.",
    )

    assert MEMORY_TOKEN in second.response.output_text
    assert first.response.usage.total_tokens > 0
    assert any(
        step.get("type") == "thought" and step.get("signature")
        for step in first.state.payload["steps"]
    )
    assert request_config["store"] is False
    assert "previous_interaction_id" not in request_config


@pytest.mark.integration
@pytest.mark.slow
def test_google_harness_summary_compaction_live() -> None:
    _require_paid_test("RUN_GOOGLE_COMPACTION_LIVE_TESTS")
    adapter, request_config = _build_live_adapter()
    first = _turn(
        adapter,
        adapter.initial_state(),
        request_config,
        f"Remember the token {MEMORY_TOKEN}. Reply only FIRST_OK.",
    )
    compactor = SummaryCompactor(
        SummaryCompactionPolicy(
            strategy="harness_summary",
            trigger_tokens=1,
            summary_max_output_tokens=1_024,
        )
    )

    assert compactor.should_compact(first.response.usage)
    compacted = compactor.compact(
        adapter=adapter,
        state=first.state,
        request_config=request_config,
        trigger_tokens=first.response.usage.total_tokens,
        max_context_length=1_048_576,
        max_retries=1,
    )
    final = _turn(
        adapter,
        compacted.state,
        request_config,
        "Reply with only the token from the compacted history.",
    )

    assert MEMORY_TOKEN in compacted.summary
    assert MEMORY_TOKEN in final.response.output_text
    assert compacted.opaque_continuity_preserved is False
