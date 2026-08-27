"""Opt-in paid tests for OpenAI encrypted replay and compaction."""

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

CONFIG_ID = "openai-gpt-5-6-sol-responses-continuous-conversation-low"
MEMORY_TOKEN = "ARC-ENCRYPTED-REPLAY-7Q"


def _require_paid_test(environment_variable: str) -> None:
    if os.environ.get(environment_variable) != "1":
        pytest.skip(f"Set {environment_variable}=1 to run this paid OpenAI test.")
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.fail(
            f"OPENAI_API_KEY is required when {environment_variable}=1.",
            pytrace=False,
        )


def _build_live_adapter(*, compact_threshold: int = 175_000):
    config = deepcopy(get_model_config(CONFIG_ID))
    config["request"]["max_output_tokens"] = 4_096
    config["request"]["context_management"] = [
        {"type": "compaction", "compact_threshold": compact_threshold}
    ]
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=CONFIG_ID,
    ).with_options(timeout=240.0, max_retries=0)
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
def test_openai_encrypted_replay_two_turn_live() -> None:
    _require_paid_test("RUN_OPENAI_LIVE_TESTS")
    adapter, request_config = _build_live_adapter()

    first = _turn(
        adapter,
        adapter.initial_state(),
        request_config,
        (
            "Calculate 97 multiplied by 89. Remember the token "
            f"{MEMORY_TOKEN}. Reply in the format FIRST_OK:<product>."
        ),
    )
    second = _turn(
        adapter,
        first.state,
        request_config,
        "Reply with only the token I asked you to remember.",
    )

    assert "8633" in first.response.output_text
    assert MEMORY_TOKEN in second.response.output_text
    assert first.response.usage.total_tokens > 0
    assert request_config["store"] is False
    assert "previous_response_id" not in request_config


@pytest.mark.integration
@pytest.mark.slow
def test_openai_encrypted_replay_compaction_end_to_end_live() -> None:
    _require_paid_test("RUN_OPENAI_COMPACTION_LIVE_TESTS")
    adapter, request_config = _build_live_adapter(compact_threshold=5_000)
    state = adapter.initial_state()

    first = _turn(
        adapter,
        state,
        request_config,
        (
            f"Remember {MEMORY_TOKEN}. Ignore this padding: "
            + " inert" * 5_500
            + ". Reply only FIRST_OK."
        ),
    )
    compacted = first
    for turn in range(2, 5):
        if compacted.transition.compaction_items_returned:
            break
        compacted = _turn(
            adapter,
            compacted.state,
            request_config,
            "Ignore more padding: " + " inert" * 2_000 + f". Reply only OK{turn}.",
        )

    assert compacted.transition.compaction_items_returned > 0
    assert compacted.state.payload["input_items"][0]["type"] == "compaction"
    final = _turn(
        adapter,
        compacted.state,
        request_config,
        "Reply with only the token from the first turn.",
    )
    assert MEMORY_TOKEN in final.response.output_text
