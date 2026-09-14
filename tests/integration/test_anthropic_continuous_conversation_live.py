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
SYSTEM_PROMPT = (
    "Follow the requested response format and remember explicit instructions."
)


def _require_paid_test(variable):
    if os.environ.get(variable) != "1":
        pytest.skip(f"Set {variable}=1 to run this paid Anthropic test.")
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.fail(f"ANTHROPIC_API_KEY is required when {variable}=1.", pytrace=False)


def _build_live_adapter():
    config = deepcopy(get_model_config(CONFIG_ID))
    config["request"]["max_tokens"] = 4096
    config["request"]["context_management"]["edits"][0]["trigger"]["value"] = 50_000
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=CONFIG_ID,
    ).with_options(timeout=240.0, max_retries=0)
    adapter = build_stateful_runtime_adapter(
        model_adapter=build_model_runtime_adapter(
            client=client,
            runtime_config=config["runtime"],
            config_id=CONFIG_ID,
        ),
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )
    return client, adapter, config["request"]


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
    client, adapter, config = _build_live_adapter()
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
    client, adapter, config = _build_live_adapter()
    with client:
        first = _turn(
            adapter,
            adapter.initial_state(),
            config,
            f"Remember the exact token {MEMORY_TOKEN} for the final check, including across summaries. Reply FIRST_OK.",
        )
        compacted = _turn(
            adapter,
            first.state,
            config,
            "Ignore this inert padding:"
            + " inert" * 60_000
            + ". Reply only SECOND_OK and keep remembering the earlier token.",
        )
        assert compacted.transition.compaction_items_returned > 0
        assert (
            compacted.state.payload["messages"][0]["content"][0]["type"] == "compaction"
        )
        final = _turn(
            adapter,
            compacted.state,
            config,
            "Reply only with the exact token I asked you to remember before the padding.",
        )
    assert MEMORY_TOKEN in final.response.output_text
    assert any(
        iteration["type"] == "compaction"
        for iteration in compacted.response.raw_response["usage"]["iterations"]
    )
    assert compacted.response.usage.total_tokens > 50_000
