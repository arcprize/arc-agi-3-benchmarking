"""Explicitly gated, paid xAI replay and native compaction checks."""

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

CONFIG_ID = "xai-grok-4-6-xhigh-provider-adapter"
MEMORY_TOKEN = "ARC-XAI-REPLAY-7Q"


def _live_adapter(gate: str, *, compact: bool):
    if os.environ.get(gate) != "1":
        pytest.skip(f"Set {gate}=1 to run this paid xAI test.")
    load_dotenv()
    if not os.environ.get("XAI_API_KEY", "").strip():
        pytest.fail(f"XAI_API_KEY is required when {gate}=1.", pytrace=False)
    config = deepcopy(get_model_config(CONFIG_ID))
    config["request"]["max_output_tokens"] = 4_096
    if compact:
        config["runtime"]["compaction"]["trigger_tokens"] = 1
    else:
        config["runtime"].pop("compaction")
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
    adapter = build_stateful_runtime_adapter(
        model_adapter=low_level,
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )
    return client, adapter, config["request"]


def _turn(adapter, state, config, content):
    return adapter.invoke_turn(
        ModelTurnRequest(
            system_prompt="Follow the requested format exactly and remember prior facts.",
            new_messages=[Message(role="user", content=content)],
            request_config=config,
            previous_state=state,
        )
    )


@pytest.mark.integration
@pytest.mark.slow
def test_xai_two_turn_native_replay_live():
    client, adapter, config = _live_adapter("RUN_XAI_LIVE_TESTS", compact=False)
    try:
        first = _turn(
            adapter,
            adapter.initial_state(),
            config,
            f"Calculate 97 times 89. Remember {MEMORY_TOKEN}. Reply with the product.",
        )
        assert "8633" in first.response.output_text
        assert any(
            item.get("encrypted_content") for item in first.state.payload["input_items"]
        )
        second = _turn(
            adapter,
            first.state,
            config,
            "Reply with the token and product from our previous turn.",
        )
        assert MEMORY_TOKEN in second.response.output_text
        assert "8633" in second.response.output_text
        assert second.response.usage.total_tokens > 0
    finally:
        client.close()


@pytest.mark.integration
@pytest.mark.slow
def test_xai_repeated_native_compaction_live():
    client, adapter, config = _live_adapter(
        "RUN_XAI_COMPACTION_LIVE_TESTS", compact=True
    )
    try:
        first = _turn(
            adapter,
            adapter.initial_state(),
            config,
            f"Calculate 97 times 89. Remember {MEMORY_TOKEN}. Reply with the product.",
        )
        assert "8633" in first.response.output_text
        assert any(
            item.get("encrypted_content") for item in first.state.payload["input_items"]
        )
        state = first.state
        for _ in range(2):
            result = _turn(
                adapter,
                state,
                config,
                "Reply with the token and product from the start of this conversation.",
            )
            assert result.transition.compaction_items_returned == 1
            assert result.state.payload["input_items"][0]["type"] == "compaction"
            assert MEMORY_TOKEN in result.response.output_text
            assert "8633" in result.response.output_text
            assert result.action_state["native_compaction"]["usage"]["total_tokens"] > 0
            state = result.state
    finally:
        client.close()
