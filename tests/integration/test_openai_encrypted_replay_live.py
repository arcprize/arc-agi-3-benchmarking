from __future__ import annotations

import os
from copy import deepcopy

import pytest
from dotenv import load_dotenv

from benchmarking.agent import BenchmarkingAgent
from benchmarking.model_config import get_model_config
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_clients import build_model_runtime_client
from benchmarking.runtime_models import Message, ModelRequest

CONFIG_ID = "openai-gpt-5-6-sol-responses-reference-medium"
LIVE_TEST_ENV = "RUN_OPENAI_LIVE_TESTS"
MEMORY_TOKEN = "ARC-ENCRYPTED-REPLAY-7Q"


def _require_live_openai_key() -> None:
    if os.environ.get(LIVE_TEST_ENV) != "1":
        pytest.skip(f"Set {LIVE_TEST_ENV}=1 to run paid OpenAI smoke tests.")

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.fail(
            "OPENAI_API_KEY is required when RUN_OPENAI_LIVE_TESTS=1.",
            pytrace=False,
        )


@pytest.mark.integration
@pytest.mark.slow
def test_openai_encrypted_replay_two_turn_live() -> None:
    """Exercise stateless encrypted replay against the real Responses API."""
    _require_live_openai_key()
    config = deepcopy(get_model_config(CONFIG_ID))
    request_config = config["request"]

    # Keep the paid smoke test small while preserving the production state,
    # encrypted-reasoning, and automatic-compaction request fields.
    request_config["max_output_tokens"] = 4_096
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=CONFIG_ID,
    )
    client = client.with_options(timeout=90.0, max_retries=0)
    adapter = build_model_runtime_adapter(
        client=client,
        runtime_config=config["runtime"],
        config_id=CONFIG_ID,
    )

    system_message = Message(
        role="system",
        content="Follow the user's requested response format exactly.",
    )
    first_user_item = {
        "role": "user",
        "content": (
            "Calculate 97 multiplied by 89. "
            f"Also remember the token {MEMORY_TOKEN}. "
            "Reply in the format FIRST_OK:<product>."
        ),
    }
    first_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            request_config=request_config,
            input_items=[first_user_item],
        )
    )
    first_output_items = BenchmarkingAgent._serialize_encrypted_replay_output(
        first_response
    )

    reasoning_items = [
        item for item in first_output_items if item.get("type") == "reasoning"
    ]
    assert first_response.output_text.strip()
    assert "8633" in first_response.output_text
    assert first_response.usage.total_tokens > 0
    assert reasoning_items
    assert any(item.get("encrypted_content") for item in reasoning_items)

    second_user_item = {
        "role": "user",
        "content": "Reply with only the token I asked you to remember.",
    }
    second_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            request_config=request_config,
            input_items=[
                first_user_item,
                *first_output_items,
                second_user_item,
            ],
        )
    )
    second_output_items = BenchmarkingAgent._serialize_encrypted_replay_output(
        second_response
    )

    assert MEMORY_TOKEN in second_response.output_text
    assert second_response.usage.total_tokens > 0
    assert any(item.get("type") == "message" for item in second_output_items)
    assert request_config["store"] is False
    assert "previous_response_id" not in request_config
    assert request_config["context_management"] == [
        {"type": "compaction", "compact_threshold": 175_000}
    ]
