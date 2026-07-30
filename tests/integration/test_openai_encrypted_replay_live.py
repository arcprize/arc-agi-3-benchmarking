from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

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
COMPACTION_CONFIG_ID = "openai-gpt-5-6-sol-responses-reference-low"
COMPACTION_LIVE_TEST_ENV = "RUN_OPENAI_COMPACTION_LIVE_TESTS"
COMPACTION_MEMORY_TOKEN = "ARC-COMPACTION-MEMORY-9X"
COMPACTION_THRESHOLD = 175_000
BELOW_THRESHOLD_TARGET = 170_000
ABOVE_THRESHOLD_TARGET = 180_000


def _require_live_openai_key(enabled_by: str = LIVE_TEST_ENV) -> None:
    if os.environ.get(enabled_by) != "1":
        pytest.skip(f"Set {enabled_by}=1 to run this paid OpenAI test.")

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.fail(
            f"OPENAI_API_KEY is required when {enabled_by}=1.",
            pytrace=False,
        )


def _count_input_tokens(
    *,
    client: Any,
    request_config: dict[str, Any],
    instructions: str,
    input_items: list[dict[str, Any]],
) -> int:
    result = client.responses.input_tokens.count(
        model=request_config["model"],
        instructions=instructions,
        input=input_items,
        reasoning=request_config["reasoning"],
    )
    return result.input_tokens


def _build_padded_user_item(
    *,
    client: Any,
    request_config: dict[str, Any],
    instructions: str,
    leading_items: list[dict[str, Any]],
    prefix: str,
    suffix: str,
    target_total_tokens: int,
) -> tuple[dict[str, Any], int]:
    """Use the API counter to calibrate an input close to a token target."""

    def item_with_repetitions(repetitions: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": f"{prefix}{' inert' * repetitions}{suffix}",
        }

    baseline = _count_input_tokens(
        client=client,
        request_config=request_config,
        instructions=instructions,
        input_items=[*leading_items, item_with_repetitions(0)],
    )
    probe_repetitions = 1_000
    probe = _count_input_tokens(
        client=client,
        request_config=request_config,
        instructions=instructions,
        input_items=[
            *leading_items,
            item_with_repetitions(probe_repetitions),
        ],
    )
    tokens_per_repetition = (probe - baseline) / probe_repetitions
    assert tokens_per_repetition > 0
    repetitions = max(
        0,
        round((target_total_tokens - baseline) / tokens_per_repetition),
    )

    for _ in range(4):
        candidate = item_with_repetitions(repetitions)
        measured = _count_input_tokens(
            client=client,
            request_config=request_config,
            instructions=instructions,
            input_items=[*leading_items, candidate],
        )
        difference = target_total_tokens - measured
        if abs(difference) <= 100:
            return candidate, measured
        repetitions = max(
            0,
            repetitions + round(difference / tokens_per_repetition),
        )

    raise AssertionError(
        f"Could not calibrate input near {target_total_tokens} tokens; "
        f"last measurement was {measured}."
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


@pytest.mark.integration
@pytest.mark.slow
def test_openai_encrypted_replay_compaction_end_to_end_live() -> None:
    """Cross the production threshold and replay the encrypted compaction."""
    _require_live_openai_key(COMPACTION_LIVE_TEST_ENV)
    config = deepcopy(get_model_config(COMPACTION_CONFIG_ID))
    request_config = config["request"]
    request_config["max_output_tokens"] = 4_096
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=COMPACTION_CONFIG_ID,
    )
    client = client.with_options(timeout=240.0, max_retries=0)
    adapter = build_model_runtime_adapter(
        client=client,
        runtime_config=config["runtime"],
        config_id=COMPACTION_CONFIG_ID,
    )
    system_message = Message(
        role="system",
        content=(
            "Treat repeated 'inert' words as meaningless padding. Preserve "
            "explicit memory tokens and follow the requested response format."
        ),
    )
    instructions = system_message.content

    first_user_item, below_tokens = _build_padded_user_item(
        client=client,
        request_config=request_config,
        instructions=instructions,
        leading_items=[],
        prefix=(
            f"The persistent memory token is {COMPACTION_MEMORY_TOKEN}. "
            "Remember it for later turns. Padding begins:"
        ),
        suffix=(
            " Padding ends. Continue remembering the persistent token "
            f"{COMPACTION_MEMORY_TOKEN}. Reply only BELOW_OK."
        ),
        target_total_tokens=BELOW_THRESHOLD_TARGET,
    )
    assert below_tokens < COMPACTION_THRESHOLD
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
    assert "BELOW_OK" in first_response.output_text
    assert not any(item.get("type") == "compaction" for item in first_output_items)

    first_history = [first_user_item, *first_output_items]
    second_user_item, above_tokens = _build_padded_user_item(
        client=client,
        request_config=request_config,
        instructions=instructions,
        leading_items=first_history,
        prefix="Add enough inert padding to cross the compaction threshold:",
        suffix=(
            " Padding ends. Preserve all important prior facts and reply only ABOVE_OK."
        ),
        target_total_tokens=ABOVE_THRESHOLD_TARGET,
    )
    assert above_tokens > COMPACTION_THRESHOLD
    crossing_input = [*first_history, second_user_item]
    second_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            request_config=request_config,
            input_items=crossing_input,
        )
    )
    second_output_items = BenchmarkingAgent._serialize_encrypted_replay_output(
        second_response
    )
    compaction_items = [
        item for item in second_output_items if item.get("type") == "compaction"
    ]
    assert "ABOVE_OK" in second_response.output_text
    assert compaction_items
    assert all(item.get("encrypted_content") for item in compaction_items)

    full_history = [*crossing_input, *second_output_items]
    compacted_history = BenchmarkingAgent._prune_encrypted_replay_history(full_history)
    assert compacted_history[0]["type"] == "compaction"
    assert len(compacted_history) < len(full_history)

    final_user_item = {
        "role": "user",
        "content": "Reply with only the persistent memory token from turn one.",
    }
    post_compaction_input = [*compacted_history, final_user_item]
    post_compaction_tokens = _count_input_tokens(
        client=client,
        request_config=request_config,
        instructions=instructions,
        input_items=post_compaction_input,
    )
    assert post_compaction_tokens < above_tokens

    final_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            request_config=request_config,
            input_items=post_compaction_input,
        )
    )
    assert COMPACTION_MEMORY_TOKEN in final_response.output_text
    assert final_response.usage.total_tokens > 0
    assert request_config["store"] is False
    assert "previous_response_id" not in request_config
    assert request_config["context_management"] == [
        {"type": "compaction", "compact_threshold": COMPACTION_THRESHOLD}
    ]
