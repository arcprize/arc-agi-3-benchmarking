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

OPUS_5_CONFIG_ID = "anthropic-opus-5-encrypted-replay-medium"
CONFIG_ID = OPUS_5_CONFIG_ID
LIVE_TEST_ENV = "RUN_ANTHROPIC_LIVE_TESTS"
COMPACTION_LIVE_TEST_ENV = "RUN_ANTHROPIC_COMPACTION_LIVE_TESTS"
MULTI_COMPACTION_LIVE_TEST_ENV = "RUN_ANTHROPIC_MULTI_COMPACTION_LIVE_TESTS"
MEMORY_TOKEN = "ARC-ANTHROPIC-REPLAY-7Q"
COMPACTION_MEMORY_TOKEN = "ARC-ANTHROPIC-COMPACTION-9X"
MULTI_COMPACTION_MEMORY_A = "ARC-MULTI-COMPACT-A7"
MULTI_COMPACTION_MEMORY_B = "ARC-MULTI-COMPACT-B9"
COMPACTION_THRESHOLD = 50_000
BELOW_THRESHOLD_TARGET = 45_000
ABOVE_THRESHOLD_TARGET = 55_000


def _require_live_anthropic_key(enabled_by: str = LIVE_TEST_ENV) -> None:
    if os.environ.get(enabled_by) != "1":
        pytest.skip(f"Set {enabled_by}=1 to run this paid Anthropic test.")

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.fail(
            f"ANTHROPIC_API_KEY is required when {enabled_by}=1.",
            pytrace=False,
        )


def _build_runtime(config: dict[str, Any]) -> tuple[Any, Any]:
    client = build_model_runtime_client(
        runtime_config=config["runtime"],
        client_config=config["client"],
        config_id=config["id"],
    ).with_options(timeout=600.0, max_retries=0)
    adapter = build_model_runtime_adapter(
        client=client,
        runtime_config=config["runtime"],
        config_id=config["id"],
    )
    return client, adapter


def _serialize_content(response: Any) -> list[dict[str, Any]]:
    return BenchmarkingAgent._serialize_anthropic_replay_content(response)


def _count_input_tokens(
    *,
    client: Any,
    request_config: dict[str, Any],
    system: str,
    messages: list[dict[str, Any]],
) -> int:
    result = client.beta.messages.count_tokens(
        model=request_config["model"],
        betas=request_config["betas"],
        system=system,
        messages=messages,
        thinking=request_config["thinking"],
        output_config=request_config["output_config"],
        context_management=request_config["context_management"],
    )
    return result.input_tokens


def _build_padded_user_message(
    *,
    client: Any,
    request_config: dict[str, Any],
    system: str,
    leading_messages: list[dict[str, Any]],
    prefix: str,
    suffix: str,
    target_total_tokens: int,
) -> tuple[dict[str, Any], int]:
    def message_with_repetitions(repetitions: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": f"{prefix}{' inert' * repetitions}{suffix}",
        }

    baseline = _count_input_tokens(
        client=client,
        request_config=request_config,
        system=system,
        messages=[*leading_messages, message_with_repetitions(0)],
    )
    probe_repetitions = 10_000
    probe = _count_input_tokens(
        client=client,
        request_config=request_config,
        system=system,
        messages=[
            *leading_messages,
            message_with_repetitions(probe_repetitions),
        ],
    )
    tokens_per_repetition = (probe - baseline) / probe_repetitions
    assert tokens_per_repetition > 0
    repetitions = max(
        0,
        round((target_total_tokens - baseline) / tokens_per_repetition),
    )

    for _ in range(5):
        candidate = message_with_repetitions(repetitions)
        measured = _count_input_tokens(
            client=client,
            request_config=request_config,
            system=system,
            messages=[*leading_messages, candidate],
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
def test_anthropic_encrypted_replay_two_turn_live() -> None:
    """Round-trip real summarized thinking and its encrypted signature."""
    _require_live_anthropic_key()
    config = deepcopy(get_model_config(CONFIG_ID))
    request_config = config["request"]
    _, adapter = _build_runtime(config)
    system_message = Message(
        role="system",
        content="Preserve explicit memory tokens and follow response formats exactly.",
    )
    first_user = {
        "role": "user",
        "content": (
            "Use the Chinese Remainder Theorem to find the smallest "
            "nonnegative integer x satisfying all five constraints: "
            "x mod 97 = 30; x mod 101 = 66; x mod 103 = 93; "
            "x mod 107 = 23; x mod 109 = 89. Carefully solve and verify "
            "every congruence using extended thinking as needed. "
            f"Remember the token {MEMORY_TOKEN}. "
            "Reply in the format FIRST_OK:<x>."
        ),
    }

    first_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[first_user],
            request_config=request_config,
        )
    )
    first_content = _serialize_content(first_response)
    thinking_blocks = [
        block for block in first_content if block.get("type") == "thinking"
    ]

    assert "987654321" in first_response.output_text
    assert first_response.reasoning_text
    assert thinking_blocks
    assert all(block.get("signature") for block in thinking_blocks)

    second_user = {
        "role": "user",
        "content": "Reply with only the memory token from the prior turn.",
    }
    second_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[
                first_user,
                {"role": "assistant", "content": first_content},
                second_user,
            ],
            request_config=request_config,
        )
    )

    assert MEMORY_TOKEN in second_response.output_text
    assert second_response.usage.total_tokens > 0
    assert request_config["context_management"] == {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 175_000},
                "pause_after_compaction": False,
            }
        ]
    }


@pytest.mark.integration
@pytest.mark.slow
def test_anthropic_encrypted_replay_compaction_end_to_end_live() -> None:
    """Cross the API minimum threshold, prune, and replay the real compaction."""
    _require_live_anthropic_key(COMPACTION_LIVE_TEST_ENV)
    config = deepcopy(get_model_config(CONFIG_ID))
    request_config = config["request"]
    request_config["max_tokens"] = 4_096
    request_config["context_management"] = {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {
                    "type": "input_tokens",
                    "value": COMPACTION_THRESHOLD,
                },
                "pause_after_compaction": False,
            }
        ]
    }
    client, adapter = _build_runtime(config)
    system_message = Message(
        role="system",
        content=(
            "Treat repeated 'inert' words as meaningless padding. Preserve "
            "explicit memory tokens and follow requested response formats."
        ),
    )
    system = system_message.content

    first_user, below_tokens = _build_padded_user_message(
        client=client,
        request_config=request_config,
        system=system,
        leading_messages=[],
        prefix=(
            f"The persistent memory token is {COMPACTION_MEMORY_TOKEN}. "
            "Remember it. Padding begins:"
        ),
        suffix=" Padding ends. Preserve the token and reply only BELOW_OK.",
        target_total_tokens=BELOW_THRESHOLD_TARGET,
    )
    assert below_tokens < COMPACTION_THRESHOLD
    first_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[first_user],
            request_config=request_config,
        )
    )
    first_content = _serialize_content(first_response)
    assert "BELOW_OK" in first_response.output_text
    assert not any(block.get("type") == "compaction" for block in first_content)

    first_history = [
        first_user,
        {"role": "assistant", "content": first_content},
    ]
    second_user, above_tokens = _build_padded_user_message(
        client=client,
        request_config=request_config,
        system=system,
        leading_messages=first_history,
        prefix="Add enough inert padding to cross the compaction threshold:",
        suffix=" Padding ends. Preserve important facts and reply only ABOVE_OK.",
        target_total_tokens=ABOVE_THRESHOLD_TARGET,
    )
    assert above_tokens > COMPACTION_THRESHOLD
    second_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[*first_history, second_user],
            request_config=request_config,
        )
    )
    second_content = _serialize_content(second_response)
    compaction_blocks = [
        block for block in second_content if block.get("type") == "compaction"
    ]

    assert "ABOVE_OK" in second_response.output_text
    assert getattr(second_response.raw_response, "stop_reason", None) != "compaction"
    assert compaction_blocks
    assert all(block.get("content") for block in compaction_blocks)
    assert all(
        "encrypted_content" not in block or block["encrypted_content"]
        for block in compaction_blocks
    )
    raw_usage = getattr(second_response.raw_response, "usage")
    iterations = getattr(raw_usage, "iterations", []) or []
    assert any(getattr(row, "type", None) == "compaction" for row in iterations)
    assert second_response.usage.input_tokens == sum(
        row.input_tokens for row in iterations
    )
    assert second_response.usage.output_tokens == sum(
        row.output_tokens for row in iterations
    )

    full_history = [
        *first_history,
        second_user,
        {"role": "assistant", "content": second_content},
    ]
    compacted_history = BenchmarkingAgent._prune_anthropic_replay_history(full_history)
    assert compacted_history[0]["role"] == "assistant"
    assert compacted_history[0]["content"][0]["type"] == "compaction"
    assert len(compacted_history) < len(full_history)

    final_user = {
        "role": "user",
        "content": "Reply with only the persistent memory token from turn one.",
    }
    final_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[*compacted_history, final_user],
            request_config=request_config,
        )
    )

    assert COMPACTION_MEMORY_TOKEN in final_response.output_text
    assert final_response.usage.total_tokens > 0


@pytest.mark.integration
@pytest.mark.slow
def test_anthropic_reasoning_survives_two_live_compaction_boundaries() -> None:
    """Replay signed reasoning into two compactions and retain both memories."""
    _require_live_anthropic_key(MULTI_COMPACTION_LIVE_TEST_ENV)
    config = deepcopy(get_model_config(OPUS_5_CONFIG_ID))
    request_config = config["request"]
    request_config["max_tokens"] = 8_192
    request_config["context_management"] = {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {
                    "type": "input_tokens",
                    "value": COMPACTION_THRESHOLD,
                },
                "pause_after_compaction": False,
            }
        ]
    }
    client, adapter = _build_runtime(config)
    system_message = Message(
        role="system",
        content=(
            "Treat repeated 'inert' words as meaningless padding. Preserve all "
            "explicit memory tokens and requested numeric results through every "
            "compaction. Follow response formats exactly."
        ),
    )
    system = system_message.content

    first_user = {
        "role": "user",
        "content": (
            "Use the Chinese Remainder Theorem to find the smallest nonnegative "
            "integer x satisfying x mod 97 = 30; x mod 101 = 66; "
            "x mod 103 = 93; x mod 107 = 23; x mod 109 = 89. "
            f"Remember {MULTI_COMPACTION_MEMORY_A}. Think carefully, verify the "
            "congruences, and reply in the format TURN1_OK:<x>."
        ),
    }
    first_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[first_user],
            request_config=request_config,
        )
    )
    first_content = _serialize_content(first_response)
    first_thinking = [
        block for block in first_content if block.get("type") == "thinking"
    ]
    assert "987654321" in first_response.output_text
    assert first_thinking
    assert all(block.get("signature") for block in first_thinking)

    first_history = [
        first_user,
        {"role": "assistant", "content": first_content},
    ]
    first_crossing_user, first_crossing_tokens = _build_padded_user_message(
        client=client,
        request_config=request_config,
        system=system,
        leading_messages=first_history,
        prefix="First compaction pressure padding begins:",
        suffix=" Padding ends. Preserve prior state and reply only TURN2_OK.",
        target_total_tokens=ABOVE_THRESHOLD_TARGET,
    )
    assert first_crossing_tokens > COMPACTION_THRESHOLD
    first_content_before_replay = deepcopy(first_content)
    first_compaction_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[*first_history, first_crossing_user],
            request_config=request_config,
        )
    )
    assert first_content == first_content_before_replay
    first_compaction_content = _serialize_content(first_compaction_response)
    first_compaction_blocks = [
        block
        for block in first_compaction_content
        if block.get("type") == "compaction" and block.get("content")
    ]
    assert "TURN2_OK" in first_compaction_response.output_text
    assert first_compaction_blocks
    assert getattr(first_compaction_response.raw_response, "stop_reason", None) != (
        "compaction"
    )

    history_through_first_compaction = [
        *first_history,
        first_crossing_user,
        {"role": "assistant", "content": first_compaction_content},
    ]
    first_compacted_history = BenchmarkingAgent._prune_anthropic_replay_history(
        history_through_first_compaction
    )
    assert first_compacted_history[0]["content"][0] == first_compaction_blocks[-1]
    assert not any(
        block.get("signature") in {row["signature"] for row in first_thinking}
        for message in first_compacted_history
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(block, dict)
    )

    second_value = 314_159_265
    second_moduli = (89, 97, 101, 103, 107)
    constraints = "; ".join(
        f"x mod {modulus} = {second_value % modulus}" for modulus in second_moduli
    )
    third_user = {
        "role": "user",
        "content": (
            "Use the Chinese Remainder Theorem again to find the smallest "
            f"nonnegative x satisfying {constraints}. Remember "
            f"{MULTI_COMPACTION_MEMORY_B}. Think carefully, verify every "
            "congruence, and reply in the format TURN3_OK:<x>."
        ),
    }
    third_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[*first_compacted_history, third_user],
            request_config=request_config,
        )
    )
    third_content = _serialize_content(third_response)
    third_thinking = [
        block for block in third_content if block.get("type") == "thinking"
    ]
    assert str(second_value) in third_response.output_text
    assert third_thinking
    assert all(block.get("signature") for block in third_thinking)

    history_before_second_compaction = [
        *first_compacted_history,
        third_user,
        {"role": "assistant", "content": third_content},
    ]
    second_crossing_user, second_crossing_tokens = _build_padded_user_message(
        client=client,
        request_config=request_config,
        system=system,
        leading_messages=history_before_second_compaction,
        prefix="Second compaction pressure padding begins:",
        suffix=" Padding ends. Preserve both memories and reply only TURN4_OK.",
        target_total_tokens=ABOVE_THRESHOLD_TARGET,
    )
    assert second_crossing_tokens > COMPACTION_THRESHOLD
    second_request_history = [
        *history_before_second_compaction,
        second_crossing_user,
    ]
    second_request_before_replay = deepcopy(second_request_history)
    second_compaction_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=second_request_history,
            request_config=request_config,
        )
    )
    assert second_request_history == second_request_before_replay
    second_compaction_content = _serialize_content(second_compaction_response)
    second_compaction_blocks = [
        block
        for block in second_compaction_content
        if block.get("type") == "compaction" and block.get("content")
    ]
    assert "TURN4_OK" in second_compaction_response.output_text
    assert second_compaction_blocks
    assert getattr(second_compaction_response.raw_response, "stop_reason", None) != (
        "compaction"
    )

    history_through_second_compaction = [
        *second_request_history,
        {"role": "assistant", "content": second_compaction_content},
    ]
    second_compacted_history = BenchmarkingAgent._prune_anthropic_replay_history(
        history_through_second_compaction
    )
    assert second_compacted_history[0]["content"][0] == second_compaction_blocks[-1]
    assert first_compaction_blocks[-1] not in [
        block
        for message in second_compacted_history
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(block, dict)
    ]

    final_user = {
        "role": "user",
        "content": (
            "Return both persistent memory tokens and both CRT answers from "
            "earlier turns. No explanation."
        ),
    }
    final_response = adapter.invoke(
        ModelRequest(
            messages=[system_message],
            native_messages=[*second_compacted_history, final_user],
            request_config=request_config,
        )
    )

    assert MULTI_COMPACTION_MEMORY_A in final_response.output_text
    assert MULTI_COMPACTION_MEMORY_B in final_response.output_text
    assert "987654321" in final_response.output_text
    assert str(second_value) in final_response.output_text
    assert final_response.usage.total_tokens > 0
