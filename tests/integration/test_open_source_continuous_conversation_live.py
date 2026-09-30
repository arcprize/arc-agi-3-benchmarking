"""Explicitly gated synthetic open-source replay and summary smoke test."""

import os
from copy import deepcopy

import pytest
from dotenv import load_dotenv

from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.model_config import get_model_config
from benchmarking.open_source_runtime import OPEN_SOURCE_ADAPTER_ID
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_clients import build_model_runtime_client
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import (
    build_stateful_runtime_adapter,
    resolve_adapter_id,
)
from benchmarking.runtime_state import ModelTurnRequest


@pytest.mark.integration
@pytest.mark.slow
def test_open_source_replay_and_compaction_live():
    if os.environ.get("RUN_OPEN_SOURCE_LIVE_TESTS") != "1":
        pytest.skip(
            "Set RUN_OPEN_SOURCE_LIVE_TESTS=1 to authorize paid synthetic calls."
        )
    load_dotenv()
    config_id = os.environ.get("OPEN_SOURCE_LIVE_CONFIG", "")
    if not config_id:
        pytest.fail(
            "OPEN_SOURCE_LIVE_CONFIG must name an installed open-source profile."
        )
    config = deepcopy(get_model_config(config_id))
    runtime = config["runtime"]
    assert resolve_adapter_id(runtime, config_id) == OPEN_SOURCE_ADAPTER_ID
    request_config = config["request"]
    output_key = (
        "max_tokens" if "max_tokens" in request_config else "max_completion_tokens"
    )
    request_config[output_key] = 4_096
    client = build_model_runtime_client(
        runtime_config=runtime, client_config=config["client"], config_id=config_id
    )
    transport = build_model_runtime_adapter(
        client=client, runtime_config=runtime, config_id=config_id
    )
    requests = []

    class CaptureTransport:
        def invoke(self, request):
            requests.append(request.model_copy(deep=True))
            return transport.invoke(request)

    adapter = build_stateful_runtime_adapter(
        model_adapter=CaptureTransport(), runtime_config=runtime, config_id=config_id
    )

    def turn(state, prompt):
        return adapter.invoke_turn(
            ModelTurnRequest(
                system_prompt="Follow the requested response format exactly.",
                new_messages=[Message(role="user", content=prompt)],
                request_config=request_config,
                previous_state=state,
            )
        )

    try:
        first = turn(
            adapter.initial_state(),
            "Remember the memory token ORCHID-739. Explain why remembering it is useful in your reasoning, then answer only READY.",
        )
        assert first.response.reasoning_text
        state = adapter.buffer_inputs(
            first.state,
            [
                Message(
                    role="user",
                    content="Pending observation: new frame marker CEDAR-528.",
                )
            ],
        )
        compacted = SummaryCompactor(
            SummaryCompactionPolicy(
                strategy="harness_summary",
                trigger_tokens=1,
                summary_max_output_tokens=4_096,
            )
        ).compact(
            adapter=adapter,
            state=state,
            request_config=request_config,
            trigger_tokens=1,
            max_context_length=config["agent"]["MAX_CONTEXT_LENGTH"],
            max_retries=0,
        )
        assert any(
            first.response.reasoning_text in message.get(field, "")
            for message in requests[1].native_input
            for field in ("content", "reasoning_content", "reasoning")
        )
        assert "CEDAR-528" not in str(requests[1].native_input)
        last = turn(
            compacted.state,
            "Return the remembered memory token and the new frame marker, with no other text.",
        )
        assert "ORCHID-739" in last.response.output_text
        assert "CEDAR-528" in last.response.output_text
        assert (
            requests[2].native_input[-2]["content"]
            == "Pending observation: new frame marker CEDAR-528."
        )
    finally:
        client.close()
