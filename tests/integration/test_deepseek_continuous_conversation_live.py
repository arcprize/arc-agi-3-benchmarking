"""Explicitly gated synthetic DeepSeek replay and summary smoke test."""

import os
from copy import deepcopy

import pytest
from dotenv import load_dotenv

from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.deepseek_runtime import DEEPSEEK_ADAPTER_ID
from benchmarking.model_config import get_model_config
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
def test_deepseek_replay_and_compaction_live():
    if os.environ.get("RUN_DEEPSEEK_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_DEEPSEEK_LIVE_TESTS=1 to authorize paid synthetic calls.")
    load_dotenv()
    config_id = "deepseek-v4-1-flash-low-provider-adapter"
    config = deepcopy(get_model_config(config_id))
    runtime = config["runtime"]
    assert resolve_adapter_id(runtime, config_id) == DEEPSEEK_ADAPTER_ID
    request_config = config["request"]
    request_config["max_tokens"] = 4_096
    client = build_model_runtime_client(
        runtime_config=runtime,
        client_config=config["client"],
        config_id=config_id,
    )
    transport = build_model_runtime_adapter(
        client=client,
        runtime_config=runtime,
        config_id=config_id,
    )
    requests = []

    class CaptureTransport:
        def invoke(self, request):
            requests.append(request.model_copy(deep=True))
            return transport.invoke(request)

    adapter = build_stateful_runtime_adapter(
        model_adapter=CaptureTransport(),
        runtime_config=runtime,
        config_id=config_id,
    )

    def turn(state, prompt):
        return adapter.invoke_turn(
            ModelTurnRequest(
                system_prompt=(
                    "Choose one action from ACTION1 or ACTION2. Preserve useful "
                    "discoveries for later turns."
                ),
                new_messages=[Message(role="user", content=prompt)],
                request_config=request_config,
                previous_state=state,
            )
        )

    try:
        first = turn(
            adapter.initial_state(),
            "Remember ORCHID-739 and submit ACTION1.",
        )
        assert first.response.reasoning_text
        state = adapter.buffer_inputs(
            first.state,
            [Message(role="user", content="Pending frame marker CEDAR-528.")],
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
        assert first.response.reasoning_text in str(requests[1].native_input)
        assert "CEDAR-528" not in str(requests[1].native_input)
        turn(compacted.state, "Submit ACTION2.")
        assert requests[2].native_input[-2]["content"] == (
            "Pending frame marker CEDAR-528."
        )
    finally:
        client.close()
