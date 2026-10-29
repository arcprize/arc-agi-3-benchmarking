"""Opt-in paid probe for Inception encrypted replay and forced compaction."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from dotenv import load_dotenv

import benchmarking.model_config as model_config
from benchmarking.compaction import SummaryCompactionPolicy, SummaryCompactor
from benchmarking.inception_runtime import INCEPTION_ADAPTER_ID
from benchmarking.runtime_adapters import build_model_runtime_adapter
from benchmarking.runtime_clients import build_model_runtime_client
from benchmarking.runtime_models import Message
from benchmarking.runtime_registry import (
    build_stateful_runtime_adapter,
    resolve_adapter_id,
)
from benchmarking.runtime_state import ModelTurnRequest, RuntimeState


@pytest.mark.integration
@pytest.mark.slow
def test_inception_encrypted_replay_and_compaction_live(monkeypatch):
    if os.environ.get("RUN_INCEPTION_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_INCEPTION_LIVE_TESTS=1 to authorize three paid requests.")
    load_dotenv()
    if os.environ.get("INCEPTION_LIVE_MODEL_CONFIGS"):
        monkeypatch.setattr(
            model_config,
            "MODEL_CONFIG_PATH",
            Path(os.environ["INCEPTION_LIVE_MODEL_CONFIGS"]),
        )
    config_id = os.environ.get(
        "INCEPTION_LIVE_CONFIG", "inception-mercury-3-preview-high-provider-adapter"
    )
    config = deepcopy(model_config.get_model_config(config_id))
    runtime = config["runtime"]
    assert resolve_adapter_id(runtime, config_id) == INCEPTION_ADAPTER_ID
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
                request_config=config["request"],
                previous_state=state,
            )
        )

    try:
        first = turn(
            adapter.initial_state(),
            "Remember the memory token ORCHID-739, then answer only READY.",
        )
        assert "READY" in first.response.output_text
        details = first.state.payload["messages"][-1]["reasoning_details"]
        assert details and all(d["type"] == "reasoning.encrypted" for d in details)
        assert first.response.usage.reasoning_tokens > 0
        state = RuntimeState.model_validate_json(first.state.model_dump_json())
        state = adapter.buffer_inputs(
            state,
            [
                Message(
                    role="user",
                    content="Pending observation: new frame marker CEDAR-528.",
                )
            ],
        )
        snapshot = state.model_dump()
        compacted = SummaryCompactor(
            SummaryCompactionPolicy(strategy="harness_summary", trigger_tokens=1)
        ).compact(
            adapter=adapter,
            state=state,
            request_config=config["request"],
            trigger_tokens=1,
            max_context_length=config["agent"]["MAX_CONTEXT_LENGTH"],
            max_retries=0,
        )
        assert requests[1].native_input[2]["reasoning_details"] == details
        assert "CEDAR-528" not in str(requests[1].native_input)
        assert state.model_dump() == snapshot
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
        readable = json.dumps([compacted.prompt, last.readable_request_messages])
        assert all(d["data"] not in readable for d in details)
    finally:
        client.close()
