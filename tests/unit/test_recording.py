import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from benchmarking.agent import BenchmarkingAgent
from benchmarking.recording import RunRecord, StepRecord, StepUsage
from benchmarking.runtime_models import ModelResponse, NormalizedUsage


def _model_response() -> ModelResponse:
    return ModelResponse(
        output_text="RESET",
        reasoning_text="restart",
        usage=NormalizedUsage(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            compute_units=559,
            reasoning_tokens=3,
            cached_tokens=5,
            cache_write_tokens=2,
            cost=0.42,
            cost_details={"provider_cost": 0.42},
        ),
    )


@pytest.mark.unit
class TestRecordingModels:
    def test_step_usage_from_response_maps_chat_completion_token_fields(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=45,
                total_tokens=168,
                compute_units=559,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=7,
                    cache_write_tokens=3,
                ),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=11),
                model_extra={"cost": 0.75, "cost_details": {"provider_cost": 0.75}},
            )
        )

        usage = StepUsage.from_response(response)

        assert usage.prompt_tokens == 123
        assert usage.completion_tokens == 45
        assert usage.total_tokens == 168
        assert usage.compute_units == 559
        assert usage.cached_tokens == 7
        assert usage.cache_write_tokens == 3
        assert usage.reasoning_tokens == 11
        assert usage.cost == 0.75
        assert usage.cost_details == {"provider_cost": 0.75}

    @pytest.mark.parametrize(
        "model_response",
        [
            _model_response(),
            ModelResponse(
                output_text="RESET",
                reasoning_text=None,
                usage=NormalizedUsage(input_tokens=36, output_tokens=87, total_tokens=123),
            ),
        ],
    )
    def test_step_and_run_records_serialize_successfully(self, model_response):
        step = StepRecord(
            step=1,
            timestamp=datetime.now(timezone.utc),
            model="gpt-5.4",
            messages_sent=[{"role": "user", "content": "frame"}],
            assistant_response=model_response.output_text,
            reasoning=model_response.reasoning_text,
            parsed_action="RESET",
            usage=StepUsage.from_normalized_usage(model_response.usage),
        )
        run = RunRecord(
            run_id="run-id",
            game_id="game-id",
            agent_name="agent",
            model="gpt-5.4",
            started_at=datetime.now(timezone.utc),
            total_steps=1,
            total_usage=step.usage,
            run_dir="recordings/run-id",
        )

        step_json = step.model_dump_json()
        run_json = run.model_dump_json()

        assert '"assistant_response":"RESET"' in step_json
        assert f'"total_tokens":{model_response.usage.total_tokens}' in run_json
        compute_units = model_response.usage.compute_units
        assert json.loads(step_json)["usage"]["compute_units"] == compute_units
        assert json.loads(run_json)["total_usage"]["compute_units"] == compute_units
        assert (step.usage + step.usage).compute_units == 2 * compute_units

    def test_legacy_recording_files_omit_new_opt_in_fields(self, tmp_path):
        agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
        agent.run_dir = str(tmp_path)
        agent.step_counter = 0
        agent.run_record = RunRecord(
            run_id="run-id",
            game_id="game-id",
            agent_name="agent",
            model="legacy-model",
            started_at=datetime.now(timezone.utc),
            run_dir=str(tmp_path),
        )
        step = StepRecord(
            step=1,
            timestamp=datetime.now(timezone.utc),
            model="legacy-model",
            messages_sent=[{"role": "user", "content": "frame"}],
            parsed_action="ACTION1",
        )

        agent._save_step(step)

        run_payload = json.loads((tmp_path / "run_meta.json").read_text())
        step_payload = json.loads((tmp_path / "step_001.json").read_text())
        assert "runtime" not in run_payload
        assert "request_record" not in step_payload
        assert "state_transition" not in step_payload
