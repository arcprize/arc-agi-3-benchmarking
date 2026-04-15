from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from benchmarking.runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
    action_metadata_from_model_response,
    normalize_chat_completion_response,
    normalize_responses_response,
)


def _normalized_model_response() -> ModelResponse:
    return ModelResponse(
        output_text="MOVE_LEFT",
        reasoning_text="shift the player left",
        usage=NormalizedUsage(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            reasoning_tokens=7,
            cached_tokens=9,
        ),
    )


def _chat_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="RESET",
                    reasoning="restart the level",
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=5,
                cache_write_tokens=2,
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            model_extra={"cost": 0.42, "cost_details": {"provider_cost": 0.42}},
        ),
    )


def _responses_response() -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="restart the level")],
                content=[],
            ),
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="RESET")],
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            input_tokens_details=SimpleNamespace(
                cached_tokens=5,
                cache_write_tokens=2,
            ),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            model_extra={"cost": 0.42, "cost_details": {"provider_cost": 0.42}},
        ),
    )


@pytest.mark.unit
class TestRuntimeModels:
    def test_model_request_validates_required_fields(self):
        with pytest.raises(ValidationError):
            ModelRequest(messages=[Message(role="user", content="frame")])

    def test_model_response_validates_required_fields(self):
        with pytest.raises(ValidationError):
            ModelResponse(output_text="MOVE_LEFT")

    def test_normalized_usage_defaults_to_zeros(self):
        usage = NormalizedUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.cost == 0.0
        assert usage.cost_details == {}

    def test_normalized_usage_supports_reasoning_and_cache_details(self):
        usage = NormalizedUsage(
            input_tokens=200,
            output_tokens=40,
            total_tokens=240,
            reasoning_tokens=10,
            cached_tokens=25,
            cache_write_tokens=5,
            cost=1.25,
            cost_details={"provider_cost": 1.25},
        )

        assert usage.reasoning_tokens == 10
        assert usage.cached_tokens == 25
        assert usage.cache_write_tokens == 5
        assert usage.cost == 1.25
        assert usage.cost_details == {"provider_cost": 1.25}

    def test_model_response_allows_empty_output_text(self):
        response = ModelResponse(output_text="", usage=NormalizedUsage())

        assert response.output_text == ""
        assert response.usage.total_tokens == 0

    def test_action_metadata_projection_maps_output_reasoning_usage_and_cost(self):
        metadata = action_metadata_from_model_response(
            _normalized_model_response(),
            pricing={"input": 2.50, "output": 15.00},
        )

        assert metadata.output == "MOVE_LEFT"
        assert metadata.reasoning == "shift the player left"
        assert metadata.usage.input_tokens == 120
        assert metadata.usage.output_tokens == 30
        assert metadata.usage.total_tokens == 150
        assert metadata.usage.input_tokens_details.cached_tokens == 9
        assert metadata.usage.output_tokens_details.reasoning_tokens == 7
        assert metadata.cost.input_cost == pytest.approx(0.0003)
        assert metadata.cost.output_cost == pytest.approx(0.00045)
        assert metadata.cost.total_cost == pytest.approx(0.00075)

    def test_chat_response_normalizer_uses_first_choice_message_content(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="FIRST",
                        reasoning="first-reasoning",
                    )
                ),
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="SECOND",
                        reasoning="second-reasoning",
                    )
                ),
            ]
        )

        model_response = normalize_chat_completion_response(response)

        assert model_response.output_text == "FIRST"
        assert model_response.reasoning_text == "first-reasoning"

    def test_chat_and_responses_metadata_projection_have_same_schema(self):
        chat_metadata = action_metadata_from_model_response(
            normalize_chat_completion_response(_chat_response()),
            pricing={"input": 2.50, "output": 15.00},
        )
        responses_metadata = action_metadata_from_model_response(
            normalize_responses_response(_responses_response()),
            pricing={"input": 2.50, "output": 15.00},
        )

        assert chat_metadata.model_dump() == responses_metadata.model_dump()

    def test_chat_usage_total_matches_input_plus_output_tokens(self):
        model_response = normalize_chat_completion_response(_chat_response())

        assert model_response.usage.total_tokens == (
            model_response.usage.input_tokens + model_response.usage.output_tokens
        )
        assert model_response.usage.total_tokens == 18

    def test_responses_usage_total_matches_input_plus_output_tokens(self):
        model_response = normalize_responses_response(_responses_response())

        assert model_response.usage.total_tokens == (
            model_response.usage.input_tokens + model_response.usage.output_tokens
        )
        assert model_response.usage.total_tokens == 18

    def test_responses_metadata_projection_maps_reasoning_usage_and_cost(self):
        raw_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="inspect the board")],
                    content=[],
                ),
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="PUSH")],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=1_000,
                output_tokens=200,
                total_tokens=1_200,
                input_tokens_details=SimpleNamespace(cached_tokens=50),
                output_tokens_details=SimpleNamespace(reasoning_tokens=25),
            ),
        )

        metadata = action_metadata_from_model_response(
            normalize_responses_response(raw_response),
            pricing={"input": 2.50, "output": 15.00},
        )

        assert metadata.output == "PUSH"
        assert metadata.reasoning == "inspect the board"
        assert metadata.usage.input_tokens == 1_000
        assert metadata.usage.output_tokens == 200
        assert metadata.usage.total_tokens == 1_200
        assert metadata.usage.input_tokens_details.cached_tokens == 50
        assert metadata.usage.output_tokens_details.reasoning_tokens == 25
        assert metadata.cost.input_cost == pytest.approx(0.0025)
        assert metadata.cost.output_cost == pytest.approx(0.003)
        assert metadata.cost.total_cost == pytest.approx(0.0055)

    def test_responses_normalizer_extracts_reasoning_summary_text_items(self):
        raw_response = {
            "output": [
                {
                    "id": "rs_123",
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "**Answering a simple question**\n\nParis is the capital.",
                        }
                    ],
                },
                {
                    "id": "msg_123",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The capital of France is Paris.",
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        }

        model_response = normalize_responses_response(raw_response)

        assert model_response.output_text == "The capital of France is Paris."
        assert model_response.reasoning_text == (
            "**Answering a simple question**\n\nParis is the capital."
        )

    def test_responses_normalizer_maps_dict_usage_schema_and_cost_without_double_counting_reasoning_tokens(
        self,
    ):
        raw_response = {
            "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
            "object": "response",
            "created_at": 1741476542,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "RESET",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "reasoning": {
                "effort": None,
                "summary": None,
            },
            "usage": {
                "input_tokens": 36,
                "input_tokens_details": {
                    "cached_tokens": 0,
                },
                "output_tokens": 87,
                "output_tokens_details": {
                    "reasoning_tokens": 10,
                },
                "total_tokens": 123,
            },
        }

        model_response = normalize_responses_response(raw_response)
        metadata = action_metadata_from_model_response(
            model_response=model_response,
            pricing={"input": 2.50, "output": 15.00},
        )

        assert model_response.output_text == "RESET"
        assert model_response.usage.input_tokens == 36
        assert model_response.usage.cached_tokens == 0
        assert model_response.usage.output_tokens == 87
        assert model_response.usage.reasoning_tokens == 10
        assert model_response.usage.total_tokens == 123
        assert model_response.usage.total_tokens == (
            model_response.usage.input_tokens + model_response.usage.output_tokens
        )
        assert metadata.usage.output_tokens_details.reasoning_tokens == 10
        assert metadata.cost.input_cost == pytest.approx(0.00009)
        assert metadata.cost.output_cost == pytest.approx(0.001305)
        assert metadata.cost.total_cost == pytest.approx(0.001395)
