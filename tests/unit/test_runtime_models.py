from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from benchmarking.exceptions import EmptyResponseError
from benchmarking.models import ActionStateMetadata
from benchmarking.runtime_models import (
    Message,
    ModelRequest,
    ModelResponse,
    NormalizedUsage,
    action_metadata_from_model_response,
    normalize_anthropic_messages_response,
    normalize_chat_completion_response,
    normalize_google_genai_response,
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


def _google_genai_part(
    *,
    text: str = "",
    thought: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=thought)


def _google_genai_response(
    *,
    parts: list[SimpleNamespace] | None = None,
    prompt_token_count: int = 11,
    candidates_token_count: int = 7,
    thoughts_token_count: int = 0,
    cached_content_token_count: int = 0,
    total_token_count: int | None = None,
) -> SimpleNamespace:
    if parts is None:
        parts = [_google_genai_part(text="MOVE_LEFT")]
    if total_token_count is None:
        total_token_count = (
            prompt_token_count + candidates_token_count + thoughts_token_count
        )
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(role="model", parts=parts),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_token_count,
            candidates_token_count=candidates_token_count,
            thoughts_token_count=thoughts_token_count,
            cached_content_token_count=cached_content_token_count,
            total_token_count=total_token_count,
        ),
    )


def _anthropic_response(
    *,
    content: list[object] | None = None,
    input_tokens: int = 15,
    output_tokens: int = 7,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    iterations: list[object] | None = None,
    thinking_tokens: int = 0,
) -> SimpleNamespace:
    if content is None:
        content = [SimpleNamespace(type="text", text="TOKEN_PROBE")]
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            iterations=iterations,
            output_tokens_details=SimpleNamespace(
                thinking_tokens=thinking_tokens,
            ),
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
        assert "state" not in metadata.to_reasoning_dict()

    def test_action_metadata_omits_unpopulated_compaction_fields(self):
        metadata = action_metadata_from_model_response(
            _normalized_model_response(),
            pricing={},
        )
        metadata.state = ActionStateMetadata(
            input_items_sent=12,
            compaction_occurred=False,
            compaction_items_returned=0,
        )

        assert metadata.to_reasoning_dict()["state"] == {
            "input_items_sent": 12,
            "compaction_occurred": False,
            "compaction_items_returned": 0,
        }

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

    def test_responses_normalizer_extracts_response_id_when_present(self):
        raw_response = _responses_response()
        raw_response.id = "resp_abc123"

        model_response = normalize_responses_response(raw_response)

        assert model_response.response_id == "resp_abc123"

    def test_responses_normalizer_response_id_is_none_when_absent(self):
        model_response = normalize_responses_response(_responses_response())

        assert model_response.response_id is None

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

    def test_anthropic_messages_normalizer_extracts_text_content_block(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                content=[SimpleNamespace(type="text", text="MOVE_LEFT")]
            )
        )

        assert model_response.output_text == "MOVE_LEFT"

    def test_anthropic_messages_normalizer_concatenates_multiple_text_blocks(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                content=[
                    SimpleNamespace(type="text", text="MOVE"),
                    SimpleNamespace(type="text", text="_LEFT"),
                ]
            )
        )

        assert model_response.output_text == "MOVE_LEFT"

    def test_anthropic_messages_normalizer_extracts_summarized_thinking(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                content=[
                    SimpleNamespace(type="thinking", thinking="inspect the top row"),
                    SimpleNamespace(type="text", text="RESET"),
                ]
            )
        )

        assert model_response.output_text == "RESET"
        assert model_response.reasoning_text == "inspect the top row"
        assert model_response.usage.reasoning_tokens == 0

    def test_anthropic_messages_normalizer_joins_multiple_thinking_summaries(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                content=[
                    SimpleNamespace(type="thinking", thinking="inspect"),
                    SimpleNamespace(type="redacted_thinking", data="opaque"),
                    SimpleNamespace(type="thinking", thinking="then move"),
                    SimpleNamespace(type="text", text="ACTION1"),
                ]
            )
        )

        assert model_response.reasoning_text == "inspect\nthen move"

    def test_anthropic_messages_usage_total_matches_input_plus_output_tokens(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(input_tokens=27, output_tokens=48)
        )

        assert model_response.usage.input_tokens == 27
        assert model_response.usage.output_tokens == 48
        assert model_response.usage.total_tokens == 75

    def test_anthropic_messages_usage_maps_cache_read_and_write_tokens(self):
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                input_tokens=100,
                output_tokens=20,
                cache_creation_input_tokens=31,
                cache_read_input_tokens=47,
            )
        )

        assert model_response.usage.cached_tokens == 47
        assert model_response.usage.cache_write_tokens == 31

    def test_anthropic_compaction_usage_sums_all_sampling_iterations(self):
        iterations = [
            SimpleNamespace(
                type="compaction",
                input_tokens=180_000,
                output_tokens=3_500,
                cache_creation_input_tokens=11,
                cache_read_input_tokens=13,
            ),
            SimpleNamespace(
                type="message",
                input_tokens=23_000,
                output_tokens=1_000,
                cache_creation_input_tokens=17,
                cache_read_input_tokens=19,
            ),
        ]
        model_response = normalize_anthropic_messages_response(
            _anthropic_response(
                input_tokens=23_000,
                output_tokens=1_000,
                iterations=iterations,
                thinking_tokens=600,
            )
        )

        assert model_response.usage.input_tokens == 203_000
        assert model_response.usage.output_tokens == 4_500
        assert model_response.usage.total_tokens == 207_500
        assert model_response.usage.reasoning_tokens == 600
        assert model_response.usage.cached_tokens == 32
        assert model_response.usage.cache_write_tokens == 28

    def test_anthropic_messages_normalizer_maps_dict_response_and_usage_schema(self):
        raw_response = {
            "content": [
                {
                    "citations": None,
                    "text": "TOKEN_PROBE",
                    "type": "text",
                }
            ],
            "usage": {
                "input_tokens": 15,
                "output_tokens": 7,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

        model_response = normalize_anthropic_messages_response(raw_response)

        assert model_response.output_text == "TOKEN_PROBE"
        assert model_response.usage.input_tokens == 15
        assert model_response.usage.output_tokens == 7
        assert model_response.usage.total_tokens == 22
        assert model_response.usage.cached_tokens == 0
        assert model_response.usage.cache_write_tokens == 0

    def test_anthropic_messages_empty_content_raises_empty_response_error(self):
        raw_response = _anthropic_response(content=[])

        with pytest.raises(EmptyResponseError) as exc_info:
            normalize_anthropic_messages_response(raw_response)

        assert str(exc_info.value) == "API returned 200 with empty output."
        assert exc_info.value.response is raw_response

    def test_anthropic_messages_content_without_text_raises_empty_response_error(self):
        raw_response = _anthropic_response(
            content=[SimpleNamespace(type="thinking", thinking="inspect")]
        )

        with pytest.raises(EmptyResponseError) as exc_info:
            normalize_anthropic_messages_response(raw_response)

        assert str(exc_info.value) == "API returned 200 with empty output."
        assert exc_info.value.response is raw_response

    def test_anthropic_messages_raw_response_is_preserved(self):
        raw_response = _anthropic_response()

        model_response = normalize_anthropic_messages_response(raw_response)

        assert model_response.raw_response is raw_response

    def test_google_genai_normalizer_extracts_text_from_visible_parts(self):
        response = _google_genai_response(
            parts=[
                _google_genai_part(text="MOVE", thought=False),
                _google_genai_part(text="_LEFT", thought=False),
            ]
        )

        model_response = normalize_google_genai_response(response)

        assert model_response.output_text == "MOVE_LEFT"

    def test_google_genai_normalizer_skips_thought_parts_in_output_text(self):
        response = _google_genai_response(
            parts=[
                _google_genai_part(text="inspect the board", thought=True),
                _google_genai_part(text="RESET", thought=False),
            ]
        )

        model_response = normalize_google_genai_response(response)

        assert model_response.output_text == "RESET"
        assert model_response.reasoning_text == "inspect the board"

    def test_google_genai_normalizer_concatenates_multiple_thought_parts(self):
        response = _google_genai_response(
            parts=[
                _google_genai_part(text="first thought", thought=True),
                _google_genai_part(text="second thought", thought=True),
                _google_genai_part(text="ANSWER", thought=False),
            ]
        )

        model_response = normalize_google_genai_response(response)

        assert model_response.output_text == "ANSWER"
        assert model_response.reasoning_text == "first thought\nsecond thought"

    def test_google_genai_normalizer_returns_none_reasoning_when_no_thought_parts(self):
        response = _google_genai_response(
            parts=[_google_genai_part(text="MOVE_LEFT", thought=False)]
        )

        model_response = normalize_google_genai_response(response)

        assert model_response.reasoning_text is None

    def test_google_genai_normalizer_raises_when_no_visible_text_parts(self):
        response = _google_genai_response(
            parts=[_google_genai_part(text="hidden thought", thought=True)]
        )

        with pytest.raises(EmptyResponseError) as exc_info:
            normalize_google_genai_response(response)

        assert str(exc_info.value) == "API returned 200 with empty output."
        assert exc_info.value.response is response

    def test_google_genai_normalizer_raises_when_no_candidates(self):
        response = SimpleNamespace(
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=5,
                candidates_token_count=0,
                thoughts_token_count=0,
                cached_content_token_count=0,
                total_token_count=5,
            ),
        )

        with pytest.raises(EmptyResponseError):
            normalize_google_genai_response(response)

    def test_google_genai_normalizer_maps_prompt_to_input_tokens(self):
        model_response = normalize_google_genai_response(
            _google_genai_response(
                prompt_token_count=120,
                candidates_token_count=30,
                thoughts_token_count=0,
            )
        )

        assert model_response.usage.input_tokens == 120

    def test_google_genai_normalizer_folds_thoughts_into_output_tokens(self):
        """Critical for cost: Gemini bills (candidates + thoughts) at the output
        rate, so output_tokens must include thoughts so that
        output_cost = output_tokens * output_price matches Gemini billing.
        """
        model_response = normalize_google_genai_response(
            _google_genai_response(
                prompt_token_count=100,
                candidates_token_count=30,
                thoughts_token_count=200,
            )
        )

        assert model_response.usage.output_tokens == 230
        assert model_response.usage.reasoning_tokens == 200

    def test_google_genai_normalizer_total_tokens_equals_prompt_plus_output_tokens(
        self,
    ):
        model_response = normalize_google_genai_response(
            _google_genai_response(
                prompt_token_count=100,
                candidates_token_count=30,
                thoughts_token_count=200,
            )
        )

        assert model_response.usage.total_tokens == (
            model_response.usage.input_tokens + model_response.usage.output_tokens
        )
        assert model_response.usage.total_tokens == 330

    def test_google_genai_normalizer_maps_cached_content_to_cached_tokens(self):
        model_response = normalize_google_genai_response(
            _google_genai_response(
                prompt_token_count=500,
                candidates_token_count=20,
                cached_content_token_count=400,
            )
        )

        assert model_response.usage.cached_tokens == 400
        assert model_response.usage.cache_write_tokens == 0

    def test_google_genai_normalizer_treats_missing_usage_fields_as_zero(self):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        role="model",
                        parts=[_google_genai_part(text="HI")],
                    )
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=None,
                candidates_token_count=None,
                thoughts_token_count=None,
                cached_content_token_count=None,
                total_token_count=None,
            ),
        )

        model_response = normalize_google_genai_response(response)

        assert model_response.usage.input_tokens == 0
        assert model_response.usage.output_tokens == 0
        assert model_response.usage.total_tokens == 0
        assert model_response.usage.reasoning_tokens == 0
        assert model_response.usage.cached_tokens == 0

    def test_google_genai_normalizer_maps_dict_response_and_usage_schema(self):
        raw_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "thinking through", "thought": True},
                            {"text": "TOKEN_PROBE", "thought": False},
                        ],
                    }
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 50,
                "candidates_token_count": 7,
                "thoughts_token_count": 13,
                "cached_content_token_count": 5,
                "total_token_count": 70,
            },
        }

        model_response = normalize_google_genai_response(raw_response)

        assert model_response.output_text == "TOKEN_PROBE"
        assert model_response.reasoning_text == "thinking through"
        assert model_response.usage.input_tokens == 50
        assert model_response.usage.output_tokens == 20
        assert model_response.usage.reasoning_tokens == 13
        assert model_response.usage.cached_tokens == 5
        assert model_response.usage.total_tokens == 70

    def test_google_genai_raw_response_is_preserved(self):
        raw_response = _google_genai_response()

        model_response = normalize_google_genai_response(raw_response)

        assert model_response.raw_response is raw_response

    def test_google_genai_metadata_projection_costs_match_gemini_billing(self):
        """End-to-end check: with `input=$2/M` and `output=$10/M`, a response
        with 1,000 prompt / 200 candidate / 300 thought tokens must price out
        as ``1_000*2e-6 + (200+300)*10e-6 = $0.007``.
        """
        raw_response = _google_genai_response(
            parts=[_google_genai_part(text="PUSH")],
            prompt_token_count=1_000,
            candidates_token_count=200,
            thoughts_token_count=300,
        )

        metadata = action_metadata_from_model_response(
            normalize_google_genai_response(raw_response),
            pricing={"input": 2.00, "output": 10.00},
        )

        assert metadata.output == "PUSH"
        assert metadata.usage.input_tokens == 1_000
        assert metadata.usage.output_tokens == 500
        assert metadata.usage.total_tokens == 1_500
        assert metadata.usage.output_tokens_details.reasoning_tokens == 300
        assert metadata.cost.input_cost == pytest.approx(0.002)
        assert metadata.cost.output_cost == pytest.approx(0.005)
        assert metadata.cost.total_cost == pytest.approx(0.007)

    def test_google_genai_reasoning_tokens_billed_at_output_price(self):
        """Cost regression: thinking tokens with no visible output must still
        be billed at the output rate, otherwise we under-report Gemini cost."""
        raw_response = _google_genai_response(
            parts=[_google_genai_part(text="X")],
            prompt_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=1_000_000,
        )

        metadata = action_metadata_from_model_response(
            normalize_google_genai_response(raw_response),
            pricing={"input": 2.00, "output": 9.00},
        )

        assert metadata.usage.output_tokens == 1_000_000
        assert metadata.cost.output_cost == pytest.approx(9.00)
        assert metadata.cost.total_cost == pytest.approx(9.00)

    def test_google_genai_metadata_projection_matches_anthropic_metadata_schema(
        self,
    ):
        gemini_metadata = action_metadata_from_model_response(
            normalize_google_genai_response(
                _google_genai_response(
                    parts=[_google_genai_part(text="PUSH")],
                    prompt_token_count=1_000,
                    candidates_token_count=200,
                    thoughts_token_count=0,
                )
            ),
            pricing={"input": 5.00, "output": 25.00},
        )
        anthropic_metadata = action_metadata_from_model_response(
            normalize_anthropic_messages_response(
                _anthropic_response(
                    content=[SimpleNamespace(type="text", text="PUSH")],
                    input_tokens=1_000,
                    output_tokens=200,
                )
            ),
            pricing={"input": 5.00, "output": 25.00},
        )

        assert gemini_metadata.cost.model_dump() == anthropic_metadata.cost.model_dump()
        assert (
            gemini_metadata.usage.input_tokens == anthropic_metadata.usage.input_tokens
        )
        assert (
            gemini_metadata.usage.output_tokens
            == anthropic_metadata.usage.output_tokens
        )
        assert (
            gemini_metadata.usage.total_tokens == anthropic_metadata.usage.total_tokens
        )

    def test_anthropic_messages_metadata_projection_maps_usage_and_cost(self):
        raw_response = _anthropic_response(
            content=[SimpleNamespace(type="text", text="PUSH")],
            input_tokens=1_000,
            output_tokens=200,
        )

        metadata = action_metadata_from_model_response(
            normalize_anthropic_messages_response(raw_response),
            pricing={"input": 5.00, "output": 25.00},
        )

        assert metadata.output == "PUSH"
        assert metadata.reasoning is None
        assert metadata.usage.input_tokens == 1_000
        assert metadata.usage.output_tokens == 200
        assert metadata.usage.total_tokens == 1_200
        assert metadata.cost.input_cost == pytest.approx(0.005)
        assert metadata.cost.output_cost == pytest.approx(0.005)
        assert metadata.cost.total_cost == pytest.approx(0.01)
