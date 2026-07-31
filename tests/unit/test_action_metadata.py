import re

import pytest
from arcengine import ActionInput

from benchmarking.action_metadata import (
    MAX_ACTION_METADATA_BYTES,
    TRUNCATION_MARKER,
    fit_action_metadata_payload,
    serialized_action_metadata_size,
)


def _truncation_stats(original: str, truncated: str) -> tuple[int, int]:
    match = re.search(r"\.\.\. truncated (\d+) characters \.\.\.", truncated)
    assert match is not None
    removed_chars = int(match.group(1))
    marker = TRUNCATION_MARKER.format(removed_chars=removed_chars)
    retained_chars = len(truncated) - len(marker)
    assert removed_chars == len(original) - retained_chars
    return retained_chars, removed_chars


@pytest.mark.unit
class TestActionMetadataPayloadFitter:
    def test_exact_boundary_payload_is_unchanged(self):
        empty_payload = {"output": "", "reasoning": None}
        overhead = serialized_action_metadata_size(empty_payload)
        payload = {
            "output": "x" * (MAX_ACTION_METADATA_BYTES - overhead),
            "reasoning": None,
        }

        fitted = fit_action_metadata_payload(payload)

        assert serialized_action_metadata_size(payload) == MAX_ACTION_METADATA_BYTES
        assert fitted == payload

    def test_non_encrypted_reasoning_payload_fits_complete_dictionary(self):
        payload = {
            "output": "ACTION1",
            "reasoning": "r" * 50_000,
            "usage": {"total_tokens": 123},
            "cost": {"total_cost": 0.01},
        }

        fitted = fit_action_metadata_payload(payload)

        retained, removed = _truncation_stats(
            payload["reasoning"],
            fitted["reasoning"],
        )
        assert retained > 0
        assert removed > 0
        assert fitted["reasoning"].startswith("r")
        assert fitted["reasoning"].endswith("r")
        assert serialized_action_metadata_size(fitted) <= MAX_ACTION_METADATA_BYTES

    def test_20k_output_and_20k_reasoning_are_reduced_together(self):
        payload = {
            "output": "o" * 20_000,
            "reasoning": "r" * 20_000,
            "usage": {"total_tokens": 123},
            "cost": {"total_cost": 0.01},
        }

        fitted = fit_action_metadata_payload(payload)

        output_retained, _ = _truncation_stats(payload["output"], fitted["output"])
        reasoning_retained, _ = _truncation_stats(
            payload["reasoning"],
            fitted["reasoning"],
        )
        assert output_retained == reasoning_retained
        assert serialized_action_metadata_size(fitted) <= MAX_ACTION_METADATA_BYTES

    def test_unicode_and_json_escaping_count_toward_byte_budget(self):
        payload = {
            "output": "ACTION1",
            "reasoning": ('\ud83e\udde9"\\\n' * 10_000) + "TAIL",
        }

        fitted = fit_action_metadata_payload(payload)

        retained, removed = _truncation_stats(
            payload["reasoning"],
            fitted["reasoning"],
        )
        assert retained > 0
        assert removed > 0
        assert fitted["reasoning"].endswith("TAIL")
        assert serialized_action_metadata_size(fitted) <= MAX_ACTION_METADATA_BYTES

    def test_input_dictionary_is_not_modified(self):
        payload = {"output": "o" * 30_000, "reasoning": "r" * 30_000}

        fitted = fit_action_metadata_payload(payload)

        assert payload == {"output": "o" * 30_000, "reasoning": "r" * 30_000}
        assert fitted != payload

    def test_fitted_payload_passes_arc_reasoning_validator(self):
        fitted = fit_action_metadata_payload(
            {"output": "o" * 50_000, "reasoning": "r" * 50_000}
        )

        action_input = ActionInput(reasoning=fitted)

        assert action_input.reasoning == fitted

    def test_raises_when_non_text_metadata_alone_exceeds_budget(self):
        payload = {
            "output": None,
            "reasoning": None,
            "usage": {"raw": "s" * 20_000},
        }

        with pytest.raises(ValueError, match="cannot fit within the byte budget"):
            fit_action_metadata_payload(payload)
