"""Native action calls use the same one-action contract as structured text."""

import json
from copy import deepcopy

import pytest
from arcengine import GameAction
from google.genai import types

from benchmarking.agent import BenchmarkingAgent
from benchmarking.exceptions import InvalidProviderResponseError
from benchmarking.runtime_models import normalize_google_genai_response


def response(parts):
    return {
        "candidates": [{"content": {"parts": parts}, "finish_reason": "STOP"}],
        "usage_metadata": {"prompt_token_count": 7, "candidates_token_count": 2},
    }


def call(name="ACTION3", args=None):
    return {"functionCall": {"name": name, "args": {} if args is None else args}}


def parse(raw, available):
    normalized = normalize_google_genai_response(raw)
    agent = BenchmarkingAgent.__new__(BenchmarkingAgent)
    return agent._parse_action(normalized.output_text, available)


@pytest.mark.parametrize("sdk_object", [False, True])
@pytest.mark.parametrize("name", ["ACTION3", "google:ACTION3", "RESET"])
def test_native_simple_action_is_normalized_without_losing_raw_parts(name, sdk_object):
    part = call(name)
    part["thoughtSignature"] = "b3BhcXVl"
    raw = response([{"text": "reasoning", "thought": True}, part, {"text": ""}])
    if sdk_object:
        raw = types.GenerateContentResponse.model_validate(raw)
    before = deepcopy(raw)
    normalized = normalize_google_genai_response(raw)
    expected = name.removeprefix("google:")
    assert json.loads(normalized.output_text) == {
        "actions": [{"action_type": expected}]
    }
    assert normalized.reasoning_text == "reasoning"
    assert normalized.usage.total_tokens == 9
    assert normalized.raw_response is raw
    assert raw == before
    assert parse(raw, [GameAction.from_name(expected)]).name == expected


@pytest.mark.parametrize("name", ["ACTION6", "google:ACTION6"])
def test_native_click_reuses_coordinate_validation(name):
    raw = response([call(name, {"x": 22, "y": 29})])
    action = parse(raw, [GameAction.ACTION6])
    assert action.name == "ACTION6"
    assert action.action_data.x == 22
    assert action.action_data.y == 29


def test_snake_case_function_call_is_supported():
    raw = response([{"function_call": {"name": "ACTION3", "args": {}}}])
    assert parse(raw, [GameAction.ACTION3]).name == "ACTION3"


@pytest.mark.parametrize("sdk_object", [False, True])
@pytest.mark.parametrize("args", [{}, {"args": None}])
def test_simple_call_can_omit_empty_arguments(sdk_object, args):
    raw = response([{"functionCall": {"name": "ACTION3", **args}}])
    if sdk_object:
        raw = types.GenerateContentResponse.model_validate(raw)
    assert parse(raw, [GameAction.ACTION3]).name == "ACTION3"


def test_single_call_is_authoritative_over_incidental_text():
    raw = response([{"text": "ACTION1"}, call("ACTION3")])
    assert parse(raw, [GameAction.ACTION1, GameAction.ACTION3]).name == "ACTION3"


@pytest.mark.parametrize(
    "parts",
    [
        [call(), call("ACTION4")],
        [call(), call()],
        [call(), {"function_call": {"name": "ACTION4", "args": {}}}],
        [{"functionCall": {}}],
        [call("other:ACTION3")],
        [call("google:other:ACTION3")],
        [call("")],
        [call(3)],
        [{"functionCall": {"name": "ACTION3", "args": []}}],
        [{"functionCall": {"name": "ACTION3", "args": "{}"}}],
        [call(args={"action_type": "ACTION1"})],
        [call(args={"x": float("nan")})],
        [call(args={"x": object()})],
        [{"functionCall": {"name": "ACTION3", "args": {}, "willContinue": True}}],
        [{"functionCall": {"name": "ACTION3", "args": {}, "partialArgs": [1]}}],
        [{**call(), "thought": True}],
    ],
)
def test_invalid_native_calls_never_fall_back_to_action_text(parts):
    raw = response([*parts, {"text": "ACTION1"}])
    with pytest.raises(InvalidProviderResponseError) as exc:
        normalize_google_genai_response(raw)
    assert exc.value.response is raw
    assert exc.value.usage.total_tokens == 9


@pytest.mark.parametrize(
    "name,args,available",
    [
        ("ACTION3", {}, [GameAction.ACTION4]),
        ("action3", {}, [GameAction.ACTION3]),
        ("run_command", {}, [GameAction.ACTION3]),
        ("ACTION3", {"x": 1, "y": 2}, [GameAction.ACTION3]),
        ("ACTION6", {}, [GameAction.ACTION6]),
        ("ACTION6", {"x": 1}, [GameAction.ACTION6]),
        ("ACTION6", {"x": 1, "y": 2, "extra": 3}, [GameAction.ACTION6]),
        ("ACTION6", {"x": 1, "y": 2, "action": "ACTION3"}, [GameAction.ACTION6]),
        *[
            ("ACTION6", {"x": x, "y": 2}, [GameAction.ACTION6])
            for x in [-1, 64, True, 2.0, "2", None]
        ],
    ],
)
def test_native_actions_cannot_bypass_existing_strict_validation(name, args, available):
    assert parse(response([call(name, args)]), available) is None
