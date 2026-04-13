from __future__ import annotations

from typing import Any, Protocol

from .runtime_models import (
    ModelRequest,
    ModelResponse,
    normalize_chat_completion_response,
    normalize_responses_response,
)

SUPPORTED_RUNTIME_STATE = "manual_rolling"


class ModelRuntimeAdapter(Protocol):
    def invoke(self, request: ModelRequest) -> ModelResponse: ...


class OpenAIChatCompletionsAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def invoke(self, request: ModelRequest) -> ModelResponse:
        raw_response = self._client.chat.completions.create(
            messages=[message.model_dump() for message in request.messages],
            **request.request_config,
        )
        return normalize_chat_completion_response(raw_response)


class OpenAIResponsesAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _build_request_kwargs(request: ModelRequest) -> dict[str, Any]:
        request_kwargs = dict(request.request_config)
        request_kwargs.pop("previous_response_id", None)
        request_kwargs.pop("conversation", None)

        messages = [message.model_dump() for message in request.messages]
        if request.messages and request.messages[0].role == "system":
            request_kwargs["instructions"] = request.messages[0].content
            request_kwargs["input"] = messages[1:]
            return request_kwargs

        request_kwargs["input"] = messages
        return request_kwargs

    def invoke(self, request: ModelRequest) -> ModelResponse:
        raw_response = self._client.responses.create(
            **self._build_request_kwargs(request),
        )
        return normalize_responses_response(raw_response)


def build_model_runtime_adapter(
    *,
    client: Any,
    runtime_config: dict[str, Any],
    config_id: str,
) -> ModelRuntimeAdapter:
    runtime_state = runtime_config.get("state")
    if runtime_state != SUPPORTED_RUNTIME_STATE:
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime_state!r}, "
            f"but only '{SUPPORTED_RUNTIME_STATE}' is supported in phase 3."
        )

    runtime_key = (runtime_config.get("sdk"), runtime_config.get("api"))
    if runtime_key == ("openai-python", "chat_completions"):
        return OpenAIChatCompletionsAdapter(client)
    if runtime_key == ("openai-python", "responses"):
        return OpenAIResponsesAdapter(client)

    raise ValueError(
        f"Model config '{config_id}' uses unsupported runtime "
        f"(sdk={runtime_config.get('sdk')!r}, api={runtime_config.get('api')!r})."
    )
