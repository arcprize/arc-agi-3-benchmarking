from __future__ import annotations

from typing import Any, Protocol

from .runtime_models import (
    ModelRequest,
    ModelResponse,
    normalize_anthropic_messages_response,
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


class AnthropicMessagesAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _build_request_kwargs(request: ModelRequest) -> dict[str, Any]:
        request_kwargs = dict(request.request_config)
        messages = [message.model_dump() for message in request.messages]

        if request.messages and request.messages[0].role == "system":
            request_kwargs["system"] = request.messages[0].content
            request_kwargs["messages"] = messages[1:]
            return request_kwargs

        request_kwargs["messages"] = messages
        return request_kwargs

    @staticmethod
    def _should_stream(request_kwargs: dict[str, Any]) -> bool:
        stream = request_kwargs.pop("stream", False)
        if isinstance(stream, str):
            return stream.strip().lower() == "true"
        return bool(stream)

    @staticmethod
    def _stream_text_delta(event: Any) -> str | None:
        if getattr(event, "type", None) != "content_block_delta":
            return None

        delta = getattr(event, "delta", None)
        if getattr(delta, "type", None) != "text_delta":
            return None
        return getattr(delta, "text", "") or ""

    @staticmethod
    def _stream_usage(event: Any) -> Any | None:
        usage = getattr(event, "usage", None)
        if usage is not None:
            return usage

        delta = getattr(event, "delta", None)
        return getattr(delta, "usage", None)

    @staticmethod
    def _stream_response(
        *,
        final_message: Any,
        text_parts: list[str],
        fallback_usage: Any | None,
    ) -> Any:
        usage = getattr(final_message, "usage", None) or fallback_usage
        if text_parts:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "".join(text_parts),
                    }
                ],
                "usage": usage,
                "stream_final_message": final_message,
            }

        if fallback_usage is not None and getattr(final_message, "usage", None) is None:
            return {
                "content": getattr(final_message, "content", []),
                "usage": fallback_usage,
                "stream_final_message": final_message,
            }

        return final_message

    def _invoke_streaming(self, request_kwargs: dict[str, Any]) -> ModelResponse:
        text_parts: list[str] = []
        latest_usage = None

        with self._client.messages.stream(**request_kwargs) as stream:
            for event in stream:
                text_delta = self._stream_text_delta(event)
                if text_delta is not None:
                    text_parts.append(text_delta)

                event_usage = self._stream_usage(event)
                if event_usage is not None:
                    latest_usage = event_usage

            final_message = stream.get_final_message()

        return normalize_anthropic_messages_response(
            self._stream_response(
                final_message=final_message,
                text_parts=text_parts,
                fallback_usage=latest_usage,
            )
        )

    def invoke(self, request: ModelRequest) -> ModelResponse:
        request_kwargs = self._build_request_kwargs(request)
        if self._should_stream(request_kwargs):
            return self._invoke_streaming(request_kwargs)

        raw_response = self._client.messages.create(
            **request_kwargs,
        )
        return normalize_anthropic_messages_response(raw_response)


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
    if runtime_key == ("anthropic-python", "messages"):
        return AnthropicMessagesAdapter(client)

    raise ValueError(
        f"Model config '{config_id}' uses unsupported runtime "
        f"(sdk={runtime_config.get('sdk')!r}, api={runtime_config.get('api')!r})."
    )
