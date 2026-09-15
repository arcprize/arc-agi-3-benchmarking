from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol

from google.genai import _interactions as google_genai_interactions
from google.genai import types as google_genai_types

from .anthropic_runtime import (
    native_mapping,
    normalize_native_response,
    normalize_native_usage,
    validate_continuous_conversation_request,
)
from .exceptions import (
    ContextOverflowError,
    InvalidProviderResponseError,
    TransientProviderError,
)
from .runtime_models import (
    ModelRequest,
    ModelResponse,
    normalize_anthropic_messages_response,
    normalize_chat_completion_response,
    normalize_google_genai_response,
    normalize_google_interaction_response,
    normalize_responses_response,
)
from .runtime_state import (
    CONTINUOUS_CONVERSATION_RUNTIME_STATE,
    DEFAULT_RUNTIME_STATE,
    SERVER_RUNTIME_STATE,
    SUPPORTED_RUNTIME_STATES,
)

# Backwards-compatible alias for the default (client-managed) state.
SUPPORTED_RUNTIME_STATE = DEFAULT_RUNTIME_STATE
# Server-managed state is only available on the OpenAI Responses runtime.
SERVER_STATE_RUNTIME_KEYS = frozenset({("openai-python", "responses")})
CONTINUOUS_CONVERSATION_RUNTIME_KEYS = frozenset(
    {
        ("anthropic-python", "messages"),
        ("google-genai", "interactions"),
        ("openai-python", "responses"),
    }
)

_GOOGLE_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "exceeds the maximum number of tokens",
    "input token count exceeds",
    "maximum input tokens",
    "too many input tokens",
    "too many tokens",
)


def _is_google_context_overflow(error: Exception) -> bool:
    if not isinstance(error, google_genai_interactions.APIStatusError):
        return False
    if error.status_code not in {400, 413}:
        return False
    details = " ".join(
        str(value)
        for value in (error.message, error.body)
        if value is not None
    ).lower()
    return any(marker in details for marker in _GOOGLE_CONTEXT_OVERFLOW_MARKERS)


def _is_google_transient_error(error: Exception) -> bool:
    if isinstance(error, google_genai_interactions.APIConnectionError):
        return True
    return isinstance(error, google_genai_interactions.APIStatusError) and (
        error.status_code in {408, 409, 429} or error.status_code >= 500
    )


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
            request_kwargs["input"] = (
                list(request.native_input)
                if request.native_input is not None
                else messages[1:]
            )
            return request_kwargs

        request_kwargs["input"] = (
            list(request.native_input)
            if request.native_input is not None
            else messages
        )
        return request_kwargs

    def invoke(self, request: ModelRequest) -> ModelResponse:
        raw_response = self._client.responses.create(
            **self._build_request_kwargs(request),
        )
        return normalize_responses_response(raw_response)


class OpenAIResponsesServerStateAdapter:
    """OpenAI Responses adapter using server-managed conversation state.

    Instead of resending the whole transcript, the caller sends only the new
    message(s) for this turn plus a ``previous_response_id`` (in the request
    config). OpenAI holds the conversation state, and long runs are kept in
    budget by server-side compaction (``context_management`` /
    ``compact_threshold``).

    Requires ``store=true`` so the response chain resolves server-side; this
    runtime therefore needs a non-ZDR key.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _build_request_kwargs(request: ModelRequest) -> dict[str, Any]:
        request_kwargs = dict(request.request_config)

        # store=true is required for previous_response_id chaining.
        request_kwargs.setdefault("store", True)

        # Translate our compaction knob into the API's structured parameter.
        # `context_management` is newer than the pinned SDK's typed signature, so
        # send it via extra_body to avoid an SDK version bump.
        compact_threshold = request_kwargs.pop("compact_threshold", None)
        if compact_threshold is not None:
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            extra_body["context_management"] = [
                {"type": "compaction", "compact_threshold": compact_threshold}
            ]
            request_kwargs["extra_body"] = extra_body

        # Drop a null previous_response_id on the first turn so we never send it.
        if request_kwargs.get("previous_response_id") is None:
            request_kwargs.pop("previous_response_id", None)

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
        request_kwargs = deepcopy(request.request_config)
        messages = [message.model_dump() for message in request.messages]

        if request.native_input is not None:
            request_kwargs["messages"] = deepcopy(request.native_input)
            if request.messages and request.messages[0].role == "system":
                request_kwargs["system"] = request.messages[0].content
            return request_kwargs

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

    def _invoke_native_streaming(self, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        stopped = False
        try:
            with self._client.beta.messages.stream(**request_kwargs) as stream:
                for event in stream:
                    if event.type == "message_start":
                        message = native_mapping(event.message)
                        usage.update(message.get("usage", {}))
                    elif event.type == "message_delta":
                        delta = native_mapping(event.delta)
                        for key in ("stop_details", "stop_reason", "stop_sequence"):
                            if key in delta:
                                metadata[key] = delta[key]
                        usage.update(
                            {
                                key: value
                                for key, value in native_mapping(event.usage).items()
                                if value is not None
                            }
                        )
                    elif event.type == "message_stop":
                        stopped = True
                if not stopped:
                    raise RuntimeError("Anthropic stream ended before message_stop.")
                final = native_mapping(stream.get_final_message())
        except Exception as exc:
            raise InvalidProviderResponseError(
                f"Anthropic stream did not complete ({type(exc).__name__}).",
                response=metadata,
                usage=normalize_native_usage(usage),
            ) from None
        final.update(metadata)
        final["usage"] = usage
        return final

    def invoke(self, request: ModelRequest) -> ModelResponse:
        request_kwargs = self._build_request_kwargs(request)
        if request.native_input is not None:
            validate_continuous_conversation_request(request.request_config)
            if self._should_stream(request_kwargs):
                raw_response = self._invoke_native_streaming(request_kwargs)
            else:
                raw_response = self._client.beta.messages.create(**request_kwargs)
            return normalize_native_response(raw_response, request.request_config)
        if self._should_stream(request_kwargs):
            return self._invoke_streaming(request_kwargs)

        raw_response = self._client.messages.create(
            **request_kwargs,
        )
        return normalize_anthropic_messages_response(raw_response)


class GoogleGenAIGenerateContentAdapter:
    """Adapter for the native Google `google-genai` SDK.

    Translates our common ``Message`` schema into Gemini's ``Contents`` shape:
    a leading ``system`` message becomes ``GenerateContentConfig.system_instruction``,
    ``assistant`` is renamed to ``model``, and everything else flows through as
    text ``Part``s on the matching role.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _build_call_kwargs(request: ModelRequest) -> dict[str, Any]:
        request_config = dict(request.request_config)
        model = request_config.pop("model", None)
        if not model:
            raise ValueError(
                "Google GenAI request_config is missing required 'model'."
            )

        system_instruction: str | None = None
        messages = list(request.messages)
        if messages and messages[0].role == "system":
            system_instruction = messages[0].content
            messages = messages[1:]

        contents: list[google_genai_types.Content] = []
        for message in messages:
            role = "model" if message.role == "assistant" else message.role
            contents.append(
                google_genai_types.Content(
                    role=role,
                    parts=[google_genai_types.Part(text=message.content)],
                )
            )

        config_kwargs: dict[str, Any] = dict(request_config)
        if system_instruction is not None:
            config_kwargs.setdefault("system_instruction", system_instruction)

        config = google_genai_types.GenerateContentConfig(**config_kwargs)
        return {
            "model": model,
            "contents": contents,
            "config": config,
        }

    def invoke(self, request: ModelRequest) -> ModelResponse:
        raw_response = self._client.models.generate_content(
            **self._build_call_kwargs(request),
        )
        return normalize_google_genai_response(raw_response)


class GoogleGenAIInteractionsAdapter:
    """Adapter for the stateless Gemini Interactions API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _message_step(message: Any) -> dict[str, Any]:
        if message.role == "user":
            return {
                "type": "user_input",
                "content": [{"type": "text", "text": message.content}],
            }
        if message.role == "assistant":
            return {
                "type": "model_output",
                "content": [{"type": "text", "text": message.content}],
            }
        raise ValueError(
            "Google Interactions system messages must be sent as "
            "system_instruction."
        )

    @classmethod
    def _build_call_kwargs(cls, request: ModelRequest) -> dict[str, Any]:
        request_config = dict(request.request_config)
        model = request_config.pop("model", None)
        if not model:
            raise ValueError(
                "Google Interactions request_config is missing required 'model'."
            )

        messages = list(request.messages)
        if messages and messages[0].role == "system":
            request_config["system_instruction"] = messages[0].content
            messages = messages[1:]

        request_config["model"] = model
        request_config["input"] = (
            list(request.native_input)
            if request.native_input is not None
            else [cls._message_step(message) for message in messages]
        )
        return request_config

    def invoke(self, request: ModelRequest) -> ModelResponse:
        try:
            raw_response = self._client.interactions.create(
                **self._build_call_kwargs(request),
            )
        except google_genai_interactions.APIError as exc:
            if _is_google_context_overflow(exc):
                raise ContextOverflowError(str(exc)) from exc
            if _is_google_transient_error(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        return normalize_google_interaction_response(raw_response)


def build_model_runtime_adapter(
    *,
    client: Any,
    runtime_config: dict[str, Any],
    config_id: str,
) -> ModelRuntimeAdapter:
    runtime_state = runtime_config.get("state")
    if runtime_state not in SUPPORTED_RUNTIME_STATES:
        supported = ", ".join(repr(s) for s in sorted(SUPPORTED_RUNTIME_STATES))
        raise ValueError(
            f"Model config '{config_id}' uses runtime.state={runtime_state!r}, "
            f"but only {supported} are supported."
        )

    runtime_key = (runtime_config.get("sdk"), runtime_config.get("api"))

    if runtime_state == SERVER_RUNTIME_STATE:
        if runtime_key not in SERVER_STATE_RUNTIME_KEYS:
            raise ValueError(
                f"Model config '{config_id}' uses runtime.state={SERVER_RUNTIME_STATE!r}, "
                f"which is only supported on the OpenAI Responses runtime "
                f"(sdk='openai-python', api='responses')."
            )
        return OpenAIResponsesServerStateAdapter(client)

    if runtime_state == CONTINUOUS_CONVERSATION_RUNTIME_STATE:
        if runtime_key not in CONTINUOUS_CONVERSATION_RUNTIME_KEYS:
            raise ValueError(
                f"Model config '{config_id}' uses runtime.state="
                f"{CONTINUOUS_CONVERSATION_RUNTIME_STATE!r}, which is not supported "
                f"for sdk={runtime_key[0]!r}, api={runtime_key[1]!r}."
            )
        if runtime_key == ("anthropic-python", "messages"):
            return AnthropicMessagesAdapter(client)
        if runtime_key == ("openai-python", "responses"):
            return OpenAIResponsesAdapter(client)
        return GoogleGenAIInteractionsAdapter(client)

    if runtime_key == ("openai-python", "chat_completions"):
        return OpenAIChatCompletionsAdapter(client)
    if runtime_key == ("openai-python", "responses"):
        return OpenAIResponsesAdapter(client)
    if runtime_key == ("anthropic-python", "messages"):
        return AnthropicMessagesAdapter(client)
    if runtime_key == ("google-genai", "generate_content"):
        return GoogleGenAIGenerateContentAdapter(client)
    if runtime_key == ("google-genai", "interactions"):
        return GoogleGenAIInteractionsAdapter(client)

    raise ValueError(
        f"Model config '{config_id}' uses unsupported runtime "
        f"(sdk={runtime_config.get('sdk')!r}, api={runtime_config.get('api')!r})."
    )
