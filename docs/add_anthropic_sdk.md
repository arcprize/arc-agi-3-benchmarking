# Native Anthropic SDK Plan

## Goal

Add native Anthropic SDK support so Anthropic models can be selected by model
config the same way OpenAI models are selected today.

The end state should let a config choose:

- `runtime.sdk: "openai-python"` with `runtime.api: "chat_completions"`
- `runtime.sdk: "openai-python"` with `runtime.api: "responses"`
- `runtime.sdk: "anthropic-python"` with `runtime.api: "messages"`

This should also set up the runtime layer so a future native Google SDK adapter can
be added without changing agent orchestration again.

## Current State

The repo already has the right first abstraction: the agent builds a normalized
`ModelRequest`, and runtime adapters convert that request into provider-specific
SDK calls.

The current coupling is client construction:

- `BenchmarkingAgent` imports `OpenAI` directly.
- `BenchmarkingAgent` always builds an OpenAI client before adapter selection.
- existing Anthropic configs are currently routed through the OpenAI-compatible
  chat-completions shape, not the native Anthropic SDK.
- config validation accepts `runtime.api` values globally, but does not validate
  supported `(runtime.sdk, runtime.api)` pairs.

The native Anthropic implementation should remove those assumptions without
changing the ARC-facing agent behavior.

## Non-Goals For The First Cut

The first native Anthropic SDK implementation should not include:

- streaming
- API-managed conversation state
- prompt caching configuration beyond passing configured request kwargs through
- tool use
- multimodal message content
- automatic provider fallback
- a broad runtime registry framework

Keep the first cut small: one new client route, one new adapter route, one native
Anthropic config, and focused tests.

## Target Runtime Boundary

The agent should only know this flow:

1. load `agent`, `runtime`, `client`, `request`, and `pricing` config sections
2. build a provider SDK client from `runtime` plus `client`
3. build a runtime adapter from `runtime` plus the client
4. send normalized `ModelRequest` objects to the adapter
5. receive normalized `ModelResponse` objects back

Provider-specific behavior should live below the agent:

- client construction belongs in a runtime client factory
- request/response translation belongs in runtime adapters
- config compatibility validation belongs in model config loading

## Proposed Config Shape

Keep the same config sections already used by OpenAI configs.

```yaml
- id: "anthropic-opus-4-7-low"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5.0
    MAX_CONTEXT_LENGTH: 175_000

  runtime:
    sdk: "anthropic-python"
    api: "messages"
    state: "manual_rolling"

  client:
    api_key_env: "ANTHROPIC_API_KEY"

  request:
    model: "claude-opus-4-7"
    max_tokens: 20_000

  pricing:
    input: 5.00
    output: 25.00
```

For extended thinking, prefer native Anthropic request fields instead of
OpenAI-compatible wrappers:

```yaml
  request:
    model: "claude-opus-4-7"
    max_tokens: 20_000
    thinking:
      type: "adaptive"
    output_config:
      effort: "low"
```

Notes:

- Native Anthropic configs should not use `extra_body`.
- Native Anthropic configs should not need `client.base_url` for the first cut.
- Do not keep OpenAI-compatible Anthropic configs unless there is a specific
  benchmark comparison need.

## Runtime Compatibility Rules

Model config validation should validate supported runtime pairs, not just
individual API names.

Supported first-cut pairs:

```text
("openai-python", "chat_completions")
("openai-python", "responses")
("anthropic-python", "messages")
```

All first-cut pairs should keep:

```yaml
runtime:
  state: "manual_rolling"
```

This preserves the current agent-owned transcript behavior and keeps provider APIs
stateless from the app's perspective.

## Recommended Module Shape

Add a small client-construction module:

```text
benchmarking/runtime_clients.py
```

Responsibilities:

- read `runtime.sdk`
- read `client.api_key_env`
- validate that the environment variable is set
- build the correct SDK client
- return the client object to the agent
- raise clear config-specific errors for unsupported SDKs

Keep `benchmarking/runtime_adapters.py` responsible for adapter selection and
request/response translation.

The split should be:

- `model_config.py`: validates config shape and supported runtime pairs
- `runtime_clients.py`: builds SDK clients
- `runtime_adapters.py`: maps normalized requests to provider APIs and normalizes
  responses
- `agent.py`: orchestrates the run and never imports provider SDKs directly

This is intentionally not a heavy plugin registry. A small pair of factory
functions is enough for OpenAI, Anthropic, and a future Google SDK.

## Anthropic Messages Adapter Contract

Add an `AnthropicMessagesAdapter` implementing the existing runtime protocol:

```python
class ModelRuntimeAdapter(Protocol):
    def invoke(self, request: ModelRequest) -> ModelResponse: ...
```

The adapter should:

- copy `request.request_config`
- convert the first `system` message into Anthropic's top-level `system` field
- send remaining `user` and `assistant` messages as Anthropic `messages`
- call `client.messages.create(...)`
- normalize the raw Anthropic response into `ModelResponse`

Expected request mapping:

```text
ModelRequest.messages:
  system: "You are playing a game."
  user: "frame 1"
  assistant: "MOVE_LEFT"
  user: "frame 2"

Anthropic request:
  system: "You are playing a game."
  messages:
    - role: "user", content: "frame 1"
    - role: "assistant", content: "MOVE_LEFT"
    - role: "user", content: "frame 2"
```

If there is no leading system message, send all messages as `messages`.

## Anthropic Response Normalization

Add Anthropic normalization alongside the existing OpenAI normalization helpers.

The normalized output should preserve the current app contract:

```python
ModelResponse(
    output_text=str,
    reasoning_text=str | None,
    usage=NormalizedUsage(...),
    raw_response=response,
)
```

For native Anthropic responses:

- `output_text`: concatenate text from response content blocks with `type == "text"`
- `reasoning_text`: `None` for the first native Anthropic pass; Claude thinking
  blocks are intentionally deferred until analysis mode is implemented
- `input_tokens`: `response.usage.input_tokens`
- `output_tokens`: `response.usage.output_tokens`
- `total_tokens`: input plus output if no direct total is available
- `cache_write_tokens`: provider cache creation tokens if present
- `cached_tokens`: provider cache read tokens if present

If no text content block is present, raise `EmptyResponseError` with the raw
response attached, matching the current OpenAI adapter behavior.

## Phase Checklist

Work through these phases in order. Each phase should be independently testable
before moving on.

### Phase 1: Add Runtime Client Factory

Purpose: remove provider SDK construction from the agent before adding another
native SDK.

- [x] Add `benchmarking/runtime_clients.py`.
- [x] Add a `build_model_runtime_client(...)` factory.
- [x] Move OpenAI client construction into the client factory.
- [x] Keep OpenAI behavior unchanged for existing configs.
- [x] Move API-key environment validation into the client factory.
- [x] Change `BenchmarkingAgent` to call the client factory instead of importing
  `OpenAI` directly.
- [x] Keep adapter construction in `runtime_adapters.py`.
- [x] Confirm no provider SDK is imported directly in `agent.py`.

Tests to implement:

- [x] Client factory builds an OpenAI client for `runtime.sdk == "openai-python"`.
- [x] Client factory reads the configured API key env var.
- [x] Missing API key raises the same clear config-specific error.
- [x] Unsupported `runtime.sdk` raises a clear error.
- [x] `BenchmarkingAgent` still works with a fake OpenAI config/client route.
- [x] Existing OpenAI adapter tests still pass.

### Phase 2: Tighten Runtime Config Validation

Purpose: make SDK/API compatibility explicit before adding Anthropic.

- [x] Replace global `SUPPORTED_RUNTIME_APIS` validation with supported
  runtime-pair validation.
- [x] Keep `("openai-python", "chat_completions")` accepted.
- [x] Keep `("openai-python", "responses")` accepted.
- [x] Add `("anthropic-python", "messages")` as an accepted pair.
- [x] Keep `runtime.state == "manual_rolling"` validation for every supported
  pair.
- [x] Make unsupported-pair errors list the actual supported pairs.
- [x] Confirm config validation prevents accidental OpenAI-compatible and native
  Anthropic config mixing.

Tests to implement:

- [x] OpenAI chat-completions config still validates.
- [x] OpenAI responses config still validates.
- [x] Anthropic messages config validates.
- [x] `("anthropic-python", "chat_completions")` fails clearly.
- [x] `("openai-python", "messages")` fails clearly.
- [x] Missing `runtime.sdk` fails clearly.
- [x] Missing `runtime.api` fails clearly.
- [x] Invalid `runtime.state` fails clearly.
- [x] Checked-in config listing still works.

### Phase 3: Add Anthropic SDK Dependency

Purpose: make the native client available through normal project dependency
management.

- [x] Add `anthropic` to project dependencies.
- [x] Refresh `uv.lock`.
- [x] Import the Anthropic SDK only from `runtime_clients.py`.
- [x] Add Anthropic client construction to `build_model_runtime_client(...)`.
- [x] Use `client.api_key_env: "ANTHROPIC_API_KEY"` for native Anthropic
  configs.
- [x] Avoid constructing Anthropic clients in the agent or adapter layer.

Tests to implement:

- [x] Dependency lock update succeeds.
- [x] `uv run python -c "import anthropic"` succeeds.
- [x] Client factory builds an Anthropic client for
  `runtime.sdk == "anthropic-python"`.
- [x] Client factory reads `ANTHROPIC_API_KEY` from the configured env var.
- [x] Missing `ANTHROPIC_API_KEY` raises a clear config-specific error.
- [x] No production code outside the runtime client layer constructs Anthropic
  clients.

### Phase 4: Add Anthropic Messages Adapter Request Mapping

Purpose: route normalized model turns to native Anthropic Messages API.

- [x] Add `AnthropicMessagesAdapter`.
- [x] Add adapter dispatch for `("anthropic-python", "messages")`.
- [x] Copy `request.request_config` before modifying request kwargs.
- [x] Map the first leading `system` message to Anthropic's top-level `system`
  field.
- [x] Send remaining `user` and `assistant` turns as Anthropic `messages`.
- [x] If there is no leading `system` message, send all messages as Anthropic
  `messages`.
- [x] Pass through request config kwargs such as `model`, `max_tokens`, and
  `thinking`.
- [x] Preserve analysis-mode replay content as ordinary assistant message
  content.

Tests to implement:

- [x] Adapter dispatch selects `AnthropicMessagesAdapter`.
- [x] First system message maps to top-level `system`.
- [x] Remaining user/assistant turns map to Anthropic `messages`.
- [x] Requests without a leading system message send all turns as `messages`.
- [x] Request config kwargs pass through unchanged.
- [x] Thinking config passes through unchanged.
- [x] Analysis-mode assistant replay content is preserved.
- [x] Unsupported runtime state still fails before adapter selection.

### Phase 5: Add Anthropic Response Normalization

Purpose: convert native Anthropic responses into the existing normalized response
contract.

- [x] Add `normalize_anthropic_messages_response(...)`.
- [x] Extract assistant text from content blocks with `type == "text"`.
- [x] Concatenate multiple text blocks in response order.
- [x] Deferred: extract thinking text from content blocks with `type == "thinking"`
  when Claude analysis mode is implemented.
- [x] Deferred: concatenate multiple thinking blocks in response order when Claude
  analysis mode is implemented.
- [x] Normalize Anthropic usage fields into `NormalizedUsage`.
- [x] Map provider cache creation tokens to `cache_write_tokens` if present.
- [x] Map provider cache read tokens to `cached_tokens` if present.
- [x] Raise `EmptyResponseError` for empty or malformed responses.
- [x] Attach the raw response to `ModelResponse.raw_response`.

Tests to implement:

- [x] Text content block becomes `ModelResponse.output_text`.
- [x] Multiple text content blocks are concatenated correctly.
- [x] Deferred: thinking content block becomes `ModelResponse.reasoning_text` when
  Claude analysis mode is implemented.
- [x] Deferred: multiple thinking content blocks are concatenated correctly when
  Claude analysis mode is implemented.
- [x] Thinking content is ignored for now and returns `reasoning_text is None`.
- [x] Usage input/output/total tokens normalize correctly.
- [x] Cache read/write usage fields normalize when present.
- [x] Empty content raises `EmptyResponseError`.
- [x] Content with no text block raises `EmptyResponseError`.
- [x] Raw response is preserved on `ModelResponse.raw_response`.
- [x] Anthropic usage projects to pricing metadata without losing input/output
  tokens.

### Phase 6: Add Native Anthropic Configs

Purpose: make native Anthropic selectable from `--config`.

- [x] Add one conservative native Anthropic config first.
- [x] Use the existing ID `anthropic-opus-4-7-low`.
- [x] Set `runtime.sdk: "anthropic-python"`.
- [x] Set `runtime.api: "messages"`.
- [x] Set `runtime.state: "manual_rolling"`.
- [x] Set `client.api_key_env: "ANTHROPIC_API_KEY"`.
- [x] Use native request fields such as `model` and `max_tokens`.
- [x] Do not use `extra_body` in native Anthropic configs.
- [x] Add one optional thinking-enabled config only if needed for benchmark
  experiments.
- [x] Remove OpenAI-compatible Anthropic configs now that native Anthropic is the
  supported route.
- [x] Update README config examples.

Tests to implement:

- [x] Checked-in config IDs include the new native Anthropic config.
- [x] Config loader accepts the native Anthropic config.
- [x] Config loader rejects native Anthropic configs with incompatible
  `runtime.sdk`/`runtime.api` pairs.
- [x] Config loader rejects native Anthropic configs that use OpenAI-only request
  fields if strict request validation is added.
- [x] `uv run main.py --list-configs` includes the new config.
- [x] Checked-in Anthropic configs do not use OpenAI compatibility fields.

### Phase 7: Smoke Test Against Anthropic

Purpose: verify the native SDK route works outside unit-test fakes.

- [x] Set `ANTHROPIC_API_KEY`.
- [x] Run a one-game smoke test:

```bash
ANTHROPIC_API_KEY=... uv run main.py --game=ls20 --config=anthropic-opus-4-7-low
```

- [x] Confirm no OpenAI client is constructed for the native Anthropic config.
- [x] Confirm the request reaches `client.messages.create(...)`.
- [x] Confirm assistant response text parses into a valid game action.
- [x] Confirm usage appears in step recordings.
- [x] Confirm missing or invalid API key errors mention `ANTHROPIC_API_KEY`.
- [x] Confirm at least one benchmark step completes using the native Anthropic
  SDK.
- [x] Confirm generated recording includes normalized response, usage, and
  selected action.

Manual smoke result:

- `uv run main.py --game=ls20 --config=anthropic-opus-4-7-low` reached
  `POST https://api.anthropic.com/v1/messages` with `HTTP/1.1 200 OK`.
- Step 1 returned `ACTION1`, parsed `ACTION1`, recorded `total_tokens=12557`,
  and saved `step_001.json`.
- The run was interrupted after one action and the scorecard closed cleanly.
- `ANTHROPIC_API_KEY=` fails before gameplay with an error naming
  `ANTHROPIC_API_KEY`.

Tests to implement:

- [x] Add a small integration-style test or documented manual test for the native
  Anthropic route.
- [x] Add a regression test that selecting the native Anthropic config routes to
  `anthropic-python/messages`.
- [x] Add a regression test that selecting OpenAI configs still routes to the
  existing OpenAI adapters.

## Future Google SDK Extension

The Anthropic work should leave a clear path for Google:

1. add Google SDK dependency
2. add `runtime.sdk: "google-genai"` or whatever SDK name is chosen
3. add a supported runtime pair, for example `("google-genai", "models")`
4. add client construction in `runtime_clients.py`
5. add a Google adapter in `runtime_adapters.py`
6. add response normalization into `ModelResponse`
7. add provider-specific configs

Do not design a generic provider plugin system until at least Anthropic and
Google have both been implemented. Two concrete native SDKs will make the real
shared abstraction clearer than guessing up front.

## Final Done Checklist

- [x] `agent.py` does not import provider SDKs directly.
- [x] Runtime client construction is isolated in `runtime_clients.py`.
- [x] Runtime adapter selection is keyed by `(runtime.sdk, runtime.api)`.
- [x] Config validation rejects unsupported SDK/API combinations.
- [x] Anthropic native configs use `anthropic-python/messages`.
- [x] Anthropic native configs can be selected with `--config`.
- [x] Anthropic native responses normalize into `ModelResponse`.
- [x] OpenAI configs still behave as before.
- [x] The same pattern is clear enough to add Google later without another agent
  refactor.

## Add-On Plan: Native Anthropic Streaming

Purpose: allow a native Anthropic config to opt into streaming with
`stream: true`, especially for longer adaptive-thinking calls, without changing
the agent-facing `ModelResponse` contract.

Use the Python SDK streaming API documented in `docs/anthropic_streaming.md`:

```python
with client.messages.stream(
    model="claude-opus-4-7",
    max_tokens=20_000,
    thinking={"type": "adaptive"},
    output_config={"effort": "low"},
    messages=[...],
) as stream:
    for event in stream:
        ...
```

The config should look like:

```yaml
request:
  model: "claude-opus-4-7"
  max_tokens: 20_000
  stream: true
  thinking:
    type: "adaptive"
  output_config:
    effort: "low"
```

Implementation notes:

- Treat `request.stream` as an app routing flag, not as a normal Anthropic
  request kwarg.
- In `AnthropicMessagesAdapter.invoke(...)`, copy request kwargs as today, then
  remove `stream` before calling the SDK.
- If `stream` is falsy or absent, keep using `client.messages.create(...)`.
- If `stream` is true, call `client.messages.stream(...)`.
- Keep streaming implementation inside `AnthropicMessagesAdapter`; do not expose
  streaming to `agent.py`.
- Aggregate streamed `text_delta` chunks into final `output_text`.
- Ignore streamed `thinking_delta` chunks for now, matching the current
  non-streaming decision to defer Claude thinking blocks until analysis mode.
- Preserve token accounting by normalizing the final streamed message or final
  message-delta usage into the same `NormalizedUsage` contract as non-streaming
  Anthropic responses.
- Preserve `raw_response` with the final accumulated message or a small
  provider-shaped object containing final content and usage.
- If the stream finishes without any text output, raise `EmptyResponseError`.

### Phase 8: Add Anthropic Streaming Adapter Path

Purpose: make `stream: true` selectable from native Anthropic configs.

- [x] Add `_should_stream(...)` or equivalent helper for Anthropic request kwargs.
- [x] Pop `stream` before sending kwargs to Anthropic.
- [x] Keep non-streaming `client.messages.create(...)` behavior unchanged.
- [x] Add streaming path using `client.messages.stream(...)`.
- [x] Aggregate `content_block_delta` events where `delta.type == "text_delta"`.
- [x] Ignore `thinking_delta` events for now.
- [x] Capture final usage from the streamed response.
- [x] Return the same `ModelResponse` shape as non-streaming Anthropic calls.
- [x] Raise `EmptyResponseError` when a stream completes without text.
- [x] Add `stream: true` to the intended native Anthropic config.

Tests to implement:

- [x] `stream: true` routes to `client.messages.stream(...)`.
- [x] `stream: false` or missing `stream` routes to `client.messages.create(...)`.
- [x] `stream` is not forwarded as a provider request kwarg.
- [x] Streaming text deltas concatenate into `ModelResponse.output_text`.
- [x] Streaming thinking deltas do not populate `reasoning_text` yet.
- [x] Final streaming usage maps input/output/total/cache tokens correctly.
- [x] Final accumulated message usage takes precedence over intermittent event
  usage when both are present.
- [x] Empty streaming output raises `EmptyResponseError`.
- [x] Existing non-streaming Anthropic adapter tests still pass.
- [x] Checked-in streaming Anthropic config validates.

Manual smoke test:

- [x] Run `uv run main.py --game=ls20 --config=anthropic-opus-4-7-low`.
- [x] Confirm logs show a successful Anthropic request from the streaming config.
- [x] Confirm one action parses and records usage.
- [x] Confirm generated step JSON has the same usage fields as the non-streaming
  smoke test.

Manual smoke result:

- [x] `anthropic-opus-4-7-low` completed one `ACTION1` step with `stream: true`
  before the run was interrupted.
- [x] Step usage normalized as `prompt_tokens=12548`,
  `completion_tokens=9`, `total_tokens=12557`, `cached_tokens=0`,
  and `cache_write_tokens=0`.
- [x] Direct SDK probe with adaptive thinking confirmed
  `stream.get_final_message().usage` includes the full request totals:
  `input_tokens=30`, `output_tokens=9`, `cache_creation_input_tokens=0`, and
  `cache_read_input_tokens=0`.
