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
- id: "anthropic-claude-opus-4-6-native"
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
    model: "claude-opus-4-6"
    max_tokens: 128_000

  pricing:
    input: 5.00
    output: 25.00
```

For extended thinking, prefer native Anthropic request fields instead of
OpenAI-compatible wrappers:

```yaml
  request:
    model: "claude-opus-4-6"
    max_tokens: 128_000
    thinking:
      type: "enabled"
      budget_tokens: 2000
```

Notes:

- Native Anthropic configs should not use `extra_body`.
- Native Anthropic configs should not need `client.base_url` for the first cut.
- Keep existing OpenAI-compatible Anthropic configs temporarily if they are useful
  for comparison, but give native configs distinct IDs.

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
- `reasoning_text`: concatenate text from content blocks with `type == "thinking"`
  if present
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

- [ ] Add `AnthropicMessagesAdapter`.
- [ ] Add adapter dispatch for `("anthropic-python", "messages")`.
- [ ] Copy `request.request_config` before modifying request kwargs.
- [ ] Map the first leading `system` message to Anthropic's top-level `system`
  field.
- [ ] Send remaining `user` and `assistant` turns as Anthropic `messages`.
- [ ] If there is no leading `system` message, send all messages as Anthropic
  `messages`.
- [ ] Pass through request config kwargs such as `model`, `max_tokens`, and
  `thinking`.
- [ ] Preserve analysis-mode replay content as ordinary assistant message
  content.

Tests to implement:

- [ ] Adapter dispatch selects `AnthropicMessagesAdapter`.
- [ ] First system message maps to top-level `system`.
- [ ] Remaining user/assistant turns map to Anthropic `messages`.
- [ ] Requests without a leading system message send all turns as `messages`.
- [ ] Request config kwargs pass through unchanged.
- [ ] Thinking config passes through unchanged.
- [ ] Analysis-mode assistant replay content is preserved.
- [ ] Unsupported runtime state still fails before adapter selection.

### Phase 5: Add Anthropic Response Normalization

Purpose: convert native Anthropic responses into the existing normalized response
contract.

- [ ] Add `normalize_anthropic_messages_response(...)`.
- [ ] Extract assistant text from content blocks with `type == "text"`.
- [ ] Concatenate multiple text blocks in response order.
- [ ] Extract thinking text from content blocks with `type == "thinking"` when
  present.
- [ ] Concatenate multiple thinking blocks in response order.
- [ ] Normalize Anthropic usage fields into `NormalizedUsage`.
- [ ] Map provider cache creation tokens to `cache_write_tokens` if present.
- [ ] Map provider cache read tokens to `cached_tokens` if present.
- [ ] Raise `EmptyResponseError` for empty or malformed responses.
- [ ] Attach the raw response to `ModelResponse.raw_response`.

Tests to implement:

- [ ] Text content block becomes `ModelResponse.output_text`.
- [ ] Multiple text content blocks are concatenated correctly.
- [ ] Thinking content block becomes `ModelResponse.reasoning_text`.
- [ ] Multiple thinking content blocks are concatenated correctly.
- [ ] Missing thinking content returns `reasoning_text is None`.
- [ ] Usage input/output/total tokens normalize correctly.
- [ ] Cache read/write usage fields normalize when present.
- [ ] Empty content raises `EmptyResponseError`.
- [ ] Content with no text block raises `EmptyResponseError`.
- [ ] Raw response is preserved on `ModelResponse.raw_response`.

### Phase 6: Add Native Anthropic Configs

Purpose: make native Anthropic selectable from `--config`.

- [ ] Add one conservative native Anthropic config first.
- [ ] Use a clear ID such as `anthropic-claude-opus-4-6-native`.
- [ ] Set `runtime.sdk: "anthropic-python"`.
- [ ] Set `runtime.api: "messages"`.
- [ ] Set `runtime.state: "manual_rolling"`.
- [ ] Set `client.api_key_env: "ANTHROPIC_API_KEY"`.
- [ ] Use native request fields such as `model` and `max_tokens`.
- [ ] Do not use `extra_body` in native Anthropic configs.
- [ ] Add one optional thinking-enabled config only if needed for benchmark
  experiments.
- [ ] Keep existing OpenAI-compatible Anthropic configs temporarily if useful for
  comparison.
- [ ] Update README config examples only if the recommended default changes.

Tests to implement:

- [ ] Checked-in config IDs include the new native Anthropic config.
- [ ] Config loader accepts the native Anthropic config.
- [ ] Config loader rejects native Anthropic configs with incompatible
  `runtime.sdk`/`runtime.api` pairs.
- [ ] Config loader rejects native Anthropic configs that use OpenAI-only request
  fields if strict request validation is added.
- [ ] `uv run main.py --list-configs` includes the new config.

### Phase 7: Smoke Test Against Anthropic

Purpose: verify the native SDK route works outside unit-test fakes.

- [ ] Set `ANTHROPIC_API_KEY`.
- [ ] Run a one-game smoke test:

```bash
ANTHROPIC_API_KEY=... uv run main.py --game=ls20 --config=anthropic-claude-opus-4-6-native
```

- [ ] Confirm no OpenAI client is constructed for the native Anthropic config.
- [ ] Confirm the request reaches `client.messages.create(...)`.
- [ ] Confirm assistant response text parses into a valid game action.
- [ ] Confirm usage appears in step recordings.
- [ ] Confirm missing or invalid API key errors mention `ANTHROPIC_API_KEY`.
- [ ] Confirm at least one benchmark step completes using the native Anthropic
  SDK.
- [ ] Confirm generated recording includes normalized response, usage, and
  selected action.

Tests to implement:

- [ ] Add a small integration-style test or documented manual test for the native
  Anthropic route.
- [ ] Add a regression test that selecting the native Anthropic config routes to
  `anthropic-python/messages`.
- [ ] Add a regression test that selecting OpenAI configs still routes to the
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

- [ ] `agent.py` does not import provider SDKs directly.
- [ ] Runtime client construction is isolated in `runtime_clients.py`.
- [ ] Runtime adapter selection is keyed by `(runtime.sdk, runtime.api)`.
- [ ] Config validation rejects unsupported SDK/API combinations.
- [ ] Anthropic native configs use `anthropic-python/messages`.
- [ ] Anthropic native configs can be selected with `--config`.
- [ ] Anthropic native responses normalize into `ModelResponse`.
- [ ] OpenAI configs still behave as before.
- [ ] The same pattern is clear enough to add Google later without another agent
  refactor.
