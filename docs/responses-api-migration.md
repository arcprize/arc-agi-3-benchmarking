# Responses API Migration Plan

## Goal

Migrate the benchmarking agent from a hardcoded `chat.completions.create(...)` path to an adapter-driven runtime that can support:

- `chat_completions` as the current baseline
- `responses` as the first new runtime

This plan keeps the ARC agent logic stable and moves provider/API differences behind a normalized turn contract.

Adapter selection should be config-driven from the runtime tuple:

- `runtime.sdk`
- `runtime.api`

## Non-Goals For The First Cut

The first implementation should **not** include:

- multi-provider adapter support beyond the current OpenAI client shape
- API-managed conversation state (`previous_response_id`, `conversation`)
- background mode, streaming, or webhook polling
- opaque encrypted continuation-state handoff between turns

The first `responses` implementation should be:

- `sdk: openai-python`
- `api: responses`
- `state: manual_rolling`

## Current Constraints

Today the code assumes all of the following:

- model config entries use `name` rather than `id`
- `call` means kwargs for `chat.completions.create(...)`
- the agent directly calls the OpenAI SDK
- the agent directly reads `response.choices[0].message`
- usage extraction is coupled to chat-completions/OpenRouter-like shapes

That means the first migration step is not "add Responses flags". The first migration step is to make the execution contract explicit.

## Proposed Config Shape

Current config sections should become:

- `id`: stable identifier for selection from CLI and code
- `agent`: ARC agent behavior
- `runtime`: execution profile
- `client`: SDK client construction
- `request`: adapter-specific request defaults
- `pricing`: pricing lookup

Example baseline (Chat Completions API):

```yaml
- id: "openai-gpt-5-4-2026-03-05"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5.0
    MAX_CONTEXT_LENGTH: 175_000

  runtime:
    sdk: "openai-python"
    api: "chat_completions"
    state: "manual_rolling"

  client:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"

  request:
    model: "gpt-5.4-2026-03-05"
    max_completion_tokens: 128_000
    reasoning_effort: "low"

  pricing:
    input: 2.50
    output: 15.00
```

Example new runtime (Responses API):

```yaml
- id: "openai-gpt-5-4-2026-03-05-responses"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5.0
    MAX_CONTEXT_LENGTH: 175_000

  runtime:
    sdk: "openai-python"
    api: "responses"
    state: "manual_rolling"

  client:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"

  request:
    model: "gpt-5.4-2026-03-05"
    max_output_tokens: 128_000
    reasoning:
      effort: "low"

  pricing:
    input: 2.50
    output: 15.00
```

## Proposed Normalized Contracts

The first implementation should keep these contracts as small as possible. The real system is:

- the agent owns one conversation history
- for V1, that history uses `runtime.state == "manual_rolling"`
- the agent decides which messages to send on a given turn
- the agent sends the full sliced transcript on each turn instead of relying on API-managed continuation state
- the runtime adapter sends those messages to the model
- the runtime adapter normalizes the model output
- the agent appends the assistant response back into history

### Message

```python
Message(
    role: Literal["system", "user", "assistant"],
    content: str,
)
```

Purpose:

- represent one transcript item in a runtime-neutral way
- keep the normalized role set to `system`, `user`, and `assistant`
- let each adapter map `system` to the provider-specific instruction channel (`instructions`, `system`, etc.)

### ModelRequest

This is the request object built by the ARC agent and sent to the runtime adapter.

```python
ModelRequest(
    messages: list[Message],
    request_config: dict[str, Any],
)
```

Purpose:

- represent the exact slice of conversation history being sent on one model call
- carry the adapter-specific request kwargs exactly as configured

Notes:

- level and available actions remain embedded in the rendered user message
- the agent decides how to construct and trim those messages
- `request_config` includes `model` and the rest of the per-call kwargs from YAML

### NormalizedUsage

```python
NormalizedUsage(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost: float = 0.0,
    cost_details: dict[str, float] = {},
)
```

### ModelResponse

This is the only result shape the ARC agent should depend on.

```python
ModelResponse(
    output_text: str,
    reasoning_text: str | None,
    usage: NormalizedUsage,
    raw_response: Any | None,
)
```

Notes:

- the ARC-facing action metadata schema remains the payload sent through the environment
- that metadata schema is **not** the adapter boundary
- the ARC-facing metadata payload should be built from `ModelResponse`
- `reasoning_text` means human-readable reasoning summary/text only
- do **not** overload `reasoning_text` with opaque encrypted continuation blobs
- if a future runtime needs to pass opaque continuation state across manual stateless turns, add a dedicated adapter-state field instead of stuffing that data into transcript text

## Adapter Boundary

Create one runtime adapter interface:

```python
class ModelRuntimeAdapter(Protocol):
    def invoke(self, request: ModelRequest) -> ModelResponse: ...
```

First implementations:

- `OpenAIChatCompletionsAdapter`
- `OpenAIResponsesAdapter`

Dispatch and validation should use `(runtime.sdk, runtime.api)` as the adapter key.

For V1, both adapters should require `runtime.state == "manual_rolling"`.

Responsibilities:

### Agent-owned

- system prompt construction
- conversation history
- message slicing per turn
- frame rendering
- context trimming
- retry policy
- action parsing
- ARC-facing metadata payload construction

### Adapter-owned

- request translation
- provider-specific mapping from normalized message roles into the target API request shape
- raw SDK invocation
- response normalization
- usage extraction
- request-config interpretation

## Required Naming Cleanup

Before or during migration, rename config identity terminology to be explicit:

- config entry field: `name` -> `id`
- user-facing variable naming: `config_name` -> `config_id`

Reason:

- the config selector is not a display name
- the CLI and loader semantics are identifier-based
- future migrations will be easier if config identity is explicit everywhere

## Cross-Field Validation Rules

The config loader should validate these rules:

- `runtime.sdk` is required
- `runtime.api` is required
- `runtime.state` is required

Adapter-compatibility rules:

- if `(runtime.sdk, runtime.api) == ("openai-python", "chat_completions")`, validate chat-completions request fields
- if `(runtime.sdk, runtime.api) == ("openai-python", "responses")`, validate responses request fields
- for V1, reject any runtime where `runtime.state != "manual_rolling"`
- reject unsupported `(runtime.sdk, runtime.api)` pairs with a clear config error

## Phased Checklist

Use this section as the execution plan. Complete each phase in order. Do not start the next phase until the current phase exit criteria are satisfied.

### Phase 0: Freeze The Current Baseline

Purpose: capture current behavior before changing structure.

- [x] Add or tighten tests around current config loading
- [x] Add or tighten tests around current chat-completions result parsing
- [x] Document the current assumptions in tests

Tests to add and pass: **6**

- [x] Test 0.1: model config list returns all current config identifiers
- [x] Test 0.2: missing config raises a clear config-not-found error
- [x] Test 0.3: missing `client.api_key_env` raises a clear error
- [x] Test 0.4: current chat-completions usage extraction maps prompt/completion/total tokens correctly
- [x] Test 0.5: current assistant text extraction uses the first choice message content
- [x] Test 0.6: current action metadata payload contains output, reasoning, usage, and cost

Exit criteria:

- [x] All baseline tests pass
- [x] No behavioral change yet

### Phase 1: Reshape Config Schema

Purpose: introduce explicit config semantics without changing runtime behavior.

- [x] Rename config field `name` to `id`
- [x] Rename code references from `config_name` terminology to `config_id`
- [x] Split config sections into `agent`, `runtime`, `client`, `request`, `pricing`
- [x] Keep runtime execution still wired to current chat-completions behavior

Tests to add and pass: **10**

- [x] Test 1.1: config loader reads `id` successfully
- [x] Test 1.2: config loader rejects entries missing `id`
- [x] Test 1.3: config loader rejects duplicate `id` values
- [x] Test 1.4: `list_model_config_ids()` returns `id` values, not `name`
- [x] Test 1.5: config lookup by `config_id` succeeds
- [x] Test 1.6: config lookup by missing `config_id` fails with available IDs listed
- [x] Test 1.7: `runtime` section is required
- [x] Test 1.8: `client` section is required
- [x] Test 1.9: `request` section is required
- [x] Test 1.10: legacy `name`-only entry fails clearly or is explicitly migration-handled, depending on chosen strategy

Exit criteria:

- [x] Code selects configs by `config_id`
- [x] Config schema is explicit
- [x] Runtime behavior remains unchanged

### Phase 2: Introduce Normalized Turn Contracts

Purpose: make the agent depend on one internal request/response contract.

- [x] Add `Message`
- [x] Add `ModelRequest`
- [x] Add `NormalizedUsage`
- [x] Add `ModelResponse`
- [x] Add translation from `ModelResponse` to the ARC-facing metadata schema
- [x] Keep chat-completions as the only implemented runtime path for now

Tests to add and pass: **9**

- [x] Test 2.1: `ModelRequest` validates required fields
- [x] Test 2.2: `ModelResponse` validates required fields
- [x] Test 2.3: normalized usage defaults to zeros
- [x] Test 2.4: normalized usage supports reasoning and cache token details
- [x] Test 2.5: `ModelResponse` with empty output text is allowed or rejected according to the chosen contract
- [x] Test 2.6: ARC-facing metadata projection from normalized result maps output correctly
- [x] Test 2.7: ARC-facing metadata projection maps reasoning correctly
- [x] Test 2.8: ARC-facing metadata projection maps usage correctly
- [x] Test 2.9: ARC-facing metadata projection maps pricing-derived cost correctly

Exit criteria:

- [x] The agent consumes only normalized results
- [x] No raw SDK response shape is referenced outside runtime translation code

### Phase 3: Introduce Adapter Interface And Move Chat Completions Behind It

Purpose: move the existing chat-completions path behind the adapter boundary.

- [x] Add `ModelRuntimeAdapter`
- [x] Add `OpenAIChatCompletionsAdapter`
- [x] Add adapter selection keyed by `(runtime.sdk, runtime.api)`
- [x] Move chat request translation into adapter
- [x] Move chat response normalization into adapter
- [x] Validate that V1 adapter routes only run with `runtime.state == "manual_rolling"`
- [x] Keep agent retries and parsing unchanged

Tests to add and pass: **13**

- [x] Test 3.1: agent builds `ModelRequest` correctly from conversation state
- [x] Test 3.2: agent slices messages correctly for the outbound request
- [x] Test 3.3: chat adapter translates request conversation into chat messages correctly
- [x] Test 3.4: chat adapter passes through request-config kwargs correctly, including `model`
- [x] Test 3.5: chat adapter returns normalized assistant text correctly
- [x] Test 3.6: chat adapter returns normalized reasoning text correctly
- [x] Test 3.7: chat adapter normalizes usage correctly
- [x] Test 3.8: empty choices response raises the expected empty-response error
- [x] Test 3.9: unparseable assistant response still retries at the agent layer
- [x] Test 3.10: current chat path remains behaviorally equivalent to baseline tests
- [x] Test 3.11: current chat path still produces the ARC-facing metadata payload from normalized responses
- [x] Test 3.12: adapter dispatch selects the chat adapter from `(runtime.sdk, runtime.api)`
- [x] Test 3.13: unsupported `runtime.state` fails clearly for the chat route

Exit criteria:

- [x] Current chat-completions path is fully adapter-based
- [x] The agent no longer directly calls `client.chat.completions.create(...)`

### Phase 4: Add OpenAI Responses Adapter

Purpose: add the first `responses` runtime with the smallest possible behavioral delta.

- [x] Add `OpenAIResponsesAdapter`
- [x] Support `runtime.api == "responses"`
- [x] Convert agent-managed manual rolling message history into responses input format
- [x] Map the first `system` message into Responses `instructions` and the remaining turn history into `input`
- [x] Do not send `previous_response_id` or `conversation` in the V1 manual rolling route
- [x] Normalize responses output into `ModelResponse`

Tests to add and pass: **15**

- [x] Test 4.1: config validation accepts `runtime.api == "responses"`
- [x] Test 4.2: responses adapter translates system/user/assistant conversation correctly
- [x] Test 4.3: responses adapter passes request-config kwargs correctly, including `model`
- [x] Test 4.4: responses adapter passes request reasoning config correctly
- [x] Test 4.5: responses adapter extracts assistant text correctly from responses output
- [x] Test 4.6: responses adapter extracts human-readable reasoning summary/text correctly if present
- [x] Test 4.7: responses adapter normalizes usage fields correctly
- [x] Test 4.8: responses adapter handles empty or malformed output with the expected error
- [x] Test 4.9: agent can parse an action from a normalized responses result exactly as it does for chat
- [x] Test 4.10: responses path still produces the ARC-facing metadata payload from normalized responses
- [x] Test 4.11: responses config works with the same agent-owned trimming strategy
- [x] Test 4.12: config listing supports mixed `chat_completions` and `responses` configs
- [x] Test 4.13: responses adapter maps the first `system` message into `instructions`
- [x] Test 4.14: responses adapter sends remaining user/assistant turns in `input`
- [x] Test 4.15: responses adapter omits `previous_response_id` and `conversation` for `runtime.state == "manual_rolling"`

Exit criteria:

- [x] Both `chat_completions` and `responses` work through the same adapter boundary

### Phase 5: Regression And Parity Validation

Purpose: validate that the new architecture did not change core agent behavior unexpectedly.

- [x] Add parity-oriented tests
- [x] Run controlled comparisons between chat and responses configs

Tests to add and pass: **8**

- [x] Test 5.1: the same synthetic normalized output yields the same parsed action regardless of runtime
- [x] Test 5.2: ARC-facing metadata generated from chat and responses normalized results has the same schema
- [x] Test 5.3: usage totals remain internally consistent for chat results
- [x] Test 5.4: usage totals remain internally consistent for responses results
- [x] Test 5.5: recordings still serialize successfully for chat runs
- [x] Test 5.6: recordings still serialize successfully for responses runs
- [x] Test 5.7: config listing still works with mixed chat and responses configs
- [x] Test 5.8: CLI validation still surfaces missing API key errors correctly for both config types

Exit criteria:

- [x] No regression in chat support
- [x] Responses support is integrated behind the same contract
- [x] The full local test suite runs cleanly end to end

### Phase 6: CI And GitHub Actions

Purpose: make the test plan enforceable on every change.

- [ ] Add a GitHub Actions workflow that installs dependencies and runs the full test suite
- [ ] Configure the workflow to run on pull requests and pushes to the main development branch
- [ ] Make the workflow fail on any test failure
- [ ] Document the workflow as the required validation path for future runtime work

Tests to add and pass: **3**

- [ ] Test 6.1: the GitHub Actions workflow installs the project successfully
- [ ] Test 6.2: the GitHub Actions workflow runs the full test suite successfully
- [ ] Test 6.3: the GitHub Actions workflow is triggered for the intended GitHub events

Exit criteria:

- [ ] A GitHub Actions workflow exists for the full test suite
- [ ] The workflow is suitable to gate future runtime changes

### Overall Done Criteria

- [ ] Configs are selected by `config_id`
- [ ] Runtime selection is explicit via `runtime`
- [ ] The agent depends only on normalized model responses
- [ ] The ARC-facing metadata schema is still emitted for the ARC environment
- [ ] Existing chat-completions support still works
- [ ] Responses support works via the same agent-managed message history model
- [ ] The full local test suite passes
- [ ] GitHub Actions is set up to run the full test suite automatically

## Recommended Test Mix

For the phases above, the test mix should be:

- unit tests for config loading and validation
- unit tests for request translation and response normalization
- unit tests for action metadata projection
- unit tests for retry and empty-response behavior
- a small number of integration-like tests using stubbed SDK responses

Avoid live API tests for the first migration unless they are explicitly separated and optional.

## Recommended Implementation Order

1. freeze baseline behavior with tests
2. rename `name` to `id` and `config_name` to `config_id`
3. introduce explicit config sections
4. introduce normalized turn contracts
5. move chat-completions behind adapter boundary
6. add responses adapter
7. run parity/regression pass and the full local test suite
8. add GitHub Actions coverage for the full test suite

## Minimum Success Definition

The migration is successful when:

- configs are selected by `config_id`
- runtime selection is explicit via `runtime`
- the agent depends only on normalized model responses
- the ARC-facing metadata schema is still emitted for the ARC environment
- existing chat-completions support still works
- responses support works via the same agent-managed message history model
- the full local test suite passes
- GitHub Actions runs the full test suite automatically
