# Runtime state adapters

The benchmark harness separates four concerns:

1. `BenchmarkingAgent` owns the ARC scaffold: frame rendering, retry policy,
   action parsing, action submission, and the readable conversation mirror.
2. A provider API adapter translates a normalized model request to one SDK API
   call and normalizes the response.
3. A state strategy turns new frame messages plus the last accepted
   `RuntimeState` into the next provider request and a provisional next state.
4. A model configuration selects a local adapter ID, a state strategy, and API
   request settings.

Provider adapters return model output; they do not choose ARC actions. The
harness accepts a provisional state only after it parses a valid action. A retry
therefore starts from the same last accepted state and cannot add an orphaned
reasoning item or response ID to later turns.

The new turn contract is opt-in. Only `continuous_conversation` enters this
path. Existing `manual_rolling` configurations continue to use the original
agent-owned transcript and trimming code, and `previous_response_id` continues
to use its original request builder and response-handle fields.

## State contract

For continuous conversation, `RuntimeState` is a JSON-serializable envelope with:

- `schema_version`: the common state schema version
- `adapter_id`: the stable local implementation identifier
- `strategy`: `manual_rolling`, `previous_response_id`, or `continuous_conversation`
- `payload`: state owned and validated by the selected adapter

The harness validates schema versions, adapter IDs, strategies, and JSON
serialization. State is kept in memory for a run. This implementation does not
write encrypted checkpoints or support process restart and rehydration. A
durable design would also need explicit access control, encryption, retention,
deletion, and resume-compatibility policies.

`manual_rolling` stores the active normalized message window and applies the
existing estimated-token limit by removing the oldest complete turns. It is not
defined as "the last 10 messages". `previous_response_id` stores the latest
server response handle and any inputs waiting for the next API turn.

## Continuous conversation

`continuous_conversation` carries provider-native conversation and reasoning
state from one accepted turn to the next. It is implemented by
`openai.responses.v1`, `google.interactions.v1`, and `anthropic.messages.v1`.

### OpenAI Responses

The OpenAI adapter implements continuous conversation
through the Responses API with `store: false`. It requests
`reasoning.encrypted_content` and sends each accepted user input plus every
native `response.output` item into the following turn.
Replaying only serialized reasoning is not enough: every native output item can
be part of the model's state.

The adapter removes only two SDK response fields that the input schema rejects:
`status` from reasoning items and `created_by` from compaction items. When a
response contains compaction items, it retains the latest compaction item and
everything after it. Compaction remains a separate request setting under
`request.context_management`; the supplied GPT-5.6 Sol profile uses a 175k
threshold.

The OpenAI continuous-conversation configuration must:

- use the OpenAI Responses API and `runtime.adapter_id: openai.responses.v1`
- set `request.store: false`
- include `reasoning.encrypted_content`
- set `reasoning.context: auto` and `reasoning.summary: auto`
- avoid `previous_response_id`, conversations, and background mode

The checked-in profile uses max reasoning. The harness automatically removes
the manual carry-forward instruction when the
selected adapter provides continuous conversation. This is not a configurable
model setting. Existing `manual_rolling` configurations keep the instruction
because their visible replies are the state carried across turns.

This flow is ZDR-compatible, but it does not enable Zero Data Retention for an
organization. OpenAI must separately approve and configure ZDR for the
organization. See OpenAI's official documentation for
[stateless Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses#4-decide-when-to-use-statefulness),
[compaction](https://developers.openai.com/api/docs/guides/compaction), and
[Zero Data Retention controls](https://developers.openai.com/api/docs/guides/your-data#zero-data-retention).

### Anthropic Messages

The Anthropic Provider Adapter is implemented in `benchmarking/anthropic_runtime.py`
and registered as `anthropic.messages.v1`, with review status `unreviewed`.
Its `continuous_conversation` state carries native Messages content in memory;
it does not use server-hosted conversation IDs. The existing Anthropic
`manual_rolling` path stays separate and retains its existing behavior.

The adapter sends the system prompt separately and replays every accepted native
assistant content block in its original order, including readable thinking,
empty thinking text, signatures, redacted thinking, and compaction blocks.
Serialization preserves supplied empty fields but excludes SDK-added unset
fields. Native replay also removes response-only `parsed_output` from text
blocks and null `encrypted_content` from compaction blocks. Non-null encrypted
metadata, signatures, and unknown provider fields remain intact in memory.
State is provisional until the common agent parses a valid ARC action.
Retries reuse the last accepted state, including buffered GAME_OVER/reset
observations. The model and system prompt cannot change during a session.

The checked-in `anthropic-opus-5-low-provider-adapter` profile uses:

- `claude-opus-5`, adaptive thinking, summarized display, and low effort
- streaming, 128k maximum output, and 1,000k context capacity
- a 175k completed-context compaction trigger and 5x baseline action budget
- native on-demand compaction with `compact-2026-09-04`, capped at 8k output tokens
- standard-speed configured prices of $5/$25 per million input/output tokens
- the native `ANTHROPIC_API_KEY`, with no OpenAI `store` parameter
- the optional `thinking-token-count-2026-05-13` beta for reported thinking usage

Native compaction is optional and configured outside the provider request:

```yaml
runtime:
  compaction:
    strategy: native
    trigger_tokens: 175_000
    summary_max_output_tokens: 8_192
request:
  betas: ["compact-2026-09-04", "thinking-token-count-2026-05-13"]
```

The adapter checks the most recent accepted action's normalized input plus
output tokens as a completed-context estimate. It does not use cumulative run
usage, retry totals, or compaction-call usage as the trigger. At the threshold,
it sends only history through the last completed assistant action in a separate
Messages request with `compaction: {type: summarize}`. Pending observations,
including buffered GAME_OVER/reset inputs and the newest frame, are held aside.
The first frame cannot trigger compaction because no completed history exists.

The summary request keeps the same model, system prompt, thinking settings,
and effort. It uses the provider's default summary prompt, removes action-only
stop sequences and structured-output formatting, and caps output separately.
On success, exactly one nonempty signed `compaction` block replaces that entire
prefix. It goes first in the next action request, followed by every pending
observation unchanged and in order. Repeated compaction replaces the previous
summary and completed turns with one new signed block. Signatures are replayed
verbatim in memory, never reconstructed from the readable summary.

Action requests have neither `compaction` nor `context_management`. The adapter
rejects threshold compaction, its old beta, and both values of
`pause_after_compaction`; changing that flag alone would not protect the newest
frame. A signed block and the new beta are sent on every continuation request.
This on-demand API is available on the direct Claude API, not Bedrock or Google
Cloud. There is no harness-summary fallback, background compaction, tool
execution, or model switching.

Summary and action state form one provisional turn: neither is committed until
the action parser accepts a valid action. Failed, empty, unsigned, refused,
truncated, or interrupted compaction preserves the accepted history and pending
observations. If compaction succeeds but the action fails, retries compact the
unchanged accepted history again; returned usage from both calls is still billed
to the attempted turn. Retries remain bounded by the existing agent policy.

Only `end_turn` or an explicitly configured `stop_sequence` can supply an action.
Refusals, truncated or paused responses, and unfinished streams cannot execute
actions, even if partial text names an action. The adapter records sanitized
refusal details and uses the common bounded retry policy, without switching
models. An inline compaction block is also rejected on an action response. The
pinned Anthropic SDK remains `0.95.0`; its `extra_body` extension sends the new
top-level `compaction` parameter without a dependency upgrade. Signed summaries
arrive whole in `content_block_start`, with no content delta. The adapter retains streamed
`stop_details` explicitly because this SDK does not copy them into its final
accumulated message. Complete native content and per-iteration usage are retained
for both streaming and non-streaming requests.

Failed requests save allowlisted provider diagnostics: exception class, provider
error type, HTTP status, and request ID when available. This metadata survives
transport and runtime wrapping for both action and compaction requests. Provider
error messages and raw error bodies are not copied into those diagnostics.

Token accounting sums `usage.iterations` when present, including compaction;
top-level usage is a fallback, not an additional contribution. Native normalized
input tokens include uncached input plus cache reads and writes, with the cache
breakdowns retained separately. Output tokens already include thinking.
When available, `usage.output_tokens_details.thinking_tokens` populates
`reasoning_tokens` as a provider-reported breakdown, not additional output or
cost. The top-level breakdown covers non-compaction iterations and is used once;
per-iteration thinking counts are a fallback, with any separately reported
compaction thinking added once. Missing counts remain zero, meaning unreported,
and are never estimated from readable summaries. Streaming captures the
breakdown from the final `message_delta`, including usage retained on failure.
Separate compaction and action usage is added once per attempt. The accepted
action's native-compaction metadata also records the compaction-only usage and
the number of completed history items summarized, without an opaque payload.
Returned usage from unsuccessful attempts is attributed to the next accepted
action, or persisted in `run_meta.json` if retries are exhausted. No failed action
step is fabricated. Configured-price action estimates do not reconstruct cache
discounts or invoice adjustments, and are not written into provider-reported
`usage.cost`.

`run_meta.json` records the validated native policy in `runtime.compaction`,
including its strategy, trigger, summary output cap, and configured context
limit. Policy defaults are included even when omitted from the profile. This
does not instantiate the harness-summary compactor or its compaction counter.

The adapter supplies the shared `ModelTurnResult.readable_request_messages`
field for `messages_sent`. This is a readable projection of the active native
request, including
the current compaction summary, later observations, readable thinking summaries,
and text responses. Summarized-away history does not reappear in later model
request records. `request_record.input_items` records structural descriptors,
not opaque bodies. Signatures and redacted-thinking data are removed from
diagnostics, settings, logs, and action metadata; exact native replay state is
in memory only. Client-managed state is not a guarantee of provider-side zero
data retention: use credentials and data-retention terms appropriate for the
benchmark data.

Contract references, with on-demand compaction rechecked September 17, 2026:

- [Anthropic compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Thinking and replay](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [Thinking-token usage](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)
- [Stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Opus 5 specifications](https://platform.claude.com/docs/en/models/opus-5/overview)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)

### Google Gemini Interactions

The Google adapter uses the Interactions API with `store: false`. It preserves
each accepted user input and every model-generated step, including opaque
thought signatures, exactly as returned. Readable thought summaries are mapped
to the existing reasoning field. The configuration must set
`generation_config.thinking_summaries: auto` and must not use
`previous_interaction_id` or background mode. Setting `store: false` opts out of
Interaction state retention; project-level ZDR approval remains a separate
requirement. See Google's documentation for
[stateless interactions](https://ai.google.dev/gemini-api/docs/thought-signatures#stateless-mode)
and [Zero Data Retention](https://ai.google.dev/gemini-api/docs/zdr).

## Harness summary compaction

Harness-managed reconstruction is an optional adapter capability, represented
by `SummaryCompactionRuntimeAdapter`. OpenAI and Google implement it; Anthropic
implements only the common stateful turn contract and uses native compaction.
Both configuration validation and the summary compactor reject harness-summary
requests for Anthropic before making a provider call.

Providers without native compaction can select `runtime.compaction.strategy:
harness_summary`. `runtime.compaction.trigger_tokens` controls when the harness
schedules compaction, while `agent.MAX_CONTEXT_LENGTH` records the provider's
hard context capacity. Configuration validation reserves the requested summary
output plus `summary_input_headroom_tokens` between those values. The supplied
Gemini 3.8 Flash profile triggers at 175k against the model's
[documented 1,048,576-token input limit](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).
The selected model receives a separate, domain-neutral request to summarize the
conversation using its judgment about what matters for continuing the task in a
future context.

The resulting summary is inserted as one user-role continuation message in a
fresh provider state. This ends opaque reasoning continuity for the history
represented by the summary. Newer input takes precedence over the summary if
they conflict. Empty, incomplete, and failed summary responses are retried from
unchanged accepted state and are never installed as continuation context.
Provider adapters also classify transient failures so compaction can retry them
with bounded backoff without changing or unwinding state. All usage returned by
these attempts remains billable and is accounted for, including when compaction
ultimately fails.

Provider adapters classify context-limit rejections separately from other
failures. On a summary-request overflow, compaction retries against progressively
shorter candidate states using explicit accepted-turn boundaries. Removed recent
turns are excluded from the summary request, then appended after the summary in
chronological order using their exact provider-native items. Buffered inputs
following those turns are retained separately and restored in their original
position. Opaque reasoning state is therefore preserved for the exact native
tail; only the summarized prefix loses opaque continuity. Accepted-turn
boundaries are remapped to the rebuilt state so later overflow recovery can
unwind those turns again. If an ordinary action request exceeds provider
capacity before the prior response schedules compaction, the harness compacts
the last accepted state while leaving the current inputs pending, then retries
that action request once. A second overflow fails immediately instead of
repeating the same oversized request.

The accepted state is not replaced until compaction succeeds. Reaching the
protected continuation boundary, producing an oversized continuation state, or
exhausting summary-response retries fails closed.

Each harness compaction is written to `compaction_NNN.json` with its summary,
trigger, item counts, attempts, overflow recoveries, excluded-turn counts, and
token usage. The first subsequent `step_NNN.json` also records a safe
`continuation` containing the compaction number, generated summary, and exact
bridge string sent as the new user-role context. Later steps omit this field.
Opaque provider state is not included. The usage is also added to the local run
total. Monetary cost remains provider-reported only in `usage.cost`; the harness
does not write a configured-price estimate into that field.

Published v3 costs are reconstructed downstream from ARC-facing action token
usage and configured input/output prices. A harness compaction's usage is
therefore attributed to the next model-generated action. The action metadata's
top-level usage and calculated cost cover every model request since the previous
action, while `state.harness_compaction` retains the compaction-only usage and
cost breakdown. The local `step_NNN.json` and `compaction_NNN.json` files remain
separate, so `run_meta.json` adds each request exactly once. The
summary-and-bridge structure and completed-turn unwinding are inspired by
[Stirrup](https://github.com/ArtificialAnalysis/Stirrup), which is MIT licensed;
the prompts and exact-tail preservation here are independently adapted and
domain-neutral.

## Recording and provenance

Opaque provider state is never written to ordinary step records, logs, action
metadata, or public artifacts. Step records retain the readable model
output and reasoning summary, sanitized input item types and IDs, and counts
for items sent, compaction items returned, and history size before and after
pruning. Visible output and summaries continue to use the harness's 16k action
metadata fitting.

Run metadata records the adapter ID, strategy, config ID, sanitized settings,
implementation path, adapter version, review status, and harness commit. Set
`ARC_HARNESS_COMMIT_SHA` in the build or runtime environment to produce an
immutable source permalink. The value is `null` when the environment variable
is absent; runtime metadata never depends on `.git` being present in a Docker
image.

## Adding or reviewing a provider

New provider implementations should add a stable ID and descriptor to the local
registry, implement the common turn contract in a provider-specific module, and
add contract and artifact-redaction tests. Configurations may select only local
registered IDs; the harness never downloads or executes code from a URI.

A lab review should identify the exact adapter ID, implementation path, version,
commit, supported strategies, and reviewed request/response behavior. Record
`provider_approved` only when explicit approval is documented. The OpenAI
adapter is currently labeled `provider_reference` because it derives from a
provider reference implementation but has not been separately approved.
