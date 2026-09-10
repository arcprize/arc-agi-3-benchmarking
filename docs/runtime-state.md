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
`openai.responses.v1` and `google.interactions.v1`. The OpenAI adapter uses the
Responses API with `store: false`. It requests
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
they conflict. Empty summaries are retried from unchanged accepted state.
Provider adapters classify context-limit rejections separately from other
failures. On a context overflow, compaction retries against progressively
shorter candidate states using explicit accepted-turn boundaries. Removed
recent turns are excluded from the summary request, then appended after the
summary in chronological order using their exact provider-native items. Their
opaque reasoning state is therefore preserved; only the summarized prefix loses
opaque continuity. Accepted-turn boundaries are remapped to the rebuilt state so
later overflow recovery can unwind those turns again. The accepted state is not
replaced until compaction succeeds. Reaching the protected continuation
boundary, producing an oversized continuation state, or exhausting empty summary
retries fails closed.

Each harness compaction is written to `compaction_NNN.json` with its summary,
trigger, item counts, attempts, overflow recoveries, excluded-turn counts, and
token usage. The usage is also added to the local run total. Monetary cost
remains provider-reported only in `usage.cost`; the harness does not write a
configured-price estimate into that field.

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
