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
state from one accepted turn to the next. It is currently implemented only by
`openai.responses.v1`. The OpenAI adapter implements continuous conversation
through the Responses API with `store: false`. It requests
`reasoning.encrypted_content` and sends each accepted user input plus every
native `response.output` item into the following turn.
Replaying only serialized reasoning is not enough: message, tool, and other
native output items can also be part of the model's state.

The adapter removes only two SDK response fields that the input schema rejects:
`status` from reasoning items and `created_by` from compaction items. When a
response contains compaction items, it retains the latest compaction item and
everything after it. Compaction remains a separate request setting under
`request.context_management`; the supplied GPT-5.6 Sol profiles use a 175k
threshold.

The OpenAI continuous-conversation configuration must:

- use the OpenAI Responses API and `runtime.adapter_id: openai.responses.v1`
- set `request.store: false`
- include `reasoning.encrypted_content`
- set `reasoning.context: all_turns` and `reasoning.summary: auto`
- avoid `previous_response_id`, conversations, and background mode

The checked-in profiles cover low, medium, high, xhigh, and max reasoning. The
harness automatically removes the manual carry-forward instruction when the
selected adapter provides continuous conversation. This is not a configurable
model setting. Existing `manual_rolling` configurations keep the instruction
because their visible replies are the state carried across turns.

This flow is ZDR-compatible, but it does not enable Zero Data Retention for an
organization. OpenAI must separately approve and configure ZDR for the
organization. See OpenAI's official documentation for
[stateless Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses#4-decide-when-to-use-statefulness),
[compaction](https://developers.openai.com/api/docs/guides/compaction), and
[Zero Data Retention controls](https://developers.openai.com/api/docs/guides/your-data#zero-data-retention).

## Recording and provenance

Encrypted provider state is never written to ordinary step records, logs,
action metadata, or public artifacts. Step records retain the readable model
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
