# xAI Provider Adapter

`xai.responses.v1` adds opt-in client-managed native Responses state to the
Provider Adapter harness. It uses `openai-python` against xAI, not the xAI SDK
and not the OpenAI-specific continuous-conversation adapter. The checked-in
profile is `xai-grok-4-6-low-provider-adapter`; set `XAI_API_KEY` to use it.
Existing xAI Chat Completions profiles and other providers are unchanged.

## Example configuration

This complete example matches the profile in `benchmarking/model_configs.yaml`:

```yaml
- id: "xai-grok-4-6-low-provider-adapter"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5.0
    MAX_CONTEXT_LENGTH: 500_000
  runtime:
    sdk: "openai-python"
    api: "responses"
    adapter_id: "xai.responses.v1"
    state: "continuous_conversation"
    compaction:
      strategy: "native"
      trigger_tokens: 175_000
  client:
    base_url: "https://api.x.ai/v1"
    api_key_env: "XAI_API_KEY"
  request:
    model: "grok-4.6"
    max_output_tokens: 128_000
    store: false
    reasoning:
      effort: "low"
    include:
      - "reasoning.encrypted_content"
  pricing:
    input: 2.00
    output: 6.00
```

## Request and replay contract

- Select `sdk: openai-python`, `api: responses`, `state: continuous_conversation`,
  and the explicit `adapter_id: xai.responses.v1`. Without the explicit ID, the
  legacy Responses mapping still selects OpenAI; routing is not inferred from
  hostnames or model names.
- Action requests require `store: false` and
  `include: [reasoning.encrypted_content]`. The system prompt starts the native
  input window. Accepted output items, including opaque encrypted reasoning,
  are replayed in their original order without reconstructing them from text.
  SDK serialization excludes unset defaults and retains returned fields.
- `reasoning.effort` is provider/model-specific; the checked-in profile uses
  `low`. OpenAI's `reasoning.context` and inline `context_management` are not
  sent. A readable reasoning summary is recorded when returned, not fabricated
  or treated as a replacement for the encrypted replay item.
- Only completed responses with reusable native items and visible text become
  candidate state. The agent accepts that state only after parsing a valid game
  action. Rejected, empty, incomplete, refused, or tool-bearing responses do not
  enter subsequent history. Returned reasoning items must have encrypted state.
- This adapter supports non-streaming text action requests. Streaming, tools,
  background execution, automatic truncation, server conversation IDs,
  request-supplied input/instructions, and extra-body/query overrides are
  rejected rather than silently changing state ownership. Custom endpoints can
  be explicitly configured; defaults are `https://api.x.ai/v1` and `XAI_API_KEY`.

## Native compaction

Compaction is optional: omit `runtime.compaction` to disable it. When enabled,
the adapter compares the previous accepted action response's total tokens with
the trigger. At the next turn it calls `POST /v1/responses/compact` using only
`model` and the accepted native prefix. The first call includes the system
prompt; subsequent compactions can include an earlier compaction item.

The current frame and any buffered observations after the last accepted turn
are excluded from compaction. They are appended unchanged, in order, after the
returned compaction output. The complete returned output replaces the old
prefix; the adapter requires the documented single nonempty encrypted
compaction item and never selects, prunes, or edits its blob. It does not
duplicate the system prompt outside the compacted prefix.

Compaction plus the next action request is transactional. Both remain
provisional until the action is accepted. If either request fails or the action
cannot be parsed, retries start from the last accepted state, including its
pending inputs; a successful but uncommitted compaction may therefore be billed
again. There is no lossy manual-rolling or harness-summary fallback.

The trigger is bookkeeping, not a hard context limit or a token-count preflight.
The profile uses a 175k trigger, a 128k action output cap, and a 500k context
limit. Validation requires the trigger plus output allowance to be below the
configured context limit. Large new inputs can still overflow, and compaction
cannot rescue a prefix that already exceeds the provider's context capacity.
Provider failures use the existing bounded action retry budget without pruning
accepted state.

## Accounting and artifacts

- All returned usage from compaction, actions, and rejected attempts contributes
  to the eventual action and local run totals. If retries are exhausted, billed
  usage is still written to the local run metadata. Transport failures without
  returned usage cannot be priced from token counts.
- Compaction usage is separate in `state.native_compaction.usage` while
  top-level action usage includes it exactly once. The breakdown accumulates
  returned compaction usage across all attempts for the accepted action,
  including rejected compaction responses and compactions followed by rejected
  actions or transport failures. It resets for the next action without
  committing rejected conversation state. The next trigger uses action-context
  tokens alone, not compaction costs or accumulated retry usage.
- `run_meta.json` records the validated compaction strategy, trigger, and context
  limit. Native compaction does not create harness-summary compaction files.
- Readable requests show a placeholder for opaque compacted history, followed
  by the actual remaining messages. Old frames do not reappear after compaction.
  Encrypted values remain in memory, not ordinary recordings, action metadata,
  diagnostics, serialization warnings, or transport-error logs.
- Provider-reported cost is normalized from `usage.cost_in_usd_ticks` to
  dollars by dividing by `10_000_000_000`. Integer ticks take precedence,
  including zero; missing, null, or non-integer ticks fall back to `usage.cost`
  for custom endpoints. This applies to actions, compactions, and rejected
  responses. If neither cost field is reported, the normalized zero is not
  evidence that the request was free.
- The profile's flat input/output prices are short-context estimates. They do
  not model caching discounts or xAI's long-context pricing tiers. Actual
  provider-reported cost remains distinct from configured-price estimates.

Client-managed replay and `store: false` do not establish provider-side zero
data retention. The adapter is marked `unreviewed`, not provider-approved.
This implementation adds neither persistent native-state checkpoints nor resume.

## Verification

Unit tests use the pinned OpenAI SDK with mocked HTTP to exercise both real SDK
endpoints, exact replay, repeated compaction, pending/reset ordering, response
validation, retry isolation, native billed cost, retry-wide compaction usage,
configuration, and artifact redaction. Accounting tests also verify persisted
step and run totals and usage retained after retries are exhausted.
Mocked acceptance is not live provider verification or a benchmark result.

Two paid tests are skipped unless explicitly enabled:

```bash
RUN_XAI_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_xai_continuous_conversation_live.py::test_xai_two_turn_native_replay_live

RUN_XAI_COMPACTION_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_xai_continuous_conversation_live.py::test_xai_repeated_native_compaction_live
```

These use Grok 4.6 at low effort with a 4k action output cap. The second test
lowers the trigger to one token on a small synthetic history and checks recall
through two native compactions; the compaction endpoint has no output-cap
parameter. They do not launch ARC games.

## Provider references

- [Responses and encrypted reasoning replay](https://docs.x.ai/developers/model-capabilities/text/generate-text)
- [Native context compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)
- [Responses API reference](https://docs.x.ai/developers/rest-api-reference/inference/responses)
- [Grok 4.6 and reasoning effort](https://docs.x.ai/developers/grok-4-6)
- [Pricing](https://docs.x.ai/developers/pricing)
- [Actual per-request cost](https://docs.x.ai/developers/cost-tracking)
