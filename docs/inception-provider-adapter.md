# Inception Provider Adapter

`inception.chat_completions.v1` supports Mercury's Chat Completions responses
containing encrypted `reasoning_details`. It uses the OpenAI Python SDK and
continuous conversation with plain-text ARC actions. The generic
`open_source.chat_completions.v1` adapter continues to reject opaque reasoning.

## Contract and evidence

Live Mercury 3 Preview requests through Baseten on October 29, 2026 returned
assistant `content` plus `reasoning_details` blocks with `type:
reasoning.encrypted`, `format: inception-v1`, integer `index`, and string `data`.
The adapter preserves these blocks without decoding them and replays them on
the corresponding assistant messages. Stream fragments are assembled by index;
metadata must agree, and `data` fragments are concatenated in arrival order.
Unknown block types, formats, fields, or conflicting metadata are rejected.
Non-reasoning responses may omit the field.

The public [Inception API reference](https://docs.inceptionlabs.ai/api-reference/chat/create-a-chat-completion)
documents Chat Completions, streaming, token limits, and reasoning controls, but
does not specify the encrypted replay schema. That schema is based on the
observed Baseten response and live replay tests. Successful replay and recall
do not establish how the server uses the encrypted contents internally. Do not
assume other endpoints or future encrypted formats have the same contract.

Only a single stopped choice with a nonempty final answer can become a candidate
turn. The shared harness commits it only after accepting a valid ARC action.
Failures, incomplete responses, and invalid actions retain the prior state.
Reasoning usage is a breakdown of output tokens, never an additional cost.
Native usage aliases are supported when nested OpenAI-style fields are absent.

## Configuration

The checked-in `inception-mercury-3-preview-high-provider-adapter` profile in
`benchmarking/model_configs.yaml` runs Mercury 3 Preview high through Baseten.
Start a full LS20 run with the normal 5x baseline action budget:

```bash
uv run main.py --game ls20 --config inception-mercury-3-preview-high-provider-adapter
```

The profile uses these provider-adapter settings:

```yaml
- id: inception-mercury-3-preview-high-provider-adapter
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5
    MAX_CONTEXT_LENGTH: 1000000
  runtime:
    sdk: openai-python
    api: chat_completions
    adapter_id: inception.chat_completions.v1
    state: continuous_conversation
    compaction:
      strategy: harness_summary
      trigger_tokens: 175000
  client:
    base_url: https://inference.baseten.co/v1
    api_key_env: BASETEN_API_KEY
  request:
    model: inception/mercury-3-preview
    reasoning_effort: high
    max_tokens: 131072
    stream: true
    stream_options:
      continuous_usage_stats: true
    extra_headers:
      X-Baseten-Experimental-Predict-Timeout-Seconds: "3600"
  pricing:
    input: 0.10
    cached_input: 0.01
    output: 0.35
```

Do not set `reasoning_replay`: the native encrypted replay field is fixed for
this adapter. Streaming injects `include_usage: true`. Sampling parameters and
the compaction headroom use their defaults. The output limit is explicit because
the shared compactor needs it to budget the summary request.

The 1M context ceiling and 131k output limit follow the
[Inception model documentation](https://docs.inceptionlabs.ai/get-started/models).
The full Baseten context ceiling has not been independently exercised. The
175k compaction trigger keeps ordinary history below that configured ceiling.

## Compaction and recordings

The shared `SummaryCompactor` submits completed native history, including its
encrypted reasoning, to the same model using the normal output cap. A completed
summary replaces the older history with a text bridge. The newest pending game
observations remain outside the summary and are restored in order afterward.
Context overflow can unwind recent accepted turns and restore their exact
encrypted blocks after the summary. Failed compaction does not commit new state.

Encrypted blocks remain in the replay state and the harness's private
`state/state_step_*.json` restart snapshots. Loading a snapshot restores the
blocks unchanged. Readable requests, responses, diagnostics, and compaction
artifacts remove their `data` fields; those readable artifacts cannot restore
encrypted replay. Keep the raw restart snapshots private, following the existing
rehydration contract. Compaction is lossy at the summary boundary, as it is for
the other harness-summary adapters.

## Validation

Offline tests cover native replay, fragmented streams, completion validation,
usage, retry isolation, context overflow, compaction, and redaction:

```bash
uv run python -m pytest -q tests/unit/test_inception_runtime.py
```

This opt-in live test makes three paid requests, tests encrypted replay through
forced compaction, and checks pending-input ordering and post-compaction recall:

```bash
RUN_INCEPTION_LIVE_TESTS=1 \
uv run python -m pytest -q tests/integration/test_inception_continuous_conversation_live.py
```

For a bounded local LS20 test, run from the sibling `arc-agi-tools` checkout:

```bash
uv run arc-tools speed test --model inception-mercury-3-preview \
  --benchmark-dir ../arc-agi-3-benchmarking \
  --model-configs ../arc-agi-3-benchmarking/benchmarking/model_configs.yaml \
  --harness provider-adapter --reasoning high --game ls20 \
  --actions 20 --concurrency 1
```

This uses the online game API but runs the agent locally and submits no cloud
benchmark job. A local test does not deploy this adapter: cloud workers must
include it before a cloud config can select `inception.chat_completions.v1`.
