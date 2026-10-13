# Open-source Provider Adapter

`open_source.chat_completions.v1` provides opt-in continuous conversation for
OpenAI-compatible endpoints serving open-weight models. The initial supported
profiles are GLM-5.3-Flash and Qwen3.8-27B. The adapter keeps the existing ARC
action parser and uses plain-text actions; it does not invent tool calls.

The implementation is separate from `deepseek.chat_completions.v1`. DeepSeek's
documented thinking protocol requires tools for reasoning replay, while GLM and
Qwen document native reasoning replay without tools.

## Runtime contract

The adapter:

- accepts exactly one completed plain-text Chat Completions choice
- captures `reasoning_content` and `reasoning` from streamed or non-streamed
  responses and rejects conflicting nonempty values
- replays the exact reasoning string in the configured native field or fields
- keeps state provisional until the common harness accepts a valid ARC action
- keeps pending observations outside summary requests
- uses the shared provider-neutral summary compactor
- requires streaming usage so cost and compaction triggers remain accurate
- classifies streamed and non-streamed context, connection, rate-limit, timeout,
  and server errors before generic invalid-response handling

Tool calls, structured outputs, custom stop sequences, provider-native
compaction, opaque reasoning, and caller-owned message history are outside this
adapter's contract.

## Reasoning replay modes

Set `runtime.reasoning_replay` explicitly:

- `reasoning_content`: replay only `reasoning_content`; use for GLM-5.3-Flash.
- `reasoning`: replay only `reasoning`; use for endpoints that document that
  field exclusively.
- `reasoning_aliases`: replay the same exact string in both
  `reasoning_content` and `reasoning`; use for Qwen3.8 endpoints following the
  official model-card example.

The adapter never converts reasoning into `<previous_reasoning>` text. Native
field acceptance alone does not prove a server consumed the field, so each
model/endpoint combination still needs a two-turn recall check.

## GLM-5.3-Flash

The checked-in `zai-glm-5-3-flash-low-provider-adapter` profile uses Baseten,
low reasoning, and `reasoning_content`. It inherits GLM's sampling and
`clear_thinking: false` defaults. Z.ai documents preserved thinking as replaying
complete, unmodified reasoning in original order. The selected server and chat
template must expose and consume the same field.

```bash
uv run main.py --game=ls20 --config=zai-glm-5-3-flash-low-provider-adapter
```

## Qwen3.8-27B

The checked-in `alibaba-qwen3-8-27b-low-provider-adapter` profile uses the
dedicated Baseten deployment and `BASETEN_API_KEY`. Tools are optional and the
profile does not enable them. A different OpenAI-compatible deployment must
expose Qwen's reasoning fields; for vLLM, configure `--reasoning-parser qwen3`.

Qwen's official model card enables thinking and `preserve_thinking` by default,
shows historical assistant reasoning under both `reasoning_content` and
`reasoning`, and provides the thinking-mode sampling defaults in the model's
generation configuration. The profile inherits those model defaults and the
server's neutral sampling defaults, overriding only the reasoning effort to
low. Its native context is 262k, so the summary trigger is lower than the GLM
profile. The dedicated endpoint can occasionally stop after emitting reasoning
without final answer text, so this profile allows eight total attempts before
failing a turn.

Run the checked-in Baseten profile with:

```bash
uv run main.py --game=ls20 --config=alibaba-qwen3-8-27b-low-provider-adapter
```

## Compaction

The adapter reuses `SummaryCompactor`. Before the configured trigger, each
accepted assistant answer and exact native reasoning field is replayed. At the
trigger, the same model summarizes completed history. Only after a completed,
nonempty summary does the adapter replace the old prefix with a text bridge.

Pending observations are excluded from the summary and restored unchanged.
When a summary request exceeds context, complete recent accepted turns are
temporarily removed, the older prefix is summarized, and those exact recent
turns are restored. Failed compaction does not mutate accepted state.

Compacted history is intentionally lossy. The provider's exact reasoning is
preserved only before that compaction boundary.

## Verification

Mocked tests make no paid calls:

```bash
uv run pytest -q tests/unit/test_open_source_runtime.py
```

The optional synthetic live test makes three paid model requests but does not
launch an ARC game:

```bash
RUN_OPEN_SOURCE_LIVE_TESTS=1 \
OPEN_SOURCE_LIVE_CONFIG=zai-glm-5-3-flash-low-provider-adapter \
uv run pytest -q tests/integration/test_open_source_continuous_conversation_live.py
```

## Sources

- [Z.ai thinking mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)
- [vLLM Qwen3.8-27B recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml)
