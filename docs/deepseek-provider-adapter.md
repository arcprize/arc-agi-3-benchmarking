# DeepSeek Provider Adapter

`deepseek.chat_completions.v1` implements the DeepSeek Provider Adapter used by
`deepseek-v4-1-flash-low-provider-adapter`. It is intentionally specific to
DeepSeek thinking mode. It is not a generic OpenAI-compatible adapter and it
does not support text-tag reasoning replay.

## Protocol

DeepSeek documents a special rule for thinking mode: historical
`reasoning_content` is replayed only when the request includes `tools`. The
adapter therefore owns the complete tool protocol for every model turn.

For an ARC action request, the adapter:

1. Sends one function in `tools`, named `submit_action`.
2. Omits `tool_choice`, as required by DeepSeek thinking mode.
3. Requires exactly one `submit_action` call in the response.
4. Converts the tool arguments to the harness's existing structured action
   JSON.
5. Stores the exact assistant message, including `reasoning_content` and
   `tool_calls`, followed by a matching `tool` result.
6. Replays that assistant/tool pair before the next game observation.

The resulting request history has this shape:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "...game frame..."},
  {
    "role": "assistant",
    "content": null,
    "reasoning_content": "...",
    "tool_calls": [
      {
        "id": "call_...",
        "type": "function",
        "function": {
          "name": "submit_action",
          "arguments": "{\"action_type\":\"ACTION1\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_...",
    "content": "The action was accepted. The resulting game state will be provided in the next user message."
  },
  {"role": "user", "content": "...next game frame..."}
]
```

This is model output formatting, not general-purpose tool use. The model cannot
call arbitrary external tools. `submit_action` only carries the ARC action the
harness would otherwise parse from text.

## Compaction

The adapter uses the shared harness summary compactor. Summary requests expose
one adapter-owned function named `return_summary`, and require exactly one call.
Pending game observations are excluded from the summary request and restored
unchanged afterward. If the summary request overflows context, complete recent
accepted turns can be temporarily unwound and then restored exactly, including
their `reasoning_content` and tool messages.

After successful compaction, older exact history is replaced by a text summary.
This is intentionally lossy and is separate from DeepSeek's native reasoning
replay within the retained history.

## Configuration

The checked-in profile is:

```yaml
- id: "deepseek-v4-1-flash-low-provider-adapter"
  runtime:
    sdk: "openai-python"
    api: "chat_completions"
    adapter_id: "deepseek.chat_completions.v1"
    state: "continuous_conversation"
    compaction:
      strategy: "harness_summary"
      trigger_tokens: 175_000
      summary_max_output_tokens: 8_192
      summary_input_headroom_tokens: 8_192
```

The adapter rejects config-supplied tools, tool choices, custom history,
structured-output constraints, stop sequences, and provider-native compaction.
It requires one positive `max_tokens` or `max_completion_tokens` value and
`n=1`. `store` must be absent or false.

Run the profile with:

```bash
uv run main.py --config=deepseek-v4-1-flash-low-provider-adapter
```

## Verification

Unit tests use the real OpenAI SDK with mocked HTTP and make no paid calls:

```bash
uv run pytest -q tests/unit/test_deepseek_runtime.py
```

The optional live smoke test makes paid synthetic DeepSeek requests but does
not launch an ARC benchmark:

```bash
RUN_DEEPSEEK_LIVE_TESTS=1 \
uv run pytest -q tests/integration/test_deepseek_continuous_conversation_live.py
```

## Sources

- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- [DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls/)
