# Analysis Mode Plan

## Goal

Add an opt-in **analysis replay mode** for the benchmarking agent where each prior assistant turn can be replayed to the model with its saved reasoning summary inserted inline before the assistant's final action text.

This mode should let a future model call see:

- what the previous assistant said
- the previous assistant's reasoning summary
- the final action text in a clear "reasoning first, action second" layout

Normal runs should remain unchanged unless this mode is explicitly enabled in the model config.

When analysis mode is enabled, the agent should also route to a slightly more helper-oriented
system prompt so the model is explicitly instructed to use the replayed reasoning summaries as
supporting context. When analysis mode is disabled, it should keep the current baseline system
prompt unchanged.

## Current State

The current code is already close to the right shape, but reasoning summaries are recorded and then discarded from future prompt context:

- `BenchmarkingAgent.conversation` is a rolling transcript of plain `{"role", "content"}` messages.
- `BenchmarkingAgent._build_model_request()` converts that transcript directly into `ModelRequest(messages=[Message(...), ...])` using the current system prompt.
- `BenchmarkingAgent.choose_action()` appends only `model_response.output_text` to `self.conversation` for the assistant turn.
- `StepRecord.reasoning` records `model_response.reasoning_text`, but that reasoning text is not replayed into later turns.
- `OpenAIResponsesAdapter` consumes `ModelRequest.messages` and maps the first `system` message to `instructions`, then sends the rest as `input`.
- `normalize_responses_response()` already extracts human-readable reasoning summaries into `ModelResponse.reasoning_text`.

Relevant files:

- `benchmarking/agent.py`
- `benchmarking/runtime_models.py`
- `benchmarking/runtime_adapters.py`
- `benchmarking/recording.py`
- `benchmarking/model_configs.yaml`

## Recommendation

Implement analysis replay as an **agent-level transcript rendering mode**.

Do not change the runtime adapter contract for the first implementation. The runtime should continue to receive `ModelRequest(messages=..., request_config=...)` and should not know whether an assistant message was rendered with extra inline reasoning text or not.

### Config shape

Put the mode under `agent`, not `runtime`, because this is a transcript policy owned by the agent:

```yaml
- id: "openai-gpt-5-4-2026-03-05-responses-analysis"
  agent:
    MAX_ACTIONS_BASELINE_MULTIPLIER: 5.0
    MAX_CONTEXT_LENGTH: 175_000
    analysis_mode: true

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
      effort: "high"
      summary: "auto"

  pricing:
    input: 2.50
    output: 15.00
```

Notes:

- `agent.analysis_mode: true` gates all replay behavior.
- The XML replay format is hard-coded in agent code for the first cut. If this mode needs more options later, `analysis_mode` can be expanded from a boolean to a dict in a separate migration.
- `request.reasoning.summary` still belongs under `request` because it is a provider/API request knob, not an agent policy. For Responses configs, set it when you want `ModelResponse.reasoning_text` to be populated from reasoning summaries.
- Existing non-analysis configs should keep their current behavior and do not need this `agent.analysis_mode` section.

### Conversation storage shape

Keep `self.conversation` as the canonical transcript of plain `{"role", "content"}` messages.

In normal mode, assistant turns stay as plain final-action text:

```python
{"role": "assistant", "content": "MOVE_LEFT"}
```

In analysis mode, assistant turns should store the replay-ready XML block directly in `content`:

```python
{
    "role": "assistant",
    "content": """
<reasoning_summary>
I compared the reachable cells and chose the left move because ...
</reasoning_summary>

MOVE_LEFT
""",
}
```

User and system turns stay unchanged:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
```

Why this shape:

- it preserves the current one-user / one-assistant turn structure
- it keeps all replay text in the assistant message `content` field exactly where the model will see it
- it keeps trimming logic much simpler than inserting a second synthetic assistant message
- it keeps `ModelRequest` and adapter code stable for the first cut

### Outbound prompt rendering

Add an agent-owned helper that formats the **assistant turn content** at append time.

In normal mode:

- assistant turns are appended as plain `model_response.output_text`

In analysis replay mode:

- if `model_response.reasoning_text` is non-empty, the appended assistant turn's `content` is rendered with a hard-coded XML `<reasoning_summary>` block followed by the raw final assistant output text
- if no reasoning summary is available, append the plain action/output text so the turn remains compact and readable

```text
<reasoning_summary>
<saved reasoning summary text>
</reasoning_summary>

<original assistant output text>
```

This format is intentionally simple and explicit. It gives the model a stable distinction between "prior reasoning summary" and "prior final action" while still storing everything in one assistant `content` field.

User turns remain unchanged in both modes.

System turns should be selected by an agent-owned helper based on `analysis_mode`:

- in normal mode, use the current baseline system prompt unchanged
- in analysis mode, use a slightly more helper-oriented system prompt that tells the model to treat
  replayed `<reasoning_summary>` blocks as compact prior-turn context and then produce the next
  action normally

The first implementation can keep this as a hard-coded branch in agent code rather than introducing
a new system-prompt config schema.

### Where to inject this in code

Recommended implementation points:

- Parse `agent.analysis_mode` in `BenchmarkingAgent.__init__()` from the loaded config.
- Add a helper such as `_build_system_prompt()` that returns either the current baseline system
  prompt or the analysis-mode helper prompt depending on `self.analysis_mode`.
- Add a helper such as `_build_assistant_turn_content(output_text, reasoning_text)` that returns either plain action text or the XML replay block depending on the analysis-mode config.
- Change assistant appends in `BenchmarkingAgent.choose_action()` so the canonical assistant turn stores `content = _build_assistant_turn_content(model_response.output_text, model_response.reasoning_text)`.
- Keep `_build_model_request()` mostly unchanged apart from routing the system prompt through
  `_build_system_prompt()`: it can continue to convert `self.conversation` into `Message` objects
  because assistant `content` is already replay-ready in analysis mode.
- In `choose_action()`, save `StepRecord.messages_sent` from the outbound `ModelRequest.messages` that was actually used for the successful attempt, not from `self.conversation` after appending the current assistant turn.

That final logging detail matters: `messages_sent` should reflect the exact transcript sent to the model on that step, while the newly appended current assistant message becomes part of the transcript for the next step.

### Trimming behavior

Keep the existing "trim oldest user/assistant pair" strategy and preserve one assistant turn per user turn.

Do not insert separate synthetic reasoning messages into `self.conversation` for the first cut. A separate message would make `_trim_oldest_turn()` more fragile because it currently assumes the oldest removable turn starts at a `user` message and includes at most the immediately following `assistant` message.

Because replay text is stored directly in assistant `content`, the current context estimator already counts replayed reasoning text as long as it keeps summing `message["content"]`. That makes this approach safer than storing reasoning in a separate sidecar field that token estimation could accidentally ignore.

### Recording and observability

Target behavior for step logs:

- `messages_sent` should show the exact rendered transcript sent to the model on that step.
- Prior assistant turns in analysis mode should visibly include the injected reasoning summary block before the final action.
- `assistant_response` should remain the raw final assistant output for the current step.
- `reasoning` should remain the raw reasoning summary extracted from the current step response.

This gives you both:

- the exact replay prompt that the model saw
- the newly produced reasoning summary and action for the current step

### Non-goals for the first implementation

Do not include these in the first cut:

- `previous_response_id` / API-managed continuation
- native replay of Responses `reasoning` output items
- encrypted reasoning carryover
- multiple synthetic assistant messages per turn
- runtime-adapter awareness of analysis mode

The goal is strictly text-level replay of saved human-readable reasoning summaries as part of agent-owned manual rolling context.

## High-Level Plan

1. Add an agent config section that enables analysis replay mode without affecting current configs.
2. Define the analysis-mode assistant `content` format so prior reasoning summaries are stored inline before prior assistant actions.
3. Add an agent-owned helper that routes the system prompt based on `analysis_mode`, returning the
   baseline prompt in normal mode and a slightly more helper-oriented prompt in analysis mode.
4. Add an agent-owned helper that builds that XML `content` block at append time only when analysis mode is enabled.
5. Make step logs record the exact outbound request transcript while keeping canonical conversation storage stable.
6. Validate that context trimming / token estimation naturally count replayed reasoning text stored in assistant `content`.
7. Add phase-by-phase tests that prove normal configs are unchanged and analysis configs replay reasoning exactly as intended.

## Phased Checklist

### Phase 1: Define Config and Conversation Semantics

Purpose: introduce an explicit agent-owned analysis mode contract before changing request behavior.

- [x] Add `agent.analysis_mode` to the model-config schema as an optional boolean.
- [x] Define default behavior when `analysis_mode` is omitted: normal mode, no reasoning replay.
- [x] Define the canonical assistant turn shape so analysis mode stores the XML replay block directly in `content`.
- [x] Document the intended rendering format for replayed assistant reasoning.
- [x] Define the analysis-mode system prompt branch: baseline prompt when `analysis_mode` is false,
  helper-oriented replay prompt when `analysis_mode` is true.
- [x] Decide whether analysis mode is introduced through a new dedicated config ID or by editing an existing Responses config. Recommended: new dedicated config IDs only.

Tests for Phase 1:

- [x] Configs without `agent.analysis_mode` still load and behave as normal mode.
- [x] A config with `agent.analysis_mode: true` loads successfully.
- [x] A config with invalid `agent.analysis_mode` shape fails with a clear config error.
- [x] A canonical assistant turn with XML replay content can be represented without breaking existing user/system turn storage.

Exit criteria:

- [x] The config contract and conversation storage contract are explicit and reviewed.
- [x] No runtime behavior has changed yet for existing configs.

### Phase 2: Add Assistant Turn Formatting for Analysis Replay

Purpose: make analysis-mode assistant turns persist replayed reasoning inline in `content` while preserving normal mode behavior.

- [x] Add an agent helper that formats assistant `content` from `output_text` and `reasoning_text`.
- [x] In normal mode, append assistant turns exactly as plain `output_text`.
- [x] In analysis mode, append assistant turns with non-empty reasoning summaries as:
  - reasoning summary block first
  - final action block second
- [x] Use a hard-coded XML `<reasoning_summary>` block followed by the raw final action text, with no nested analysis-mode config dict.
- [x] Keep user turns unchanged in both modes, preserve the baseline system prompt in normal mode,
  and route the helper-oriented system prompt only in analysis mode.
- [x] Keep the runtime adapter contract unchanged.
- [x] Route the outbound system prompt through an agent helper that selects the baseline or
  analysis-mode helper prompt based on `analysis_mode`.

Tests for Phase 2:

- [x] Normal mode appends assistant turns as the exact same plain content as today.
- [x] Analysis mode appends assistant turns with reasoning inline before the final action inside XML tags.
- [x] Analysis mode appends an assistant turn with missing/empty reasoning as plain content.
- [x] User turns are unchanged in analysis mode, the system prompt uses the helper branch in
  analysis mode, and normal mode keeps the baseline system prompt.
- [x] Responses adapter still receives the expected `instructions` and `input` shape after the rendered transcript is built.
- [x] Chat Completions adapter still receives the expected `messages` shape after the rendered transcript is built.
- [x] In analysis mode, the outbound system message uses the helper-oriented replay prompt.
- [x] In normal mode, the outbound system message remains the current baseline prompt.

Exit criteria:

- [x] The model can be sent replayed reasoning summaries in analysis mode.
- [x] Existing adapter APIs do not need to know that analysis mode exists.

### Phase 3: Persist Replay-Ready Assistant Content

Purpose: capture each turn's reasoning summary directly in the assistant message `content` that will be replayed on later turns.

- [x] Change assistant-turn appends so `self.conversation` stores XML replay content directly in assistant `content` when analysis mode is enabled.
- [x] Keep user/system message storage unchanged.
- [x] Ensure the current assistant action parser still runs against raw `model_response.output_text`, not the XML-wrapped historical replay content.
- [x] Decide whether to backfill replay formatting only for newly generated turns or also transform pre-existing loaded transcripts. For the current codebase, newly generated turns only is likely enough because conversation state is in-memory during one run.

Tests for Phase 3:

- [x] After one successful model call, the appended assistant turn stores `content = output_text`.
- [x] After one successful analysis-mode model call with a reasoning summary, the appended assistant turn stores XML replay content with reasoning first and action second.
- [x] After one successful analysis-mode model call without a reasoning summary, the appended assistant turn stores plain output text and still works.
- [x] Action parsing still runs against the current response output, not against the replay-rendered historical block.

Exit criteria:

- [x] Prior assistant reasoning summaries are persisted in conversation state for replay on later turns.
- [x] Current action parsing and ARC stepping behavior remain unchanged.

### Phase 4: Make Step Logging Reflect the Rendered Outbound Transcript

Purpose: ensure recordings show exactly what the model saw, including replayed reasoning blocks.

- [x] Build the outbound `ModelRequest` once per attempt from the rendered transcript.
- [x] Save `StepRecord.messages_sent` from the rendered request messages, not from raw `self.conversation`.
- [x] Keep `StepRecord.assistant_response` as the current raw assistant output.
- [x] Keep `StepRecord.reasoning` as the current raw reasoning summary.
- [x] Verify that in analysis mode, the recorded transcript shows prior reasoning summaries before prior assistant actions.

Tests for Phase 4:

- [x] In normal mode, `messages_sent` in a step record matches the current plain transcript behavior.
- [x] In analysis mode, `messages_sent` includes a prior assistant reasoning summary block before the prior assistant action.
- [x] In analysis mode, the current step's `assistant_response` remains raw action/output text and is not mutated into the replay format.
- [x] In analysis mode, the current step's `reasoning` remains the extracted summary for that response.
- [x] Step records still serialize successfully when assistant messages in `messages_sent` include replayed reasoning text.

Exit criteria:

- [x] Recordings provide a faithful audit trail of the prompt text sent to the model in analysis mode.

### Phase 5: Validate Context Trimming and Token Estimation with Replay Content

Purpose: prove that replayed reasoning text stored in assistant `content` is counted and trimmed safely.

- [x] Confirm `_estimate_conversation_tokens()` counts XML replay content because it is stored directly in `message["content"]`.
- [x] Keep existing trimming semantics: remove the oldest user/assistant pair and preserve the system prompt.
- [x] Verify that XML replay content on assistant turns is removed when the containing assistant turn is trimmed.
- [x] Confirm there is no need to rewrite `_trim_oldest_turn()` for the first cut because no separate synthetic reasoning messages are introduced.

Tests for Phase 5:

- [x] Normal mode token estimation remains unchanged for plain transcripts.
- [x] Analysis mode token estimation includes replayed XML reasoning text stored in assistant `content`.
- [x] When analysis mode pushes the rendered transcript over the context limit, trimming removes the oldest user/assistant pair and eventually brings the rendered request under the limit.
- [x] Trimming removes an assistant turn's XML replay content together with that assistant turn.
- [x] Trimming still preserves the system prompt and current user turn.

Exit criteria:

- [x] Analysis replay cannot silently bypass context limits through replayed reasoning text that is not counted.

### Phase 6: Add Integration and Regression Coverage

Purpose: prove that analysis mode is opt-in, replay works end to end, and normal configs are unaffected.

- [x] Add a dedicated analysis-mode model config ID, likely a Responses config first.
- [x] Add an integration-style unit test with a fake adapter/model response sequence where turn 2 receives turn 1's replayed reasoning summary inline.
- [x] Add a regression test proving an existing non-analysis config still sends plain assistant messages.
- [x] Add serialization checks for recordings produced in analysis mode.
- [x] Update docs/examples after the behavior is reviewed.

Tests for Phase 6:

- [x] A dedicated analysis config sends plain turn 1 output as the current raw response, appends turn 1 XML replay content to `self.conversation`, and sends turn 2 with turn 1 reasoning replayed inline before turn 1's final action.
- [x] A non-analysis config with identical model/runtime settings still sends plain prior assistant content only.
- [x] Responses-path analysis mode and Chat Completions-path analysis mode both behave correctly if both are enabled through separate configs.
- [x] Recordings from analysis mode contain the rendered prior reasoning block in `messages_sent`.
- [x] Existing phase 3/4/5 regression tests still pass for normal configs.

Exit criteria:

- [x] Analysis mode is available through an explicit config ID.
- [x] Normal mode behavior is preserved.
- [x] End-to-end replay and logging behavior are covered by tests.

## Recommended Implementation Order

1. Add config parsing and defaults for `agent.analysis_mode`.
2. Add a helper that selects the system prompt based on `analysis_mode`.
3. Add a helper that formats analysis-mode assistant `content` from `output_text` and `reasoning_text`.
4. Append replay-ready XML assistant content to `self.conversation` only when analysis mode is enabled.
5. Save rendered request messages in `StepRecord.messages_sent`.
6. Validate that context estimation and trimming naturally include replayed XML content.
7. Add dedicated analysis-mode configs and phase-specific tests.

## Minimum Success Definition

The implementation is done when all of the following are true:

- analysis replay is opt-in under `agent.analysis_mode`
- system prompt routing is opt-in under `agent.analysis_mode`, with baseline prompting preserved in
  normal mode and a helper-oriented replay prompt used in analysis mode
- normal configs produce the same plain transcripts as today
- analysis-mode assistant turns store prior reasoning summaries inline in assistant `content` before prior assistant action text
- step recordings show the exact rendered transcript sent to the model
- context trimming counts replayed reasoning text and removes old turns safely
- tests cover both analysis and non-analysis behavior
