# Hypothesis (`hypothesis`) prepared agent

## Goal

Make the agent’s “science loop” explicit by maintaining **hypotheses** and proposing **experiments** to test them, rather than only keeping an unstructured scratchpad.

This agent is aiming to:

- accumulate candidate hypotheses about mechanics (“what causes score to change”, “what state transitions mean”, etc.)
- mark hypotheses as supported / contradicted as evidence accumulates
- keep an “active hypothesis” and “active experiment” to focus the next few actions

## How it works

Implemented in `src/arcagi3/examples/hypothesis_agent/agent.py`:

- Each turn still follows the ADCR-style loop (Analyze → Decide → Convert).
- Additionally, it periodically runs a **hypothesis update step**:
  - `update_hypotheses_step()` summarizes the last `hypothesis_window` actions and asks the model to output updated hypotheses + a focus hypothesis/experiment.
  - This is triggered every `hypothesis_interval` actions via `_maybe_update_hypotheses()`.
- The Decide step includes:
  - the memory scratchpad
  - a formatted list of hypotheses
  - the “active hypothesis / experiment” focus block

## State / datastore contract

HypothesisAgent stores a richer structure in `context.datastore`:

- **`memory_prompt`**: `str`
- **`previous_prompt`**: `str`
- **`previous_action`**: `dict | None`
- **`hypotheses`**: `list[dict]` where each item is roughly:
  - `{"hypothesis": str, "status": str, "evidence": str}`
- **`active_hypothesis`**: `str`
- **`active_experiment`**: `str`
- **`hypothesis_last_extracted_action`**: `int` (internal counter to rate-limit extraction)

All of these are checkpoint-persisted, so hypothesis state survives resume.

## Prompts

Prompts live in `src/arcagi3/examples/hypothesis_agent/prompts/`:

- `system.prompt`
- `analyze_instruct.prompt`
- `hypothesis_instruct.prompt` (the structured extraction/update step)
- `action_instruct.prompt`
- `find_action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner --agent hypothesis --game_id <GAME_ID> --config <CONFIG>
```

Notes:

- `hypothesis_interval` / `hypothesis_window` are constructor parameters but **not exposed as runner CLI flags** today. If you want them configurable, add args in `src/arcagi3/examples/hypothesis_agent/flags.py` (same pattern as `knowitall` and `state-transform`).


