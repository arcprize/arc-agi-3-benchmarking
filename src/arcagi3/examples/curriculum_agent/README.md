# Curriculum (`curriculum`) prepared agent

## Goal

Encourage **systematic discovery** of game mechanics early, then converge on execution, by following an explicit curriculum:

- **Exploration**: gather evidence, probe the environment, reduce unknowns
- **Refinement**: consolidate what’s been learned, resolve contradictions, reduce uncertainty
- **Execution**: optimize for progress/win using the best current understanding

This is useful when you want “the agent’s *intent*” to change over time, rather than running the same prompt shape every turn.

## How it works

Implemented in `src/arcagi3/examples/curriculum_agent/agent.py`:

- It is ADCR-shaped (Analyze → Decide → Convert), but **injects a curriculum prompt** into the Decide step.
- Phase selection is based on `context.game.play_action_counter`:
  - `< exploration_steps` → exploration
  - `< exploration_steps + refinement_steps` → refinement
  - otherwise → execution

Unlike `adcr`, there’s no breakpoint integration here by default (but the structure is similar enough to add it).

## State / datastore contract

Uses the same “memory scratchpad” pattern as ADCR:

- **`memory_prompt`**: `str`
- **`previous_prompt`**: `str`
- **`previous_action`**: `dict | None`

## Prompts

Prompts live in `src/arcagi3/examples/curriculum_agent/prompts/`:

- `system.prompt`
- `analyze_instruct.prompt`
- `curriculum_instruct.prompt` (phase instructions)
- `action_instruct.prompt`
- `find_action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner --agent curriculum --game_id <GAME_ID> --config <CONFIG>
```

Notes:

- The phase lengths (`exploration_steps`, `refinement_steps`) are constructor parameters, but **not currently exposed as CLI flags**. If you want to tune them from the CLI, copy the pattern in `src/arcagi3/examples/state_transform_adcr/flags.py` and add args + `get_kwargs`.


