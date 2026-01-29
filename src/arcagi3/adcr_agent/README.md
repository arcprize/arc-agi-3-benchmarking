# ADCR (`adcr`) agent

## Goal

Provide a **simple, extensible baseline** “control loop” for ARC-AGI-3 gameplay that:

- learns incrementally by maintaining a small **scratchpad memory**
- selects high-level “human actions” (e.g. Move Up) and then **maps** them to the concrete game API action (`ACTION1..ACTION7`)
- is easy to debug via **breakpoints**

This agent is intentionally “pattern-first”: it’s a reference implementation of a common multi-step prompting pipeline you can copy and adapt.

## How it works (Analyze → Decide → Convert → Review)

Implemented in `src/arcagi3/adcr_agent/agent.py`:

- **Analyze** (`analyze_outcome_step`): looks at the outcome of the previous action and produces an analysis.  
  - If the model includes a `---` divider, everything after it becomes the updated `memory_prompt`.
- **Decide** (`decide_human_action_step`): chooses a *human-level* action from the allowed action descriptions.
- **Convert** (`convert_human_to_game_action_step`): maps the chosen human-level action to a concrete ARC action (`ACTIONn`), respecting `available_actions`.
- **Review (memory update)**: happens as part of Analyze (the memory is updated for the next turn).

The `step()` method wires these sub-steps together and returns a `GameStep` containing:

- `action`: the concrete action dict sent to the ARC API
- `reasoning`: a compact payload (kept small due to ARC API size limits)

## State / datastore contract

ADCR uses `context.datastore` (checkpoint-persisted) with these keys:

- **`memory_prompt`**: `str` – scratchpad memory the model can update
- **`previous_prompt`**: `str` – last prompt text (useful for debugging / breakpointer)
- **`previous_action`**: `dict | None` – prior step’s chosen “human action” JSON

## Breakpoints (interactive debugging UI)

ADCR is fully breakpointer-ready:

- Runtime spec + hooks live in `src/arcagi3/adcr_agent/breakpoints.py`
- The agent registers breakpoints in `__init__()` via `self.register_breakpoints(...)`

Key pause points include:

- `analyze.post` (edit analysis + memory)
- `decide.post` (edit chosen human-action JSON + memory)
- `convert.post` (edit chosen game-action JSON + memory)

To use them:

```bash
python scripts/run_breakpoint_server.py
python -m arcagi3.runner --agent adcr --game_id <GAME_ID> --config <CONFIG> --breakpoints
```

## Prompts

Prompts are loaded relative to the module folder (`src/arcagi3/adcr_agent/prompts/`) via `PromptManager`:

- `system.prompt`
- `analyze_instruct.prompt`
- `action_instruct.prompt`
- `find_action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner --agent adcr --game_id <GAME_ID> --config <CONFIG> --max_actions 40
```

Common runner flags that affect ADCR behavior:

- `--use_vision` (defaults to true in practice; see `src/arcagi3/utils/cli.py`)
- `--memory-limit` (overrides the agent’s memory word limit)
- `--show-images` (terminal visualization)

## When to use / when not to use

- **Use it when**: you want a baseline pipeline to iterate on, or you want breakpoint-friendly debugging.
- **Avoid it when**: you want a single-pass "action-only" policy or an explicit structured reasoning object.


