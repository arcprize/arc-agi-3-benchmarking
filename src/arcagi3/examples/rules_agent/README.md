# Rules (`rules`) prepared agent

## Goal

Continuously turn raw interaction history into a **growing “rules so far” model** and use it to guide action selection.

Compared to a plain scratchpad, this agent tries to keep the learned knowledge in a more reusable form:

- a short **rules summary**
- a concrete **list of rules**
- a short list of **experiments to run next**

## How it works

Implemented in `src/arcagi3/examples/rules_agent/agent.py`:

- Each turn follows an ADCR-style loop (Analyze → Decide → Convert).
- Periodically (every `rules_interval` actions), it runs a **rules extraction step**:
  - `extract_rules_step()` summarizes the last `rules_window` actions and asks the model to output:
    - `rules_summary`
    - `rules` (list)
    - `experiments` (list)
  - `_maybe_extract_rules()` rate-limits this update using `rules_last_extracted_action`.
- The Decide step injects the current rules summary/list so the model can select actions consistent with the best-known mechanics.

## State / datastore contract

RulesAgent stores these keys in `context.datastore`:

- **`memory_prompt`**: `str`
- **`previous_prompt`**: `str`
- **`previous_action`**: `dict | None`
- **`rules_summary`**: `str`
- **`rules_list`**: `list[str]`
- **`rules_experiments`**: `list[str]`
- **`rules_last_extracted_action`**: `int`

## Prompts

Prompts live in `src/arcagi3/examples/rules_agent/prompts/`:

- `system.prompt`
- `analyze_instruct.prompt`
- `rules_instruct.prompt` (structured rules extraction/update)
- `action_instruct.prompt`
- `find_action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner --agent rules --game_id <GAME_ID> --config <CONFIG>
```

Notes:

- `rules_interval` / `rules_window` are constructor parameters but **not exposed as runner CLI flags** today. If you want them configurable, add args + `get_kwargs` in `src/arcagi3/examples/rules_agent/flags.py`.


