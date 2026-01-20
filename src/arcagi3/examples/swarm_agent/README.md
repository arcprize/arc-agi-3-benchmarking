# Swarm (`swarm`) prepared agent

## Goal

Coordinate decisions across **multiple games at once** so a single model call can:

- inspect several game contexts
- propose a plan containing actions for one or more games

This is useful for research on batching, cross-game prioritization, and “portfolio” strategies.

## How it works

Implemented in `src/arcagi3/examples/swarm_agent/agent.py`:

- `coordinate_actions()` builds a combined prompt containing a summary for each game:
  - game id, state, score, available actions
  - optional memory (per-game `context.datastore["memory_prompt"]`)
  - observation frames (images if multimodal, otherwise text grids)
- The model returns a JSON “plan” containing a list of action items, typically shaped like:
  - `{"game_id": "...", "action": "ACTIONn", "data": {...}}`
- For the normal single-game runner (`python -m arcagi3.runner ...`), `step()` wraps the single game into a “swarm plan” and executes the first returned action for that game.
- For true multi-game execution, `play_swarm(game_ids, max_rounds=...)` runs a synchronized loop across many games.

## Prompts

Prompts live in `src/arcagi3/examples/swarm_agent/prompts/`:

- `system.prompt`
- `swarm_instruct.prompt` (instructs the model how to emit a multi-game plan)

## How to run (single game via runner)

```bash
python -m arcagi3.runner --agent swarm --game_id <GAME_ID> --config <CONFIG>
```

## How to run (multi-game, programmatic)

The CLI runner is single-game. For multi-game use, call `SwarmAgent.play_swarm(...)` from a small script/runner:

- agent class: `src/arcagi3/examples/swarm_agent/agent.py`
- prepared flags (single-game): `src/arcagi3/examples/swarm_agent/flags.py`


