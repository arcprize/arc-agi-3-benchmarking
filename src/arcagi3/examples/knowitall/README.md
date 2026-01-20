# KnowItAll (`knowitall`) prepared agent

## Goal

Provide a “rules-known” baseline: the agent is given the **full game rules up front** and then only needs to choose the best action each turn.

This agent is intentionally *not* trying to discover mechanics. It’s useful for:

- benchmarking the “execution” part of gameplay if you already have good rules text
- sanity-checking that your prompt/action schema can play a game at all

## How it works

Implemented in `src/arcagi3/examples/knowitall/agent.py`:

- Each turn is a single model call:
  - input: (rules text) + (optional truncated memory) + (available actions) + (current observation)
  - output: a concrete action (`ACTIONn`) plus optional `data`
- It validates that the returned `ACTIONn` is allowed by `available_actions`.
- It does **not** update memory; it only reads `context.datastore["memory_prompt"]` if you populate it externally.

## CLI requirement: rules text

The prepared flags in `src/arcagi3/examples/knowitall/flags.py` require:

- `--game-rules "<RULES TEXT>"`

If `--game-rules` is empty, the runner raises an error.

## Prompts

Prompts live in `src/arcagi3/examples/knowitall/prompts/`:

- `system.prompt`
- `action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner \
  --agent knowitall \
  --game_id <GAME_ID> \
  --config <CONFIG> \
  --game-rules "Put your full game rules text here"
```


