# `my_agent` starter agent

This folder is a tiny example of how to build your own ARC-AGI-3 agent.

## What matters

An agent in this repo only needs three things:

1. A class that subclasses `MultimodalAgent`
2. A `step(self, context) -> GameStep` method
3. A registry entry so `arcagi3.runner` can find it

## Files in this example

- `agent.py`: the actual agent logic
- `definition.py`: registers the agent under the CLI name `my_agent`
- `README.md`: notes for extending it

## What this starter does

`MyAgent` is intentionally simple:

- reads `context.game.available_actions`
- stores its own counter and history in `context.datastore`
- cycles through the currently available actions
- avoids `ACTION6` unless it is the only option, because clicks need coordinates
- returns a valid `GameStep`

That makes it a good sandbox for learning the harness before adding prompts or model calls.

## The minimum contract

Your `step()` method should return something like:

```python
return GameStep(
    action={"action": "ACTION1"},
    reasoning={"agent": "my_agent"},
)
```

If you choose `ACTION6`, include `x` and `y` in the `0..127` range:

```python
return GameStep(
    action={"action": "ACTION6", "x": 63, "y": 63},
    reasoning={"agent": "my_agent"},
)
```

## Where to read state

The `context` object gives you the live game session:

- `context.game.available_actions`
- `context.game.current_score`
- `context.frames.frame_grids`
- `context.last_frame_grid`
- `context.datastore`

Use `context.datastore` for any state you want to survive checkpoints and resumes. Keep it JSON-serializable.

## How to run it

```bash
uv run python -m arcagi3.runner \
  --agent my_agent \
  --game_id <GAME_ID> \
  --config <CONFIG_NAME> \
  --max_actions 5
```

Notes:

- You still need `ARC_API_KEY` to talk to the ARC server.
- This starter agent does not use `self.provider`, so it will not touch the model provider unless you add model calls yourself.

## How to make it smarter

Once this wiring feels comfortable, the usual next step is:

1. Build a prompt from `context`
2. Call `self.provider.call_with_tracking(context, messages, step_name="decide")`
3. Parse the model output into an action dict
4. Return `GameStep(action=..., reasoning=...)`

If you want a richer example, `src/arcagi3/adcr_agent/` shows a full Analyze -> Decide -> Convert loop with prompts, memory, and breakpoints.
