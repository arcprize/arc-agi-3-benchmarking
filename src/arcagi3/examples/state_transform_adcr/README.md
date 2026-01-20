# StateTransform ADCR (`state-transform`) prepared agent

## Goal

Make it easy to experiment with **alternative state representations** by inserting a *state transform* step ahead of the usual ADCR prompting stages.

This agent is trying to answer:

- “What if the model sees a different representation of the same frames?”
- “Can we precompute useful features / summaries and feed those instead of raw grids/images?”

## How it works

Implemented in `src/arcagi3/examples/state_transform_adcr/agent.py`:

- On each turn, it runs a user-supplied `state_transform(context) -> StateTransformPayload`.
- The resulting payload (text and/or images) is then injected into:
  - **Analyze**
  - **Decide**
  - **Convert**
- This effectively replaces the “usual” frame blocks the model would have seen in ADCR, letting you control the observation channel without changing the rest of the pipeline.

`StateTransformPayload` supports:

- `text`: optional string representation
- `images`: optional list of images (ignored if model isn’t multimodal)
- `label`: prompt heading label

## CLI presets

The prepared runner exposes a small set of presets in `src/arcagi3/examples/state_transform_adcr/flags.py`:

- **`--state-transform basic`**: includes both text grids and images
- **`--state-transform text-only`**: includes only text grids (no images)

## Prompts

Prompts live in `src/arcagi3/examples/state_transform_adcr/prompts/`:

- `system.prompt`
- `analyze_instruct.prompt`
- `action_instruct.prompt`
- `find_action_instruct.prompt`

## How to run

```bash
python -m arcagi3.runner --agent state-transform --state-transform basic --game_id <GAME_ID> --config <CONFIG>
```

## Extending with your own transform

The prepared agent only exposes two presets. For a custom transform you typically:

- create a small custom runner (see root `README.md` “Making your own runner”)
- add a custom CLI flag that selects your transform
- pass it via `get_kwargs` as `{"state_transform": my_transform}`


