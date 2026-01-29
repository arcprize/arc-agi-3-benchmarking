# Creating your own ARC-AGI-3 agent

This repo is a harness for running **agents** against **ARC-AGI-3** games. You bring an agent policy (your code), we handle the boring parts:

- logging and tracking (checkpoints, scorecards)
- retries + orchestration
- checkpointing + resuming
- cost / token accounting
- authentication and authorization between ARC and model providers

If you want to build your own agent, you mainly need to understand:

- `MultimodalAgent` (the base class you expand upon)
- `SessionContext` (what your `step()` sees and can mutate)
- the `GameStep` contract (what your `step()` must return)
- how the runner registers and launches agents

Do these things and you can plug right into the benchmarking harness to test out your own agent designs!

---

## Quick start: build a tiny agent and run it

### 1) Create an agent file

Create a new package for your agent. For example:

```
src/arcagi3/my_agent/
  __init__.py
  agent.py
  definition.py
  prompts/            (optional)
    system.prompt
```

Here’s a minimal agent that always moves up:

```python
# src/arcagi3/my_agent/agent.py
from __future__ import annotations

from arcagi3.agent import MultimodalAgent
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext


class MyAgent(MultimodalAgent):
    def step(self, context: SessionContext) -> GameStep:
        # SessionContext is your read/write interface to the current game state
        # (frames, score, available actions) plus a persistent datastore.
        return GameStep(
            action={"action": "ACTION1"},
            reasoning={"agent": "my_agent", "note": "always ACTION1"},
        )
```

That’s enough to make an agent. Now - how do we "plug it in" to the harness so we can run our doomed-to-fail agent?

### 2) `register` your agent with the CLI runner

The CLI runner (`uv run python -m arcagi3.runner`) uses a small registry of agents. You can use `uv run python -m arcagi3.runner --list-agents` to see the list of agents; by default we only provide one at the moment.

How do we get our new agent onto that list? Simple — we `register` our agent by creating a `definition.py` file in your agent package that exports an agent `definition`:

```python
# src/arcagi3/my_agent/definition.py
from __future__ import annotations

from arcagi3.my_agent.agent import MyAgent

definition = {
    "name": "my_agent",
    "description": "Example minimal agent",
    "agent_class": MyAgent,
}
```

This `definition` dict is the **agent definition record**. The agent becomes runnable from the CLI once it’s added to the runner’s registry.

Add it to the default registry in `src/arcagi3/runner.py` (next to the existing `adcr` agent):

```python
from arcagi3.adcr_agent.definition import agents as adcr_definition
from arcagi3.my_agent.definition import definition as my_agent_definition

def _build_default_registry() -> AgentRunner:
    runner = AgentRunner()
    runner.register(adcr_definition)
    runner.register(my_agent_definition)
    return runner
```

And now you can run it! But what if you have some config options that you want to pass to your agent?

### 2.1) Add agent-specific CLI configuration (optional, but common)

You can optionally add runner-exposed configuration. Agent definitions can provide some parser options (via `argparse.ArgumentParser`):

- `add_args(parser)`: add your custom CLI flags
- `get_kwargs(args)`: translate parsed args into kwargs passed to your `agent_class`

Let's look at an example that will shed some light:

```python
# src/arcagi3/my_agent/definition.py
from __future__ import annotations

from arcagi3.my_agent.agent import MyAgent


def add_args(parser) -> None:
    parser.add_argument("--my-setting", type=int, default=123, help="My agent tuning setting")


def get_kwargs(args):
    return {"my_setting": args.my_setting}


definition = {
    "name": "my_agent",
    "description": "Example minimal agent",
    "agent_class": MyAgent,
    "add_args": add_args,
    "get_kwargs": get_kwargs,
}
```

### 3) Run it

Use the same flow as the main `README.md`:

```bash
uv run python -m arcagi3.runner --list-games
uv run python -m arcagi3.runner --list-models
```

Then:

```bash
uv run python -m arcagi3.runner \
  --agent my_agent \
  --game_id <GAME_ID> \
  --config <CONFIG> \
  --max_actions 10
```

If you had a custom setting you added:

```bash
uv run python -m arcagi3.runner \
  --agent my_agent \
  --game_id <GAME_ID> \
  --config <CONFIG> \
  --my-setting 456
```

### 3.1) Alternative: run without touching `arcagi3.runner` (programmatic)

If you don’t want to modify the CLI registry, you can run via `ARC3Tester` directly:

```python
from arcagi3.arc3tester import ARC3Tester
from arcagi3.my_agent.agent import MyAgent

tester = ARC3Tester(
    config="gpt-5-2-openrouter",
    max_actions=10,
    agent_class=MyAgent,
    agent_kwargs={"my_knob": 123},
)

result = tester.play_game("ls20")
print(result)
```

This is useful if you're doing some custom orchestration.

### 4) Checkpoints (how to resume)

By default, checkpoints are written under:

```
.checkpoint/<CARD_ID>/
```

To list and resume:

```bash
uv run python -m arcagi3.runner --list-checkpoints
uv run python -m arcagi3.runner --checkpoint <CARD_ID>
```

At this point you can build real behavior. The rest of this doc explains the moving pieces in detail, like how do we handle state? What is the SessionContext? What is the GameAction?

---

## The moving pieces (mental model)

Let's focus first on understanding what is happening behind the scenes to better understand how our code is being executed.

### What's happening at runtime?

At runtime the stack looks like:

- `uv run python -m arcagi3.runner`
  - builds an `ARC3Tester`
  - instantiates your `agent_class` (a `MultimodalAgent` subclass)
  - calls `agent.play_game(...)`
    - resets/continues game sessions
    - repeatedly calls your `agent.step(context)`
    - executes returned actions via `GameClient.execute_action(...)`
    - saves checkpoints periodically

### What you implement

You implement **one method**:

- `MyAgent.step(context: SessionContext) -> GameStep`

Everything else (game loop, counters, scorecards, retries, checkpoint I/O) is harness code that handles the heavy lifting for you.

## Implementing `step()` correctly

### The `GameStep` contract

Your `step()` must return a `GameStep` with:

- **`action`**: a dict that **must** contain at least an `"action"` string (e.g. `"ACTION1"`). For `ACTION6` (mouse click), you also provide coordinates.
- **`reasoning`**: a dict (optional, but strongly recommended; this is great for debugging and checkpoint inspection)

Example:

```python
return GameStep(
  action={"action": "ACTION3"},
  reasoning={"why": "moving left to align cursor with object"},
)
```

The harness deep-copies `reasoning` and sends it to the ARC API with the action. Keep it small: ARC payloads have size limits, and the harness does not automatically truncate this field for you.

#### What's an action?

ARC-AGI-3 gameplay is driven by a small action set:

- `RESET`: reset the game session
- `ACTION1`: Move Up
- `ACTION2`: Move Down
- `ACTION3`: Move Left
- `ACTION4`: Move Right
- `ACTION5`: Perform Action
- `ACTION6`: Click object on screen (requires coordinates)
- `ACTION7`: Undo

##### Sending action payloads (`data`, `ACTION6`, x/y)

Except `ACTION6`, most actions don’t need anything else passed with it:

```python
GameStep(action={"action": "ACTION1"}, reasoning={...})
```

`ACTION6` (mouse click) is the special case. There are two supported ways to provide coordinates:

1) Put an explicit `data` dict in the action (sent as-is to the API; coords are typically in the API grid space, 0..63):

```python
GameStep(action={"action": "ACTION6", "data": {"x": 10, "y": 20}}, reasoning={...})
```

2) Or return `"x"` / `"y"` at “pixel-ish” scale (0..127). The harness will clamp to 0..127 and then **downscale** to 0..63 before sending:

```python
GameStep(action={"action": "ACTION6", "x": 80, "y": 40}, reasoning={...})
```

You can access the list of available actions for the given game via `context.game.available_actions`. Your agent should respect this when choosing actions to avoid invalid moves.

---

### `SessionContext` (quick intuition)

`SessionContext` is the object passed into every `step()` call. Think of it as:

- “what I can see right now” (frames, score, state, actions)
- “what I remember” (an in-memory datastore that is also checkpointed to disk)
- “what the harness is tracking for me” (usage/cost, history, counters)

For your first agent, the only thing you *need* is:

- read state from `context`
- store any state your agent needs to keep track of in `context.datastore`

The next section is the full reference.

---

### `SessionContext` reference (what’s inside, how to use it)

#### Thread-safety (important)

`SessionContext` is internally lock-protected. Access via its properties and methods and avoid mutating internal dataclasses directly.

#### Game state (`context.game`)

`context.game` is a snapshot of game progress. Useful fields include:

- **`game_id`**: current game id string
- **`guid`**: the current server-side session GUID (used to continue sessions)
- **`current_score` / `previous_score`**
- **`current_state`**: `"IN_PROGRESS"`, `"WIN"`, or `"GAME_OVER"`
- **`play_num`**: which play/attempt you’re on
- **`action_counter`**: global action count across plays
- **`play_action_counter`**: action count within the current play
- **`available_actions`**: what the server says is allowed right now

You usually treat these as **read-only** inputs to your agent.

#### Frames

Frames are the state of the game world. They can either be images or text grids. Frames always arrive from the ARC server as 2D int arrays. The context keeps:

- **`context.frames.frame_grids`**: the *current* frame sequence returned by the server for this step (a tuple of grids; the last grid is usually the “latest” frame).
- **`context.frames.previous_grids`**: the *entire prior step’s* `frame_grids` sequence (useful for “what changed since my last action?” diffs). This will be empty on the very first step of a run (and may be empty right after a reset).

Convenience helpers:

- **`context.frame_images`**: current frames converted to PIL images (via `grid_to_image`)
- **`context.previous_images`**: previous frames as images
- **`context.last_frame_grid`**: last grid in the current sequence (or `None`)
- **`context.last_frame_image(resize=...)`**: last image (optionally resized)

If you’re building a text-only agent, you can stringify grids with:

- `arcagi3.utils.formatting.grid_to_text_matrix(grid)`

#### Persistent state (`context.datastore`)

`context.datastore` is a key/value store (implemented via [`threadsafe_datastore`](https://github.com/hlfshell/threadsafe_datastore)). This is where agents should store:

- memory and state your agent wants to keep track of across steps
- previous actions
- cached “derived” values (e.g., whether the model supports vision)

Treat `context.datastore` as **agent state**, not game state (game state already lives on `context.game` / `context.frames`). Don’t mutate **mutable** values you pull out of the datastore (lists/dicts) in-place unless you’re doing it atomically via the store's helper methods:

- `append(key, value)`: append to a list
- `operate(key, func)`: apply a function to a value
- `concat(key, values)`: concatenate a list.
- ...and more. See [`threadsafe_datastore`](https://github.com/hlfshell/threadsafe_datastore).

Here are some realistic examples of the kinds of things you might store:

```python
# Track the agent's current "plan" for the next few steps
context.datastore["plan"] = [
    "Locate the controllable cursor/agent",
    "Move toward the nearest interactable object",
    "Click the object and observe if score changes",
]

# Track hypotheses / beliefs you're testing
context.datastore["hypotheses"] = [
    {"name": "goal_is_reach_green", "confidence": 0.55},
    {"name": "clicking_toggles_state", "confidence": 0.35},
]

# Track model-facing scratchpad / notes (keep it reasonably sized)
context.datastore["memory_prompt"] = "Tried moving left twice; no score change. Consider clicking center object."

# Track a queue of follow-ups you want to do (useful for multi-step policies)
context.datastore["todo"] = [
    {"kind": "probe_click", "x": 18, "y": 40, "reason": "test if object is clickable"},
    {"kind": "probe_move", "action": "ACTION2", "reason": "see if agent is constrained"},
]
```

**Checkpointing requirement**:

- keys *MUST* be `str`
- values *MUST* be JSON-serializable

If you store a non-serializable object, checkpoint writes will raise an exception when the harness tries to save.

The datastore is loaded into its last state if a checkpoint is resumed.
If you want to inspect how it evolves over time, check `.checkpoint/<CARD_ID>/datastore_history.jsonl` (one datastore snapshot per action).

## Calling models (providers) the “harness-native” way

If you want to make use of our token cost accounting and model logging, you can use our provider helpers to make working with LLMs easier.

Each `MultimodalAgent` has:

- `self.provider` (a `ProviderAdapter` created from `--config`)

The recommended path is:

1) Build provider-appropriate `messages` (OpenAI-style role dicts work across all of the adapters in this repo).
2) Call `self.provider.call_with_tracking(context, messages, step_name=...)`.
3) Parse with `self.provider.extract_content(response)` (and/or your own parser).

If you want JSON outputs, use:

- `arcagi3.utils.parsing.extract_json_from_response(text)`

The reference `adcr` agent demonstrates a robust “retry once on malformed JSON” pattern.

---

## Use prompt manager (optional)

If you want to use prompts, you can use our `PromptManager` to load + render prompt files with **Jinja2 templating** to avoid having to wrangle them yourself.

### How prompt loading works

`PromptManager` loads prompts **relative to the Python file that calls it**:

- If your agent module is `src/arcagi3/my_agent/agent.py`, it will look for prompts in:
  - `src/arcagi3/my_agent/prompts/<name>.prompt` *(and then `src/arcagi3/my_agent/prompts/<name>` as a fallback)*

### Variables + Jinja2 templating

Prompt files can reference variables you pass to `render()` and can use Jinja2 features like `{% if %}` and `{% for %}`.

Example prompt file:

```text
# src/arcagi3/my_agent/prompts/system.prompt
You are an ARC-AGI-3 agent.

{% if use_vision %}
You will receive images. Use them.
{% else %}
You will receive text grids only. Use them.
{% endif %}

Memory (optional):
{{ memory_prompt }}
```

And the corresponding render call:

```python
from arcagi3.prompts import PromptManager

prompt_manager = PromptManager()
prompt = prompt_manager.render(
    "system",
    {
        "use_vision": want_vision,
        "memory_prompt": context.datastore.get("memory_prompt", ""),
    },
)
```

---
