# ARC Harness `arcagi3`

This is a developer harness for building and benchmarking agentic research workflows on the **ARC-AGI-3** corpus of reasoning games. The goal of this repository is to get developers and researchers running AI agents on ARC games as quickly as possible, with features designed to aid them in their experiments.

# Quick Start

## Prerequisites

- **Python**: `3.9+`
- **uv**: recommended package manager. Install from [uv.pm](https://github.com/astral-sh/uv) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node**: `22+` *optional* - only needed for building the breakpoint tool UI.
- **ARC-AGI-3 API key**: required to talk to the ARC server. Sign up for a key [here](https://three.arcprize.org/).

## Install

From repo root:

```bash
uv sync
```

This will create a virtual environment (if needed) and install the project and all dependencies in editable mode.

Alternatively, if you're not using `uv`:

```bash
pip install -e .
```

## Set Environment Variables

You can either set environment variables directly in your shell or utilize a `.env` file. `.env.example` demonstrates all expected environment variables.

## Running an Agent

### 1) List available prepared agents

```bash
python -m arcagi3.runner --list-agents
```

As you create your own agents, you can register them with the runner for easier launching. More on that later.

### 2) Run a single game with a prepared agent

```bash
python -m arcagi3.runner \
  --agent adcr \
  --game_id ls20 \
  --config gpt-4o-mini-2024-07-18 \
  --max_actions 40
```

- Results (if enabled) are written under `results/<config>/...json`
- A scorecard URL is printed at the end
- A checkpoint is continuously updated under `.checkpoint/<card_id>/` (unless disabled) which shows in progress actions and allows resume.

### 3) Resume from a checkpoint

```bash
python -m arcagi3.runner --list-checkpoints
python -m arcagi3.runner --checkpoint <CARD_ID>
```

When resuming, `--config` and `--game_id` can be omitted; they’re recovered from checkpoint metadata when possible. By default, checkpoints live under `.checkpoint/<card_id>/`.

### 4) Run without creating scorecards (local-only mode)

```bash
python -m arcagi3.runner \
  --agent adcr \
  --game_id ls20 \
  --config gpt-4o-mini-2024-07-18 \
  --no-scorecard-submission
```

This avoids opening/closing scorecards, but still plays the game through the ARC API. A `local-<uuid>` `card_id` is used for checkpointing.

## breakpointer (interactive debugging UI)

The breakpointer tool is a UI to help you gain a better understanding of what your model is doing, as well as allow you to manually experiment with agents during a game play session. Breakpoints let you **pause** your agent at named breakpoints and optionally **override** payload fields (e.g., edit memory, adjust the chosen action, or manually patch reasoning of the model). It is designed to be flexible to whatever pattern your agent utilizes to play games.

**Note:** Building the breakpoint UI requires Node.js (see Prerequisites).

### 0) Build the UI (first time setup)

Before using the breakpoint tool, you need to build the UI:

```bash
cd breakpointer
npm install
npm run build
cd ..
```

This creates the `breakpointer/dist` directory that the server will serve. If you modify the UI source code later, re-run `npm run build` and restart the server.

### 1) Start the breakpoint server

```bash
python scripts/run_breakpoint_server.py
```

The server will automatically serve `breakpointer/dist` if it exists, otherwise it will serve from the project root (which won't have the UI).

### 2) Run an agent with breakpoints enabled

```bash
python -m arcagi3.runner \
  --agent adcr \
  --game_id ls20 \
  --config gpt-4o-mini-2024-07-18 \
  --breakpoints
```

Breakpoints pause only at points defined in the **active breakpoint spec**. That spec can come from an agent’s runtime registration (see `src/arcagi3/examples/adcr/breakpoints.py`) and/or a JSON spec provided via `--breakpoint-schema`.

We will dive deeper into making your own breakpoints when discussing creating your own agents.

# Making your own Agent

To create your own agent, implement an `MultimodalAgent` child class (`src/arcagi3/agent.py`). Then implement the `step(context) -> GameStep` function.

- **State comes in via** `context: SessionContext`
  - Current/previous frames (as grids and optionally images)
  - Current score/state, available actions, counters
  - `context.datastore`: JSON-serializable scratch space for your agent (memory, hypotheses, plans, etc.)
  - `context.metrics`: aggregated usage/cost
- **You return** a `GameStep` with:
  - `action`: a dict with at least `"action": "ACTION1" | ... | "ACTION7"` (or `"RESET"`)
  - optional `action["data"]` dict (passed through to ARC API)
  - optional `reasoning` dict (sent to ARC API; keep it small—there’s a hard size limit)

All history of the game is stored within `context`, allowing you to implement whatever transformations or history tracking that you wish.


## Example Agents

There are several examples in this repository of creating custom agent flows in the `src/arcagi3/examples` folder. Each one demonstrates a unique experimental flow, with further explanations in their own `README`. Note that while these are different approaches for tackling ARC games, *none* can solve an ARC game at present even with, at time of writing, frontier models. These serve as excellent skeleton builds to inspire your own approaches.

- **`adcr`**: Analyze → Decide → Convert → Review reference loop (baseline). See `src/arcagi3/examples/adcr/README.md`.
- **`state-transform`**: ADCR variant with a pluggable “state transform” pre-processing step (includes a couple CLI presets). This transforms the state - text or image - per your speciications prior to performing the ADCR loop. See `src/arcagi3/examples/state_transform_adcr/README.md`.
- **`knowitall`**: Action-only agent that assumes you already know the full rules (provided via `--game-rules`). Excellent for observing agent mechanic interactions. See `src/arcagi3/examples/knowitall/README.md`.
- **`curriculum`**: Phase-based ADCR (exploration → refinement → execution) to encourage early discovery. See `src/arcagi3/examples/curriculum_agent/README.md`.
- **`hypothesis`**: Maintains explicit hypotheses + experiments; periodically updates them from recent actions. See `src/arcagi3/examples/hypothesis_agent/README.md`.
- **`rules`**: Periodically distills “rules so far” and uses them to guide decisions. See `src/arcagi3/examples/rules_agent/README.md`.
- **`swarm`**: Coordinates action selection across multiple games, having an agent benefit from seeing multiple simultaneous games to speed up mechanics discovery. See `src/arcagi3/examples/swarm_agent/README.md`.


## Making your own runner

The easiest way to run your own agent from the CLI is to create a small entrypoint that:

- registers your agent in an `AgentRunner` (via a `flags` dict)
- delegates all CLI parsing/execution to `runner.run()`

Example (`my_runner.py`):

```python
from dotenv import load_dotenv

from arcagi3.runner import AgentRunner
from my_agent import MyAgent  # your MultimodalAgent subclass


flags = {
    "name": "my-agent",
    "description": "My experimental agent",
    "agent_class": MyAgent,
    # Optional:
    # "add_args": lambda parser: parser.add_argument("--foo", default="bar"),
    # "get_kwargs": lambda args: {"foo": args.foo},
}


def main() -> None:
    load_dotenv()
    runner = AgentRunner()
    runner.add_flags(flags)
    runner.run()


if __name__ == "__main__":
    main()
```

See existing `flags` patterns:
- Minimal: `src/arcagi3/examples/adcr/flags.py`
- With custom args/kwargs:
    - `src/arcagi3/examples/knowitall/flags.py`
    - `src/arcagi3/examples/state_transform_adcr/flags.py`

## `context.datastore`

`context.datastore` is your agent’s **persistent**, **per-run** state store.

- It’s a JSON-serializable dict that lives on the `SessionContext`.
- It survives across steps/actions within a run and is saved/restored via **checkpoints** (so it persists across resume). If you run multiple plays in one run, it can also carry across plays of the same game.

For instance - if we had an LLM modifiable memory, and wished to feed the prompt of the model with that memory, a record of its previous reasoning for its move, and a long running experiment to determine game mechanics, we could do that within `step()`:

```python
# Similar to `adcr`, keep a persistent scratchpad + the last prompt/action.
memory = context.datastore.get("memory_prompt", "")
previous_reasoning = context.datastore.get("previous_prompt", "")
current_experiment = context.datastore.get("current_experiment", "")

# Utilize the memory, prior reasoning, and current long running experiment
# with current observations of the state
...
```

# Model configs

`--config` selects an entry from:

- `src/arcagi3/models.yml`
- optional `src/arcagi3/models_private.yml` (local/private; loaded automatically if it exists)

Each model config includes:
- provider (which adapter to use)
- model name
- pricing (used for cost calculation)
- optional kwargs (streaming, reasoning effort, max tokens, etc.)

When specifying a model, it is assumed the environment variables for that provider contains an appropriate key for service.

# Docker

There is a `Dockerfile` that installs the package and defaults to running `python main.py`.
It also exposes a set of environment variables that map to common CLI flags (see the `Dockerfile`).

# Troubleshooting

## “ARC_API_KEY not found”

- Set `ARC_API_KEY` in your shell or `.env`
- Confirm you’re running from a directory where `.env` is picked up (entrypoints call `load_dotenv()`)

## Provider auth errors

- Your chosen `--config` implies a provider; ensure that provider’s API key env var is set.

## Breakpoints never pause

- Start the server: `python scripts/run_breakpoint_server.py`
- Run with `--breakpoints`
- If you changed ports, set `--breakpoint-ws-url`
