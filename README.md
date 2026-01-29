# ARC Harness `arcagi3`

This is a developer harness for building and benchmarking agentic research workflows on the **ARC-AGI-3** corpus of reasoning games. The goal of this repository is to get developers and researchers running AI agents on ARC games as quickly as possible, with features designed to aid them in their experiments.

# Quick Start

## Prerequisites

- **Python**: `3.9+`
- **uv**: recommended package manager. Install from [uv.pm](https://github.com/astral-sh/uv) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **ARC-AGI-3 API key**: required to talk to the ARC server. Sign up for a key [here](https://three.arcprize.org/).

## Install

Clone the repository:

```bash
git clone git@github.com:arcprize/arc-agi-3-benchmarking.git
cd arc-agi-3-benchmarking
```

From repo root:

```bash
uv venv
uv sync
```

This will create a virtual environment (if needed) and install the project and all dependencies in editable mode.

Alternatively, if you're not using `uv` (guide will continue assuming `uv`):

```bash
pip install -e .
```

## Setting up your environment

In order to communicate with the ARC server and utilize LLM providers, we need to set up environment variables. To get an API key for ARC AGI 3, you can sign up for a key [here](https://three.arcprize.org/). For your chosen provider(s), you can go to:

- [OpenAI](https://platform.openai.com/account/api-keys)
- [Anthropic](https://console.anthropic.com/account/api-keys)
- [Google Gemini](https://console.cloud.google.com/apis/credentials)
- [OpenRouter](https://openrouter.ai/api-keys)
- [Fireworks](https://app.fireworks.ai/account/api-keys)
- [Groq](https://groq.com/account/api-keys)
- [DeepSeek](https://console.deepseek.com/account/api-keys)
- [Hugging Face](https://huggingface.co/settings/tokens)

Once you have your API keys, you can safely place them in either a `.env` file in your project directory (feel free to copy our `.env.example` for a quicker start) or set them in your environment variables directly.

To check to see if your environment variables are set correctly, you can run:

```bash
uv run python -m arcagi3.runner --check

================================================================================
Service                   Environment Variable           Status                   
================================================================================
ARC-AGI-3 API             ARC_API_KEY                    ✓ Connected (found 6 games)
OpenAI                    OPENAI_API_KEY                 ✓ Valid                  
Anthropic                 ANTHROPIC_API_KEY              ✓ Valid           
Google Gemini             GOOGLE_API_KEY                 Not configured           
OpenRouter                OPENROUTER_API_KEY             ✓ Valid                  
Fireworks                 FIREWORKS_API_KEY              Not configured           
Groq                      GROQ_API_KEY                   Not configured           
DeepSeek                  DEEPSEEK_API_KEY               ✓ Valid          
xAI                       XAI_API_KEY                    Not configured           
Hugging Face              HUGGING_FACE_API_KEY           Not configured           
================================================================================

================================================================================
✓ READY TO BENCHMARK
  - ARC-AGI-3 API: ✓ Connected
  - Provider APIs: 4 configured and working
================================================================================
```

## Select your game

If your API keys are set up, you can see what games are available to you by running:

```bash
uv run python -m arcagi3.runner --list-games

=========================================================
Game ID               Title                         
=========================================================
ls20                  LS20                          
ft09                  FT09                          
vc33                  VC33                          
                                           
=========================================================
```

## Pick your model

We use game ids to identify models in our tooling. If you have your API keys set up, run the following to see all possible models for you:

```bash
uv run python -m arcagi3.runner --list-models

================================================================================
Available Models (for enabled providers)
================================================================================

OpenRouter (12 models):
--------------------------------------------------------------------------------
  claude-4-sonnet-20250522-thinking-8k-bedrock multimodal           $3.00/$15.00 per 1M tokens
  claude-opus-4-5-openrouter               multimodal           $5.00/$25.00 per 1M tokens
  claude-sonnet-4-5-openrouter             multimodal           $3.00/$15.00 per 1M tokens
  deepseek_r1_0528-openrouter              standard             $0.40/$1.75 per 1M tokens
  gemini-2-5-pro-preview-openrouter        multimodal           $1.25/$10.00 per 1M tokens
  gemini-2-5-pro-preview-openrouter-thinking-1k multimodal           $1.25/$10.00 per 1M tokens
  gemini-3-0-pro-preview-openrouter        multimodal           $2.00/$12.00 per 1M tokens
  gpt-5-2-openrouter                       multimodal           $1.75/$14.00 per 1M tokens
  magistral-medium-2506                    standard             $2.00/$5.00 per 1M tokens
  magistral-medium-2506-thinking           standard             $2.00/$5.00 per 1M tokens
  magistral-small-2506                     standard             $0.50/$1.50 per 1M tokens
  qwen3-235b-a22b-07-25                    standard             $0.20/$0.60 per 1M tokens

================================================================================
Total: 12 models available
================================================================================
```

## Benchmark!

It's time to benchmark an agent! Let's say we want to benchmark OpenAI's `GPT-5` via openrouter the LS20 game. We can do that by running:

```bash
uv run python -m arcagi3.runner \
  --game_id ls20 \
  --config gpt-5-2-openrouter \
  --max_actions 3
```

## Scorecards

When you run a benchmark, a scorecard is saved on the ARC server.

If you're logged in, scorecards can be viewed at [three.arcprize.org/scorecards](https://three.arcprize.org/scorecards).

You can also view what your model did by looking at your local checkpoint folder in `.checkpoint/<CARD_ID>`.

# Checkpoints

While you run a benchmarking game, its progress is saved as a checkpoint locally (default folder is `.checkpoint/<CARD_ID>`). You can list and resume from checkpoints by running:

```bash
uv run python -m arcagi3.runner --list-checkpoints
uv run python -m arcagi3.runner --checkpoint <CARD_ID>
```

When resuming, `--config` and `--game_id` can be omitted; they’re recovered from checkpoint metadata when possible. By default, checkpoints live under `.checkpoint/<card_id>/`.

Each checkpoint folder contains JSON files intended for inspection:
- `metadata.json`: game/config info, scores, counters, frame grids, datastore snapshot
- `costs.json`: total usage and cost for the run
- `action_history.json`: per-action results, reasoning, and per-action cost
- `model_completion.json`: per-model-call prompts/messages and responses (with usage/cost)
- `error.json`: error details if a run failed (only present on error)

# Model Configs

Model configurations are `YAML` entries that define how to use a specific model with a provider and key metadata information. They specify the model name, provider, pricing information, API parameters, and any provider-specific settings needed to make API calls. When you run a benchmark with `--config <config_name>`, the system looks up that configuration and uses it to initialize the appropriate provider adapter.

## Configuration Files

Model configurations are stored in `YAML` files:

- **`src/arcagi3/models.yml`**: The main configuration file containing all public model definitions
- **`src/arcagi3/models_private.yml`** (optional): A private configuration file that can be used for models you don't want to commit to version control. If this file exists, its entries are merged with `models.yml` at runtime.

Both files follow the same structure: a top-level `models:` key containing a list of model configuration dictionaries.

## Configuration Structure

Each model configuration entry must include the following **required fields**:

- **`name`** (string): A unique identifier for this configuration. Must be globally unique within the runtime. This is what you pass to `--config` when running benchmarks. Examples: `"gpt-5-2-openrouter"`, `"claude-sonnet-4-5-20250929"`

- **`model_name`** (string): The actual model identifier used by the provider's API. This may differ from the config name. For example:
  - OpenAI: `"gpt-5-2025-08-07"`
  - OpenRouter: `"openai/gpt-5.2"` (includes provider prefix)
  - Anthropic: `"claude-sonnet-4-5-20250929"`

- **`provider`** (string): The provider adapter to use. Must match one of the supported providers: `"openai"`, `"anthropic"`, `"gemini"`, `"openrouter"`, `"deepseek"`, `"fireworks"`, `"huggingfacefireworks"`, or `"groq"`

- **`pricing`** (object): Pricing information for cost tracking:
  - `date` (string): Date when pricing was last updated (e.g., `"2025-08-07"`)
  - `input` (float): Cost per 1 million input tokens in USD
  - `output` (float): Cost per 1 million output tokens in USD

- **`is_multimodal`** (boolean, default: `false`): Whether the model supports image inputs. Set to `true` for vision-capable models.

- **`api_type`** (string, default: `"chat_completions"`): The API type to use. Common values:
  - `"chat_completions"`: Standard chat completion API (default)
  - `"responses"`: OpenAI's Responses API format

- **Provider-specific parameters**: Any additional fields beyond the known ones (`name`, `model_name`, `provider`, `pricing`, `kwargs`, `api_type`, `is_multimodal`) are automatically extracted into the `kwargs` dictionary and passed to the provider's API. Common examples:
  - `max_tokens`, `max_output_tokens`, `max_completion_tokens`: Token limits
  - `temperature`: Sampling temperature
  - `stream`: Whether to use streaming responses
  - `reasoning`: Reasoning configuration (for OpenAI models)
  - `thinking`: Thinking configuration (for Anthropic models)
  - `extra_body`: Additional request body parameters (for OpenRouter)

## Adding a New Model Configuration

To add a new model for a given provider, follow these steps:

### Step 1: Choose Your Configuration File

Decide whether to add the model to `models.yml` (public) or `models_private.yml` (private). If you're unsure, use `models.yml` for models that are generally available.

### Step 2: Find an Existing Example

Look for an existing model configuration from the same provider in `models.yml` to use as a template. Each provider has different parameter requirements.

### Step 3: Add Your Configuration Entry

Add a new entry to the `models` list in your chosen YAML file. Here are examples for different providers:

#### Example: OpenAI Model

```yaml
  - name: "gpt-360"
    model_name: "gpt-360"
    provider: "openai"
    is_multimodal: true
    api_type: "responses"
    reasoning:
      effort: "high"
      summary: "auto"
    max_output_tokens: 200000
    pricing:
      date: "2067-10-08"
      input: 13.25
      output: 20.00
```

### Step 4: Verify Your Configuration

After adding your configuration, verify it works:

1. **Check that the configuration loads**:
   ```bash
   uv run python -m arcagi3.runner --list-models
   ```
   Your new model should appear in the list if the provider's API key is configured.

2. **Test with a simple benchmark**:
   ```bash
   uv run python -m arcagi3.runner \
     --game_id <game_id> \
     --config <your-config-name> \
     --max_actions 1
   ```

### Step 5: Provider-Specific Notes

Different providers have different parameter conventions:

- **OpenAI**: Uses `max_output_tokens` (not `max_tokens`), supports `reasoning` configuration for reasoning models, and may use `api_type: "responses"` for newer models.

- **OpenRouter**: Uses `max_tokens`, requires provider prefix in `model_name` (e.g., `"openai/gpt-5.2"`), and uses `extra_body` for additional parameters like reasoning configuration.

- **Anthropic**: Uses `max_tokens`, supports `thinking` configuration for thinking models with `budget_tokens`, and commonly uses `stream: true`.

- **Gemini**: Uses `max_output_tokens`, supports `thinking_config` for thinking models, and may include `automatic_function_calling` settings.

- **Other providers**: Check existing examples in `models.yml` for the specific provider's conventions.

### Common Pitfalls

1. **Provider name mismatch**: The `provider` field must exactly match the adapter name (lowercase, no spaces). Check `src/arcagi3/adapters/__init__.py` for valid provider names.

2. **Model name format**: For OpenRouter, the `model_name` must include the provider prefix (e.g., `"openai/gpt-5.2"`), while direct provider APIs use just the model identifier.

3. **Pricing accuracy**: Ensure pricing reflects current rates. Incorrect pricing will lead to inaccurate cost tracking in your benchmarks.

4. **Parameter naming**: Different providers use different parameter names (e.g., `max_tokens` vs `max_output_tokens` vs `max_completion_tokens`). Always check existing examples for your provider.

5. **YAML syntax**: Ensure proper indentation (2 spaces) and that list items start with `-`. YAML is sensitive to formatting.

# Docker

There is a `Dockerfile` that installs the package and defaults to running `python main.py`.
It also exposes a set of environment variables that map to common CLI flags (see the `Dockerfile`).

## Build the image

From the repo root:

```bash
docker build -t ARCAGI3-Benchmarker .
```

## Provide environment variables

You can supply API keys and settings in three ways:

1) **Pass variables directly**

```bash
docker run --rm \
  -e ARC_API_KEY=... \
  -e OPENROUTER_API_KEY=... \
  ARCAGI3-Benchmarker \
  python -m arcagi3.runner --game_id am92-80effacb --config gpt-5-2-openrouter --max_actions 1
```

2) **Load variables from your local `.env` file**

```bash
docker run --rm \
  --env-file "$(pwd)/.env" \
  ARCAGI3-Benchmarker \
  python -m arcagi3.runner --game_id am92-80effacb --config gpt-5-2-openrouter --max_actions 1
```

3) **Use your current shell environment**

```bash
export ARC_API_KEY=...
export OPENROUTER_API_KEY=...

docker run --rm \
  -e ARC_API_KEY \
  -e OPENROUTER_API_KEY \
  ARCAGI3-Benchmarker \
  python -m arcagi3.runner --game_id am92-80effacb --config gpt-5-2-openrouter --max_actions 1
```

## Persist checkpoints and results

By default, results are written to `results/` and checkpoints to `.checkpoint/` inside the container. To keep them on your host machine, mount volumes:

```bash
mkdir -p .checkpoint results

docker run --rm \
  --env-file "$(pwd)/.env" \
  -v "$(pwd)/.checkpoint:/app/.checkpoint" \
  -v "$(pwd)/results:/app/results" \
  ARCAGI3-Benchmarker \
  python -m arcagi3.runner --game_id am92-80effacb --config gpt-5-2-openrouter --max_actions 1
```

# Creating your own agent

Want to create your own agent? Check out the [docs/create_agent.md](docs/create_agent.md) file a walkthrough on how it's easily done.

# Contributing

Install the development tools:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run all formatting and lint checks locally:

```bash
pre-commit run --all-files
```

git pre-commit hooks run automatically before each commit so you don't accidentally commit unlinted code.

Notes:
- The hooks run `isort`, `black`, and `ruff` on staged Python files.
- If you change formatting/lint configs, re-run the hooks to update files.

# Citation

```bibtex
@software{arcprize_arc_agi_3_benchmarking_2026,
  author       = {{ARC Prize Foundation}},
  title        = {{ARC-AGI-3 Benchmarking}},
  year         = {2026},
  url          = {https://github.com/arcprize/arc-agi-3-benchmarking},
  note         = {Accessed: 2026-01-28}
}
```

# LICENSE

MIT License

Copyright (c) 2026 ARC Prize Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.