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
am92                  AM92                          
as66                  AS66                          
bt11                  BT11                          
dc22                  DC22                          
is41                  IS41                          
lp85                  LP85                          
mk45                  MK45                          
ne57                  NE57                          
ra05                  RA05                          
re86                  RE86                          
sb26                  SB26                          
sp80                  SP80                          
tl01                  TL01                          
wc25                  WC25                                               
=========================================================
```

## Pick your model

We use game ids to identify models in our tooling. If you have your API keys set up, run the following to see all possible models for you:

```bash
uv run python -m arcagi3.runner --list-models

================================================================================
Available Models (for enabled providers)
================================================================================

OpenRouter (14 models):
--------------------------------------------------------------------------------
  claude-4-sonnet-20250522-thinking-8k-bedrock multimodal           $3.00/$15.00 per 1M tokens
  claude-opus-4-5-openrouter               multimodal           $5.00/$25.00 per 1M tokens
  claude-sonnet-4-5-openrouter             multimodal           $3.00/$15.00 per 1M tokens
  deepseek_r1_0528-openrouter              standard             $0.50/$2.18 per 1M tokens
  gemini-2-5-pro-preview-openrouter        multimodal           $1.25/$10.00 per 1M tokens
  gemini-2-5-pro-preview-openrouter-thinking-1k multimodal           $1.25/$10.00 per 1M tokens
  gemini-3-0-pro-preview-openrouter        multimodal           $2.00/$12.00 per 1M tokens
  gpt-5-2-openrouter                       multimodal           $1.75/$14.00 per 1M tokens
  magistral-medium-2506                    standard             $2.00/$5.00 per 1M tokens
  magistral-medium-2506-thinking           standard             $2.00/$5.00 per 1M tokens
  magistral-small-2506                     standard             $0.50/$1.50 per 1M tokens
  qwen3-235b-a22b-07-25                    standard             $0.12/$0.59 per 1M tokens

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

When you run a benchmark, a scorecard is opened on the ARC server. You can list and view scorecards by running:

```bash
uv run python -m arcagi3.runner --list-scorecards
```

You can also view a specific scorecard by running:

```bash
uv run python -m arcagi3.runner --scorecard <CARD_ID>
```

If you're logged in, scoredcards can be viewed at [three.arcprize.org/scorecards](https://three.arcprize.org/scorecards).

# Checkpoints

While you run a benchmarking game, its progress is saved as a checkpoint locally. You can list and resume from checkpoints by running:

```bash
uv run python -m arcagi3.runner --list-checkpoints
uv run python -m arcagi3.runner --checkpoint <CARD_ID>
```

When resuming, `--config` and `--game_id` can be omitted; they’re recovered from checkpoint metadata when possible. By default, checkpoints live under `.checkpoint/<card_id>/`.

# Docker

There is a `Dockerfile` that installs the package and defaults to running `python main.py`.
It also exposes a set of environment variables that map to common CLI flags (see the `Dockerfile`).