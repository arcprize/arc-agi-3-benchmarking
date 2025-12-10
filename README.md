# ARC-AGI-3 Benchmarking

A comprehensive benchmarking framework for evaluating multimodal Large Language Models (LLMs) on ARC-AGI-3 interactive games. The ARC (Abstraction and Reasoning Corpus) Prize challenges AI systems to solve abstract reasoning puzzles that require core knowledge and pattern recognition.

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Run a game
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-mini-2024-07-18" \
  --max_actions 40
```

## Features

- **Multimodal Agent** - Vision-based reasoning with image analysis using 128x128 PNG images
- **Multi-Provider Support** - 90+ model configurations across 9 providers (OpenAI, Anthropic, Gemini, Grok, DeepSeek, OpenRouter, Fireworks, Groq, XAI)
- **Cost Tracking** - Granular per-action token usage and cost calculation
- **Retry Logic** - Exponential backoff with configurable retry attempts
- **Rate Limiting** - Provider-specific async rate limiting to prevent throttling
- **Results Export** - Comprehensive JSON output with full game history and scorecard URLs
- **Checkpoint System** - Robust save/resume functionality for crash recovery
- **Extended Thinking** - Support for reasoning models (Claude thinking variants, o1-style)
- **Memory Management** - Scratchpad-based memory with automatic compression
- **Multi-Play Sessions** - Continue learning across multiple attempts
- **Hints System** - Optional game-specific guidance via YAML configuration
- **Batch Processing** - Run multiple games concurrently
- **Concurrent Execution** - Parallel game runs with per-game logging and thread-safe execution
- **Terminal Visualization** - Display game frames as colored blocks during gameplay

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd arc-agi-3-benchmarking

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables

Create a `.env` file with the following keys (only add keys for providers you'll use):

```bash
# Required
ARC_API_KEY=your_arc_api_key_here

# Provider API Keys (add as needed)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
XAI_API_KEY=your_xai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
```

### Optional Dependencies

Providers load lazily - only install SDKs for providers you'll use:

```bash
pip install openai            # For OpenAI models
pip install anthropic         # For Claude models
pip install google-genai      # For Gemini models
```

## Usage

### List Available Games

```bash
python -m arcagi3.cli --list-games
```

### Run Single Game

```bash
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-mini-2024-07-18" \
  --max_actions 40 \
  --save_results_dir "results/gpt4o"
```

### Display Images in Terminal
Show game frames directly in your terminal during gameplay:
```bash
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-mini-2024-07-18" \
  --show-images
```
This renders each frame as colored blocks in your terminal, useful for debugging and visualizing the agent's progress in real-time.

### Run Multiple Games (Batch Processing)

Run multiple games at the same time with automatic per-game log files:

```bash
# Run specific games concurrently
python -m arcagi3.cli \
  --games "ls20-fa137e247ce6,ft09-16726c5b26ff,ls20-016295f7601e" \
  --config "gpt-4o-mini-2024-07-18"

# Run all available games concurrently
python -m arcagi3.cli \
  --all-games \
  --config "claude-sonnet-4-5-20250929"
```

**Concurrent Execution Features:**
- All games run in parallel using threads (max: number of games)
- Each game gets its own log file in `logs/<config>/concurrent_<timestamp>/`
- Separate thread-safe execution for each game
- Complete summary with aggregate statistics at the end
- Scorecard URLs for all games printed together

**Example output structure:**
```
logs/
└── gpt-4o-mini-2024-07-18/
    └── concurrent_20251119_143528/
        ├── ls20-fa137e247ce6.log
        ├── ft09-16726c5b26ff.log
        └── ls20-016295f7601e.log
```

### Checkpoint & Resume

The checkpoint system automatically saves game progress for crash recovery and resumption.

#### How It Works

Checkpoints are saved automatically:
- After every action (configurable via `--checkpoint-frequency`)
- At the end of each play (for multi-play runs)
- Stored in `.checkpoint/{scorecard-id}/` directory

Each checkpoint contains:
- Complete conversation history and agent memory
- All action records with reasoning
- Cost and token usage statistics
- Session identifiers (game_id, guid, card_id)
- Previous frame images and grids
- Vision mode setting

#### Quick Start

```bash
# 1. Start a game (checkpoint created automatically)
python main.py --game_id "ls20-fa137e247ce6" --config "gpt-4o-2024-11-20"
# Note the scorecard ID from logs: "Opened scorecard: abc-123-def-456"

# 2. If interrupted (crash, Ctrl-C, etc.), list checkpoints
python main.py --list-checkpoints

# 3. Resume from checkpoint using the scorecard ID
python main.py --checkpoint abc-123-def-456
```

The game continues from where it left off with:
- ✅ Same conversation history and memory
- ✅ Accumulated costs and actions
- ✅ Same session on the server (when possible)

#### Advanced Options

```bash
# Customize checkpoint frequency (save every 5 actions instead of 1)
python main.py --game_id "ls20-fa137e247ce6" --config "gpt-4o-2024-11-20" --checkpoint-frequency 5

# Disable periodic checkpoints (only save at end of play)
python main.py --game_id "ls20-fa137e247ce6" --config "gpt-4o-2024-11-20" --checkpoint-frequency 0

# Close scorecard manually (prevents resume)
python main.py --close-scorecard abc-123-def-456

# Close scorecard on exit (even if not won)
python main.py --game_id "ls20-fa137e247ce6" --config "gpt-4o-2024-11-20" --close-on-exit
```

#### Important Notes

**Session Continuity**: When resuming, the system attempts to restore the exact game session using the saved GUID. If the session has expired (server timeout), the system:
- Opens a new game session
- Preserves the agent's memory and learned patterns
- Continues from the new session with existing knowledge

**Scorecard Management**: By default, scorecards are kept open after crashes/interrupts to enable resume. They're only closed when:
- ✅ Game is won (WIN state)
- ✅ `--close-on-exit` flag is used
- ❌ NOT on Ctrl-C or crash (kept open for resume)

**Storage**: Checkpoints include frame images and can be several MB in size. Delete old checkpoints manually with `rm -rf .checkpoint/{card-id}` if needed.

### Multi-Play Sessions

Continue learning across multiple attempts with preserved memory:

```bash
# Play the same game 3 times with accumulated knowledge
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-2024-11-20" \
  --num_plays 3 \
  --max_actions 40
```

Features:
- Same session GUID (maintains server-side state)
- Preserved memory scratchpad between plays
- Accumulated costs and actions
- Returns best result across plays
- Stops early on WIN

### Programmatic Usage

```python
from arcagi3 import MultimodalAgent, GameClient

# Initialize
client = GameClient()
games = client.list_games()

# Open scorecard for tracking
scorecard_response = client.open_scorecard([games[0]['game_id']])
card_id = scorecard_response['card_id']

# Create agent
agent = MultimodalAgent(
    config="gpt-4o-mini-2024-07-18",
    game_client=client,
    card_id=card_id,
    max_actions=40
)

# Play game and get results
result = agent.play_game(games[0]['game_id'])
print(f"Score: {result.final_score}, Cost: ${result.total_cost.total_cost:.4f}")
print(f"Scorecard: {result.scorecard_url}")

# Close scorecard when done
client.close_scorecard(card_id)
```

## CLI Options

### Main CLI ([main.py](main.py))

| Option | Description | Default |
|--------|-------------|---------|
| `--game_id` | Game ID to play | Required* |
| `--config` | Model config name from [models.yml](src/arcagi3/models.yml) | Required* |
| `--checkpoint` | Resume from checkpoint (card_id) | None |
| `--list-checkpoints` | List all available checkpoints | - |
| `--close-scorecard` | Close a scorecard by ID | - |
| `--checkpoint-frequency` | Save checkpoint every N actions (0=disabled) | 1 |
| `--close-on-exit` | Close scorecard on exit (prevents resume) | False |
| `--max_actions` | Maximum actions per game | 40 |
| `--num_plays` | Number of play attempts (continues session) | 1 |
| `--save_results_dir` | Results output directory | `results/<config>` |
| `--retry_attempts` | API retry attempts on failure | 3 |
| `--show-images` | Display frames in terminal | False |
| `--use_vision` | Use vision mode (images vs text grids) | True |
| `--memory-limit` | Memory scratchpad word limit | 500 |
| `--verbose` | Enable verbose logging | False |

*Not required when using `--checkpoint` (loaded from checkpoint)

### Batch CLI (`arcagi3.cli`)

| Option | Description |
|--------|-------------|
| `--list-games` | List available games and exit |
| `--games "id1,id2"` | Run specific games |
| `--all-games` | Run all available games |

All main CLI options also available.

### Multi-Run Orchestrator (`multimain.py`)

The multi-run orchestrator lets you sweep over models, games, and key agent parameters, running up to **Y agents concurrently** and resuming from checkpoints using a **run manifest YAML**.

#### Key Features

- **Sweepable dimensions** (any can be a single value or a list; all lists are combinatorially combined):
  - `configs` (model configs from `models.yml`)
  - `games` (game IDs)
  - `max_actions`
  - `num_plays`
  - `memory_limits` (maps to `memory_word_limit`)
  - `checkpoint_frequencies`
  - `use_vision` (True/False)
- **Sweep repetitions**:
  - `--sweep-repeats N` or YAML `sweep_repeats: N` runs the **entire sweep N times**, even if there is only a single unique combination. This is useful for building a corpus of repeated trials.
- **Run-level options** (normally single-valued, but override-able):
  - `save_results_dir`, `overwrite_results`
  - `retry_attempts`, `retries`
  - `show_images`
  - `close_on_exit`
  - `no_scorecard_submission`
  - logging options (`log_level`, `verbose`)
- **Concurrency**: run up to `--max-concurrent` agents at once (default: 5)
- **Resumable runs**: manifest YAML tracks all jobs, states, and checkpoint IDs so you can resume unfinished jobs later
- **Checkpoint safety**: only checkpoints explicitly referenced in the manifest are considered part of a multi-run; unrelated checkpoints are left untouched

#### Basic CLI Usage

Sweep over 2 models × 3 games with up to 5 concurrent agents:

```bash
python multimain.py \
  --configs gpt-4o-2024-11-20 claude-sonnet-4-5-20250929 \
  --games ls20-fa137e247ce6 ft09-16726c5b26ff ls20-016295f7601e \
  --max-actions 40 \
  --num-plays 1 3 \
  --max-concurrent 5
```

This:
- Generates all combinations of `configs × games × max_actions × num_plays` (with defaults for unspecified dimensions)
- Writes a manifest YAML at `results/multirun/multirun_<timestamp>.yml` (unless `--manifest-path` is provided)
- Saves results JSONs under `results/multirun/<run_name>/` (unless `--save-results-dir` is provided)

You can preview the jobs without running them using `--dry-run`:

```bash
python multimain.py \
  --configs gpt-4o-2024-11-20 \
  --games ls20-fa137e247ce6 ft09-16726c5b26ff \
  --max-actions 40 80 \
  --dry-run
```

#### YAML Sweep Config

You can also define sweeps in a YAML file and then override any setting via CLI. When **CLI args and YAML disagree, CLI wins**.

Example `experiments/multirun_example.yml`:

```yaml
run_name: my_model_sweep

configs:
  - gpt-4o-2024-11-20
  - claude-sonnet-4-5-20250929

games:
  - ls20-fa137e247ce6
  - ft09-16726c5b26ff
  - ls20-016295f7601e

max_actions: [40]
num_plays: [1, 3]
memory_limits: [500, 1000]
checkpoint_frequencies: [1]
use_vision: [true]

save_results_dir: "results/multirun/my_model_sweep"
max_concurrent: 5
sweep_repeats: 3
overwrite_results: false
retry_attempts: 3
retries: 3
show_images: false
close_on_exit: false
no_scorecard_submission: false
log_level: "INFO"
verbose: false
```

Run it with:

```bash
python multimain.py \
  --yaml-config experiments/multirun_example.yml \
  --max-concurrent 8   # CLI override; takes precedence over YAML
```

#### Run Manifest & Resume Behavior

Each multi-run creates (or reuses) a **manifest YAML** which tracks:

- `version`, `run_id`, `run_name`, `created_at`, `max_concurrent`
- A snapshot of the sweep definition and run options used
- A `jobs` list, where each job has:
  - `job_id`
  - `config`, `game_id`, `max_actions`, `num_plays`, `memory_limit`, `checkpoint_frequency`, `use_vision`
  - `status`: `pending`, `running`, `completed`, `failed`, or `removed`
  - `created_at`, `started_at`, `ended_at`, `updated_at`
  - `final_score`, `final_state`
  - `scorecard_url`
  - `result_file` (best-effort path to the JSON result, if `save_results_dir` is set)
  - `checkpoint_id` (card_id used for checkpointing, if a `.checkpoint/<id>/` directory exists)
  - `error` / `removed_reason` (when applicable)

By default, the manifest is written to:

- `results/multirun/<run_name>.yml`

You can override this with `--manifest-path` on initial creation, or by passing an explicit file to `--resume-from` when resuming.

##### Resuming a Multi-Run

To resume an existing multi-run, point `multimain.py` at the manifest:

```bash
python multimain.py --resume-from results/multirun/my_model_sweep.yml
```

You can also resume implicitly by **reusing a run name**:

- If you start a run with `--run-name my_model_sweep`, its manifest will be written to `results/multirun/my_model_sweep.yml`.
- On a later invocation, if you run:

  ```bash
  python multimain.py --run-name my_model_sweep
  ```

  and that manifest already exists, `multimain.py` will automatically treat this as a resume of that run and log a message indicating that it is resuming.  
  If you instead want a fresh run, either delete/rename the existing manifest file or choose a different `--run-name`.

When `--run-name` is **omitted**, `multimain.py` generates a unique UUID-based name for the run and uses it for the manifest and default results directory.

Resume rules:

- Jobs with `status == completed` are skipped.
- Jobs with a `checkpoint_id`:
  - If `.checkpoint/<checkpoint_id>/metadata.json` exists:
    - The job is scheduled with `resume_from_checkpoint=True`.
  - If the checkpoint directory is missing:
    - The job is marked `status: removed` with `removed_reason: checkpoint_missing` and is **not** scheduled (treated as not part of the run).
- Jobs without `checkpoint_id` are scheduled as fresh runs.
- Checkpoints that were not created by this multi-run (i.e., not referenced in the manifest) are never modified or inspected.

You can also combine resume with `--dry-run` to inspect the manifest:

```bash
python multimain.py \
  --resume-from results/multirun/my_model_sweep.yml \
  --dry-run
```

This prints all jobs, their configs/games, current status, and any associated checkpoint IDs without running anything.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────┐
│             Game Client                      │
│  (ARC-AGI-3 API Communication)              │
│  • HTTP client with retry logic             │
│  • Session management                       │
│  • Action execution (RESET, ACTION1-7)      │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│          Multimodal Agent                    │
│  ┌─────────────────────────────────────┐   │
│  │ 1. Convert frames to images         │   │
│  │ 2. Analyze previous results         │   │
│  │ 3. Choose next action (LLM)         │   │
│  │ 4. Convert to game command (LLM)    │   │
│  │ 5. Track costs & execute            │   │
│  │ 6. Update memory (compress if full) │   │
│  │ 7. Save checkpoint                  │   │
│  └─────────────────────────────────────┘   │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│          Provider Adapters                   │
│  (Lazy-loaded, abstract base class)         │
│  • Anthropic • OpenAI • Gemini              │
│  • DeepSeek  • Grok   • OpenRouter          │
│  • Fireworks • Groq   • XAI                 │
└─────────────────────────────────────────────┘
```

### Directory Structure

```
arc-agi-3-benchmarking/
├── src/arcagi3/              # Main package
│   ├── agent.py             # Multimodal agent (core game-playing logic)
│   ├── game_client.py       # ARC-AGI-3 API client
│   ├── checkpoint.py        # Save/resume functionality
│   ├── arc3tester.py        # High-level tester orchestrator
│   ├── cli.py               # Batch CLI for multiple games
│   ├── schemas.py           # Pydantic data models
│   ├── models.yml           # 90+ model configurations
│   ├── adapters/            # Provider-specific implementations
│   │   ├── provider.py      # Base adapter interface
│   │   ├── anthropic.py     # Claude models
│   │   ├── open_ai.py       # GPT models
│   │   ├── gemini.py        # Google Gemini
│   │   ├── deepseek.py      # DeepSeek models
│   │   ├── grok.py          # xAI Grok
│   │   ├── openrouter.py    # OpenRouter gateway
│   │   └── ...              # 9 total providers
│   └── utils/               # Utility modules
│       ├── image.py         # Grid-to-image conversion
│       ├── task_utils.py    # Config/result handling
│       ├── retry.py         # Exponential backoff
│       ├── rate_limiter.py  # Rate limiting
│       ├── metrics.py       # Performance tracking
│       └── cli.py           # CLI utilities
├── main.py                   # Single game CLI entry point
├── cli/run_all.py           # Parallel batch execution
├── scripts/example_usage.py   # Code examples
├── hints.yml                 # Optional game-specific hints
├── .checkpoint/              # Checkpoint storage (auto-created)
├── results/                  # Game results (JSON files)
└── pyproject.toml           # Package configuration
```

### Core Components

#### 1. MultimodalAgent ([agent.py](src/arcagi3/agent.py))

The core game-playing engine implementing a sophisticated three-step reasoning loop:

**Agent Workflow:**
1. **Image Processing**: Converts 64x64 game grids to 128x128 PNG images
2. **Vision Analysis**: Sends frames to multimodal LLMs for visual reasoning
3. **Three-Step LLM Reasoning**:
   - Analyze previous action results and update memory
   - Choose next high-level human action based on game state
   - Convert human intent to specific game command
4. **Memory Management**: Maintains scratchpad with word limit and automatic compression
5. **Cost Tracking**: Granular per-action token usage and cost tracking
6. **Checkpoint Support**: Auto-saves state for crash recovery

**Dual Mode Support:**
- **Vision Mode**: Sends images to multimodal models (default)
- **Text Mode**: Sends grid matrices as JSON for text-only models

#### 2. GameClient ([game_client.py](src/arcagi3/game_client.py))

HTTP client for ARC-AGI-3 API with comprehensive error handling:

**API Endpoints:**
- `GET /api/games` - List available games
- `POST /api/scorecard/open` - Create scorecard for tracking
- `POST /api/scorecard/close` - Close scorecard
- `GET /api/scorecard/{card_id}` - Get scorecard info
- `POST /api/cmd/{action}` - Execute game actions

**Game Actions:**
- ACTION1-4: Move (Up, Down, Left, Right)
- ACTION5: Perform action
- ACTION6: Click object (requires x, y coordinates)
- ACTION7: Undo
- RESET: Reset game state

#### 3. Checkpoint System ([checkpoint.py](src/arcagi3/checkpoint.py))

Robust save/resume functionality for handling crashes and interruptions:

**What's Saved:**
- Complete conversation history
- Agent memory scratchpad
- All action records with reasoning
- Cost and token usage statistics
- Session identifiers (game_id, guid, card_id)
- Previous frame images (PNG)
- Previous grids (JSON for text mode)
- Current grids for accurate resume

**Storage:** `.checkpoint/{card-id}/`

#### 4. Provider Adapter System

Abstract base class with lazy loading to avoid dependency bloat:

**Base Interface:** `ProviderAdapter` with methods for:
- `init_client()` - Initialize API client
- `call_provider()` - Call with messages
- `extract_usage()` - Get token counts
- `extract_content()` - Get response text
- `extract_json()` - Parse JSON responses

**Supported Providers:** 9 total providers with 90+ model configurations

## Configuration

### Model Configuration

Models are defined in [src/arcagi3/models.yml](src/arcagi3/models.yml) with 90+ pre-configured models:

```yaml
- name: "gpt-4o-mini-2024-07-18"
  model_name: "gpt-4o-mini-2024-07-18"
  provider: "openai"
  is_multimodal: true
  api_type: "chat_completions"
  pricing:
    date: "2025-03-15"
    input: 0.15   # per 1M tokens
    output: 0.60
  kwargs:
    temperature: 0.7
    max_tokens: 1000
```

**Configuration Fields:**
- `name`: Config identifier (use this in CLI)
- `model_name`: API model name
- `provider`: Provider adapter to use
- `is_multimodal`: Vision capability flag
- `api_type`: "chat_completions" or "responses"
- `pricing`: Input/output costs per 1M tokens
- `kwargs`: Model-specific parameters (temperature, max_tokens, etc.)

**Extended Thinking Support:**

Some models support extended reasoning (Claude thinking variants, o1-style):

```yaml
reasoning:
  effort: "high"  # or medium, low, minimal
  summary: "auto"

memory_word_limit: 1000  # Increased for thinking models
```

**Example Models Available:**
- GPT-5 family (Pro, standard, mini, nano)
- Claude 4.5 Sonnet with thinking variants (1k, 8k, 16k, 32k)
- Claude 3.7 Sonnet
- Gemini 2.0/2.5
- DeepSeek R1
- Grok-3 variants
- And many more...

### Hints Configuration

Hints for specific games can be provided via a `hints.yml` file in the project root or current working directory. Hints are automatically loaded and prepended to the system prompt at the start of all LLM calls.

**Example `hints.yml`:**
```yaml
ls20-fa137e247ce6: |
  This is a hint for game ls20-fa137e247ce6.
  
  The hint can contain multiple lines and markdown formatting:
  - Look for patterns in the grid
  - Pay attention to color changes
  
ft09-16726c5b26ff: "Single-line hint for another game."
```

The agent automatically looks for `hints.yml` in the current working directory or project root. If the file doesn't exist, no hints are used. See `hints.example.yml` for a complete example.


## Results Format

Results are saved to `{save_results_dir}/{game_id}_{config}_{timestamp}.json` with comprehensive game data:

```json
{
  "game_id": "ls20-fa137e247ce6",
  "config": "gpt-4o-mini-2024-07-18",
  "final_score": 850,
  "final_state": "WIN",
  "actions_taken": 23,
  "duration_seconds": 145.3,
  "total_cost": {
    "prompt_cost": 0.0234,
    "completion_cost": 0.0567,
    "reasoning_cost": 0.0000,
    "total_cost": 0.0801
  },
  "usage": {
    "prompt_tokens": 7800,
    "completion_tokens": 3780,
    "reasoning_tokens": 0,
    "total_tokens": 11580
  },
  "scorecard_url": "https://three.arcprize.org/scorecards/card_uuid",
  "actions": [
    {
      "action_number": 1,
      "action_type": "ACTION1",
      "reasoning": "Moving up to reach the target...",
      "result": "SUCCESS",
      "score_change": 50,
      "cost": {
        "prompt_cost": 0.0012,
        "completion_cost": 0.0028
      }
    }
  ],
  "memory_snapshot": "Final agent memory state..."
}
```

**Key Fields:**
- `scorecard_url`: Direct link to view game replay on ARC Prize platform
- `actions`: Complete action history with reasoning and per-action costs
- `memory_snapshot`: Agent's final memory state
- `reasoning_tokens`: For o1-style models with extended thinking

**Results Organization:**

```
results/
├── gpt-4o-mini-2024-07-18/
│   └── ls20-fa137e247ce6_gpt-4o-mini-2024-07-18_20251118_134935.json
├── claude-sonnet-4-5-20250929/
│   └── ls20-fa137e247ce6_claude-sonnet-4-5-20250929_20251118_140829.json
└── ...more model directories
```

## Resources

- **ARC Prize**: https://arcprize.org
- **API Documentation**: https://docs.arcprize.org/api-reference/
- **Code Examples**: See [scripts/example_usage.py](scripts/example_usage.py)

## Best Practices & Tips

### Cost Management
- Start with `--max_actions 5` for testing new models
- Use `gpt-4o-mini-2024-07-18` for development and experimentation
- Each action costs approximately $0.001-0.01 depending on the model
- Review per-action costs in result files to optimize model selection
- Use `--verbose` to see real-time cost accumulation

### Debugging & Development
- Use `--verbose` flag for detailed logs of LLM calls and reasoning
- Use `--show-images` to visualize game state in terminal
- Check `results/` directory for comprehensive JSON output
- Review action reasoning and memory state in result files
- Examine checkpoint files in `.checkpoint/{card-id}/` for state inspection

### Checkpoint Best Practices
- Checkpoints save automatically after every action (default)
- Customize save frequency with `--checkpoint-frequency N`
- Resume with `--checkpoint {card-id}` after crash/interrupt
- List all checkpoints with `--list-checkpoints`
- Clean up old checkpoints: `rm -rf .checkpoint/{card-id}`
- Use `--close-on-exit` to prevent resume (closes scorecard immediately)

### Multi-Play Strategy
- Use `--num_plays 3` to let the agent learn from multiple attempts
- Memory and patterns are preserved across plays
- Best result is returned automatically
- Execution stops early on WIN state

### Batch Processing Workflow
1. Test single game first: `python main.py --game_id "..." --config "..."`
2. If promising, run on more games: `python -m arcagi3.cli --games "id1,id2,id3"`
3. For full evaluation: `python -m arcagi3.cli --all-games`
4. For concurrent execution: Games automatically run in parallel when using `arcagi3.cli`
   - Check per-game logs in `logs/<config>/concurrent_<timestamp>/`
   - View aggregate statistics after all games complete
5. For advanced parallel execution across models: `python cli/run_all.py --game_ids "..." --model_configs "model1,model2"`

### Hints Optimization
- Create `hints.yml` with game-specific guidance
- See `hints.example.yml` for format
- Hints are automatically loaded and prepended to system prompt
- Use for providing domain knowledge or puzzle-specific strategies

### Adding New Providers
1. Create new adapter in `src/arcagi3/adapters/` inheriting from `ProviderAdapter`
2. Implement required methods: `init_client()`, `call_provider()`, etc.
3. Add to factory in `src/arcagi3/adapters/__init__.py`
4. Configure models in [models.yml](src/arcagi3/models.yml)
5. Add API key to `.env`

See existing adapters for reference implementation patterns.

## License

MIT License

---

**Version**: 0.1.0 | **Package**: `arcagi3` | **Python**: 3.9+
