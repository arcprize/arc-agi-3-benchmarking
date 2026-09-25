# ARC-AGI-3 Benchmarking

## Quickstart

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.

1. Clone the arc-agi-3-benchmarking repo, enter the directory

```bash
git clone https://github.com/arcprize/arc-agi-3-benchmarking.git
cd arc-agi-3-benchmarking
```

2. Install dependencies

```bash
uv venv
uv sync
```

3. Copy .env.example to .env

```bash
cp .env.example .env
```

4. Get an API key from the [Arc Prize Website](https://arcprize.org/) and set it as an environment variable in your .env file.

```bash
ARC_API_KEY=your_api_key_here
```

5. Run the benchmarking agent against ls20.

```bash
uv run main.py --game=ls20
```


## Running the Official Benchmarking Agent

1. Get a model provider API key

Provider key links:

- [OpenAI](https://platform.openai.com/)
- [Anthropic](https://console.anthropic.com/)
- [Google Gemini](https://console.cloud.google.com/)
- [xAI](https://console.x.ai/home)
- [DeepSeek](https://console.deepseek.com/)
- [Groq](https://groq.com/)
- [OpenRouter](https://openrouter.ai/)
- [Fireworks](https://app.fireworks.ai/)

2. Set your provider keys as environment variables in your .env file.

```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
XAI_API_KEY=your_xai_key_here
GROK_API_KEY=your_grok_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
```

3. View available games (there should be 25).

```bash
uv run main.py --list-games
```

4. View available model config.

```bash
uv run main.py --list-configs
```

5. Run the official benchmarking agent against a game:

```bash
uv run main.py --game=ls20 --config=openai-gpt-5-4-2026-03-05
```

Native Anthropic configs are also available:

```bash
uv run main.py --game=ls20 --config=anthropic-opus-4-7-low
uv run main.py --game=ls20 --config=anthropic-opus-4-7-low-thinking
```

6. Or on all games:

```bash
uv run main.py --config=openai-gpt-5-4-2026-03-05
```

## Standard and Provider Adapter harnesses

ARC-AGI-3 games span many model calls, so the harness must determine what
carries forward between actions. The Standard harness uses a provider-neutral
text history and asks the model to preserve useful discoveries in visible
notes. These configurations use `manual_rolling`.

The Provider Adapter harness uses the provider's native conversation and
reasoning state. It uses native compaction when the provider supplies it and a
domain-neutral harness summary otherwise. These configurations use
`continuous_conversation`; `openai-gpt-5-6-sol-max-provider-adapter` and
`google-gemini-3-8-flash-low-provider-adapter` are examples.
The `xai-grok-4-6-low-provider-adapter` profile adds xAI-native encrypted
reasoning replay and separate Responses compaction using `XAI_API_KEY`.
See the [xAI Provider Adapter](docs/xai-provider-adapter.md) for its request
contract, compaction boundaries, and opt-in live tests.

The Anthropic Provider Adapter profile is `anthropic-opus-5-low-provider-adapter`.
It uses Opus 5 at low reasoning effort, preserves native thinking blocks between
actions, and uses Anthropic's native on-demand compaction at a 175k completed-
context threshold. It summarizes completed history before presenting the next
frame, so the newest observation reaches the action request unchanged. It records
provider-reported thinking-token usage when available without double-counting
output tokens. The profile also enables Anthropic automatic prompt caching with
the default 5-minute TTL so repeated native history can be reused. Set
`ANTHROPIC_API_KEY` to use it:

```bash
uv run main.py --game=ls20 --config=anthropic-opus-5-low-provider-adapter
```

See [runtime state adapters](docs/runtime-state.md) for replay, compaction,
recording, and data-handling details.

DeepSeek thinking mode can opt into `deepseek.chat_completions.v1`. The adapter
uses DeepSeek's documented tools protocol, replays exact `reasoning_content`, and
uses harness summary compaction. See the
[DeepSeek Provider Adapter guide](docs/deepseek-provider-adapter.md).

Both harnesses use the same games, actions, limits, and scoring. The Standard
harness supports controlled comparisons across providers, while the Provider
Adapter harness measures performance using provider-native context management.
Their results should be reported separately and clearly labeled.

7. View your scorecard

When you run a benchmark, a scorecard is saved on the ARC server. If you are logged in, you can browse your saved scorecards at [arcprize.org/scorecards](https://arcprize.org/scorecards).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
