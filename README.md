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

5. Run the random agent (generates random actions) against ls20.

```bash
uv run main.py --agent=random --game=ls20
```


## Running a Benchmarking Agent

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

3. View available agents.

```bash
uv run main.py --list-agents
```

4. View available games (there should be 25).

```bash
uv run main.py --list-games
```

5. View available model config.

```bash
uv run main.py --list-configs
```

6. Run an agent against a game:

```bash
uv run main.py \
  --agent conversationrollingwindow \
  --game ls20 \
  --config openai-gpt-5-4-2026-03-05-high
```

7. Or on all games:

```bash
uv run main.py \
  --agent conversationrollingwindow \
  --config openai-gpt-5.4-openrouter
```

8. View your scorecard

When you run a benchmark, a scorecard is saved on the ARC server. If you are logged in, you can browse your saved scorecards at [arcprize.org/scorecards](https://arcprize.org/scorecards).

## Contest Submission

To submit your agent for the ARC-AGI-3 competition, please use this form: https://forms.gle/wMLZrEFGDh33DhzV9.

## Contributing

We welcome contributions! To contribute to ARC-AGI-3-Benchmarking, please follow these steps:

1.  Fork the repository and create a new branch for your feature or bugfix.
2.  Make your changes and ensure that all tests pass, you are welcome to add more tests for your specific fixes.
3.  This project uses `ruff` for linting and formatting. Please set up the pre-commit hooks to ensure your contributions match the project's style.
    ```bash
    pip install pre-commit
    pre-commit install
    ```
4.  Write clear commit messages describing your changes.
5.  Open a pull request with a description of your changes and the motivation behind them.

If you have questions or need help, feel free to open an issue.

## Tests

To run the tests, you will need to have `pytest` installed. Run the tests like this:

```bash
pytest
```

For more information on tests, please see the [tests documentation](https://three.arcprize.org/docs#testing).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
