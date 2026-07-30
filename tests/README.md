# Tests

To run the tests, you will need to have `pytest` installed. Run the tests like this:

```bash
pytest
```

For more information on tests, please see the [tests documentation](https://arcprize.org/docs#testing).

## Live OpenAI smoke test

The encrypted-replay integration test is skipped during normal test runs. To
run it explicitly, place `OPENAI_API_KEY` in the repository's ignored `.env`
file (or export it in the shell), then run:

```bash
RUN_OPENAI_LIVE_TESTS=1 uv run pytest -q -m integration \
  tests/integration/test_openai_encrypted_replay_live.py
```

This makes two paid API requests. It verifies `store=false`, encrypted reasoning
output, client-side replay into a second turn, and acceptance of the production
server-side compaction configuration.
