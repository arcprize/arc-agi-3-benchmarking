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

To exercise automatic compaction with a small test-local 5k threshold, run:

```bash
RUN_OPENAI_COMPACTION_LIVE_TESTS=1 uv run pytest -q -m integration \
  tests/integration/test_openai_encrypted_replay_live.py \
  -k compaction_end_to_end
```

This test calibrates its input with OpenAI's token-count endpoint, makes one
response below the threshold, crosses the threshold on the next response,
prunes to the returned encrypted compaction item, and makes a final request
proving that earlier context survived. The checked-in model configuration keeps
its production 175k threshold.
