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

## Live Anthropic encrypted-replay tests

The Anthropic smoke test uses the checked-in
`anthropic-opus-5-encrypted-replay-medium` profile. It is skipped during
normal test runs. Put `ANTHROPIC_API_KEY` in the ignored `.env` file (or export
it), then run:

```bash
RUN_ANTHROPIC_LIVE_TESTS=1 uv run pytest -q -m integration \
  tests/integration/test_anthropic_encrypted_replay_live.py \
  -k two_turn
```

This makes two paid API requests and verifies summarized thinking, encrypted
thinking signatures, exact client-owned replay, and the production 175k
compaction configuration.

To exercise native Anthropic compaction at its 50k minimum threshold, run:

```bash
RUN_ANTHROPIC_COMPACTION_LIVE_TESTS=1 uv run pytest -q -m integration \
  tests/integration/test_anthropic_encrypted_replay_live.py \
  -k compaction_end_to_end
```

This is a higher-cost three-request gate. It calibrates real input token counts,
crosses the threshold, verifies the returned compaction block and optional
encrypted metadata plus per-iteration usage, prunes the client-owned history,
and replays the compacted state to prove that an earlier memory survives. The test lowers only
`max_tokens` and the trigger threshold; production stays at 120k output tokens
and a 175k compaction trigger.

For an Opus 5 pressure test that produces signed reasoning on both sides of a
compaction, crosses the 50k threshold twice, verifies exact replay inputs, and
recalls two memory tokens plus two computed results after the second boundary:

```bash
RUN_ANTHROPIC_MULTI_COMPACTION_LIVE_TESTS=1 uv run pytest -q -m integration \
  tests/integration/test_anthropic_encrypted_replay_live.py \
  -k reasoning_survives_two_live_compaction_boundaries
```

This makes five paid Messages API requests plus token-count calibration calls.
