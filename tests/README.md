# Tests

To run the tests, you will need to have `pytest` installed. Run the tests like this:

```bash
pytest
```

Paid OpenAI continuous-conversation tests are skipped by default. The OpenAI
implementation preserves encrypted reasoning state by replaying native Responses
output items. Run them explicitly with a real ZDR-enabled OpenAI key:

```bash
RUN_OPENAI_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_openai_continuous_conversation_live.py::test_openai_continuous_conversation_two_turn_live

RUN_OPENAI_COMPACTION_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_openai_continuous_conversation_live.py::test_openai_continuous_conversation_compaction_end_to_end_live
```

For more information on tests, please see the [tests documentation](https://arcprize.org/docs#testing).
