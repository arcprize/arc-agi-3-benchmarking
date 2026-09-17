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

Anthropic Provider Adapter unit tests use synthetic responses and mocked HTTP
through the pinned Anthropic SDK. They cover native replay, streaming deltas,
signed on-demand compaction, current-frame and buffered-input isolation, SDK
replay-field cleanup, refusal details, reported thinking-token accounting,
failed-attempt usage, and opaque-state redaction. Regression coverage also checks
saved allowlisted failure diagnostics through the pinned SDK's streaming and
non-streaming transports, and native compaction policy settings in run metadata:

```bash
uv run pytest -q tests/unit/test_anthropic_runtime.py tests/unit/test_benchmarking_agent.py
```

Paid Opus 5 low tests are skipped by default and require `ANTHROPIC_API_KEY`.
They use synthetic prompts, not benchmark game data. The two-turn and single-
compaction tests cap action output at 4k tokens. The multi-compaction test caps
action output at 8k tokens and checks signed thinking and two remembered results
across two native compaction boundaries. All compaction calls have an 8k output
cap. Compaction tests lower the harness trigger to one token so small completed
histories compact deterministically; on-demand compaction has no 50k minimum and
these tests no longer send large padding prompts. They capture native requests
in memory to verify that fresh observations are absent from summary requests
and arrive unchanged in action requests. Compaction is still a separate paid
call. Each gate is separate; run it only when intended:

```bash
RUN_ANTHROPIC_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_anthropic_continuous_conversation_live.py::test_anthropic_continuous_conversation_two_turn_live

RUN_ANTHROPIC_COMPACTION_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_anthropic_continuous_conversation_live.py::test_anthropic_continuous_conversation_compaction_live

RUN_ANTHROPIC_MULTI_COMPACTION_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_anthropic_continuous_conversation_live.py::test_anthropic_continuous_conversation_two_compactions_live
```

Mocked tests establish transport and state behavior, not provider acceptance or
summary quality. Live tests are a separate verification boundary and do not
launch an ARC benchmark.

Paid Google continuous-conversation tests are also skipped by default. Run
them explicitly with a paid Gemini project and `GOOGLE_API_KEY`:

```bash
RUN_GOOGLE_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_google_continuous_conversation_live.py::test_google_continuous_conversation_two_turn_live

RUN_GOOGLE_COMPACTION_LIVE_TESTS=1 uv run pytest -q \
  tests/integration/test_google_continuous_conversation_live.py::test_google_harness_summary_compaction_live
```

For more information on tests, please see the [tests documentation](https://arcprize.org/docs#testing).
