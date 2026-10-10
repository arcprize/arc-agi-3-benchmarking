# Gemini decode continuation

The standard Gemini `google.generate_content.v1` adapter automatically follows non-streaming decode continuation responses. No enable flag is needed. Set these limits to a model's `request`
configuration alongside its model name:

```yaml
max_output_tokens: 1000000
http_options:
  timeout: 3600000
```

The output budget is cumulative across all decode slices. HTTP timeouts are in
milliseconds and apply per request; they are not a whole-turn deadline. The
continuation path defaults to a 3600-second timeout if none is specified.

When a candidate finishes with `CONTINUATION`, the adapter resends its token at
the top level of the next request with the original contents and accumulated raw
model parts. Opaque signatures are preserved. It returns one model response only
after `STOP` or terminal `MAX_TOKENS`; slices do not create extra game actions.
Thought text is kept separate from action text. Prompt, output, reasoning, and
cached token counts are summed across slices using the existing v3 accounting
convention, where output tokens include reasoning tokens.

The aggregated raw response includes the original responses in `slices` for
inspection. Logs report each slice's finish reason without printing its token.
Missing/repeated tokens, missing candidates, and unexpected terminal reasons
raise an invalid-response error. Failures after completed slices retain observed
usage on the error. Failed requests may have additional usage that the server did
not return. SDK retries are disabled for these requests; existing harness retry
and compaction policies still apply. Continuation state is in memory only and is
not resumed after process failure.

Ordinary single-response requests do not require an explicit output budget. A
continuation response without an explicit positive budget fails with observed usage
rather than starting another request with an undefined budget.

Continuation is not supported on the
Interactions API / Gemini Provider Adapter; that API needs a separately confirmed
continuation contract. No private model configuration or live evaluation is
included with this change. Tests simulate the protocol through the installed SDK;
a local ls20 run also verified CONTINUATION followed by STOP and a saved game
action without retries.

## Recoverable request failures

The Google adapter classifies SDK `code` and `status_code` values for transient
HTTP failures, including 500 and 503, plus HTTPX timeouts and network errors.
This lets summary compaction use its existing ten-failure retry budget instead
of aborting on an unclassified Google exception.

Action requests also allow up to ten classified transient failures per turn,
independently of the invalid-response budget (`MAX_RETRIES + 1`, normally four).
Backoff starts at 0.25 seconds, doubles, and caps at 30 seconds. Invalid action
responses and malformed summary responses also back off before retrying.
Failed action responses are retained in diagnostic artifacts, and observed
usage is accumulated across attempts. Provisional conversation state is only
committed after an action passes validation.

Malformed provider output is never salvaged as a game action. A terminal
malformed response still fails after the bounded invalid-response budget.
Failures after completed continuation slices retain their existing usage-aware
invalid-response handling; this change does not resume a failed slice in place.
These retries do not extend or recover an expired game session. They also do
not guarantee recovery from a persistent provider failure, and another model
request can incur additional usage.
