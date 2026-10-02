# Local Gemini provider continuation experiment

The generateContent continuous-conversation adapter preserves complete native
Content parts (including signatures) across accepted turns, and reuses decode
continuation within a turn. Rejected responses do not advance accepted state.
Harness summary compaction resets the summarized prefix while restoring any
excluded recent native turns exactly. Thought summaries appear in readable
transcripts; opaque signatures do not.

The Interactions continuation loop is an UNCONFIRMED local experiment. It assumes:

- a top-level `status: continuation` or `finish_reason: CONTINUATION` signal;
- a top-level `continuation_token` (also recognizes camel-case response fields);
- a follow-up create request with original input and a top-level
  `continuation_token` sent via SDK extra_body;
- incremental output steps and usage per slice, and an unchanged cumulative
  generation_config.max_output_tokens budget;
- final status `completed` before accepting the aggregated response.

These assumptions are not an API specification. Mocked SDK transport tests show
client serialization and aggregation only. A successful ordinary Interactions
turn does not validate decode continuation. The experiment deliberately does not
interpret `incomplete` or a timeout as permission to resume or create a new turn.
Missing/repeated tokens and failed follow-ups fail with observed usage.

Two local-only profiles are appended to model_configs.yaml, one per API, with
low thinking, a 1m output budget, a one-hour request timeout, 175k summary trigger,
and the same input capacity as the existing Gemini provider profile. Input
capacity and model access still need live confirmation. Pricing is unknown; zero
rates are placeholders. The Interactions timeout is in seconds; generateContent
http_options.timeout is in milliseconds.

Run each game in a separate terminal with separate redirected console logs;
logs.log is shared and may be overwritten by concurrent processes. No live
benchmark is launched by these changes. Neither implementation checkpoints
opaque state for process restart.
