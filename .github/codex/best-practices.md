# Best practices
This public repo is the official harness for ARC-AGI-3 benchmark runs. It contains general and provider-specific code for evaluating different AI models on the ARC-AGI-3 games.

## Context
- Games must be played through the ARC-AGI toolkit (the `arc-agi` library on PyPI) in online mode against a configurable server endpoint.
- The Standard harness keeps a growing conversation of frames and model messages. When the estimated token count exceeds `MAX_CONTEXT_LENGTH`, the oldest turns are trimmed. The action to take is parsed from the end of the model's message. The rest of the content is tracked as "reasoning".
- The provider adapter harnesses allow features like encrypted reasoning and compaction. These carry forward a richer "state" than the model's raw outputs.
- The agent supports "rehydration", which resumes a run that ended prematurely by replaying all original actions and restoring runtime context and state.

## Rules
- This repo is public. Check docs, comments, and user-facing strings for typos and grammar mistakes.
- Write diagnostic data to disk.
- Flag changes that might unintentionally break other providers. Implementations should be as provider-neutral as possible.
- Flag changes that might unintentionally break rehydration, such as provider adapter state that is not saved or restored.
- Document every new or changed model config pattern by adding an entry that uses it to `benchmarking/model_configs.yaml`, which serves as the example. Call out these changes in your review.
- Gracefully handle SIGINT and SIGTERM, and API errors on any outbound request.
- Fail fast when the environment is misconfigured.
- Security: never commit, log, or write API keys, tokens, or request headers to disk. CI logs and artifacts are public.
