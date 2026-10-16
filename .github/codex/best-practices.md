# Best practices
This public repo is the official harness for ARC-AGI-3 benchmark runs. It contains general and model provider specific code for evaluating different AI models on the ARC-AGI-3 games. 

## Important details
- All content is public and well known. Check for typos and grammar mistakes.
- Games must be played through the ARC-AGI toolkit (arc-agi library in pypi) in online mode against a configurable server endpoint.
- The standard harness consists of a rolling window of the last n (default 10) frames and messages from the model. The action to take is parsed from the end of the models message. The rest of the content is tracked as "reasoning". 
- The provider adapter harnesses allow features like encrypted reasoning and compaction. These carry forward a richer "state" than the model's raw outputs.
- The agent contains a feature for "rehydration" which allows a previous run that may have ended prematurely to be resumed by replaying all original actions and runtime context/state. Anytime a change occurs to a provider adapter, ensure it will be supported for rehydration. 
- Standard out and files written to disk on the host running this benchmark are both appropriate for observability and logging. 
- Ensure changes do not break any providers, implementations should be as neutral as possible.
- Document all new or changed model_config patterns in the model_configs.yaml file. Call these out in your comments or reviews to attract attention to them.
- Gracefully handle SIGINT and SIGTERM events, API errors on any outbound request. 
- Opt to fail fast when an environment is misconfigured.
