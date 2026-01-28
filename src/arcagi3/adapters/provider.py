import abc
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from arcagi3.schemas import Attempt, ModelCallRecord, ModelConfig
from arcagi3.utils.context import SessionContext
from arcagi3.utils.task_utils import read_models_config

logger = logging.getLogger(__name__)


class ProviderAdapter(abc.ABC):
    """
    Base class for all provider adapters.

    INTERNAL_API_PARAMS: Set of parameter names that are application-level configuration
    and should NOT be passed to provider APIs. These are filtered out before API calls.
    """

    INTERNAL_API_PARAMS: Set[str] = {"memory_word_limit"}

    def __init__(self, config: str):
        """
        Initialize the provider adapter with model configuration.

        Args:
            config: Configuration name that identifies the model and its settings
        """
        self.config = config
        self.model_config: ModelConfig = read_models_config(config)

        # Verify the provider matches the adapter
        adapter_provider = self.__class__.__name__.lower().replace("adapter", "")
        if adapter_provider != self.model_config.provider:
            raise ValueError(
                f"Model provider mismatch. Config '{config}' is for provider '{self.model_config.provider}' but was passed to {self.__class__.__name__}"
            )

        # Initialize the client
        self.client = self.init_client()

    @abc.abstractmethod
    def init_client(self):
        """
        Initialize the client for the provider. Each adapter must implement this.
        Should handle API key validation and client setup.
        """
        pass

    @abc.abstractmethod
    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        """
        Make a prediction with the model and return an Attempt object's answer

        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
            pair_index: Optional pair index to include in metadata
        The implementation should ensure that the config name is included in the metadata.
        """
        pass

    def chat_completion(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = []
    ) -> Any:
        """
        Make a raw API call to the provider and return the response
        """
        # Base implementation can raise or be empty, subclasses should override if needed
        # OpenAI-style adapters use _chat_completion internally.
        raise NotImplementedError("Subclasses must implement chat_completion if used directly.")

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """
        Extract JSON from various possible formats in the response
        """
        pass

    @abc.abstractmethod
    def extract_usage(self, response: Any) -> tuple[int, int, int]:
        """
        Extract token usage from provider response.

        Returns:
            Tuple of (prompt_tokens, completion_tokens, reasoning_tokens)
        """
        pass

    @abc.abstractmethod
    def extract_content(self, response: Any) -> str:
        """
        Extract text content from provider response
        """
        pass

    @abc.abstractmethod
    def call_provider(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Call provider with retry logic
        """
        pass

    def call_with_tracking(
        self,
        context: SessionContext,
        messages: List[Dict[str, Any]],
        *,
        step_name: Optional[str] = None,
    ) -> Any:
        """
        Call provider and append usage/cost into the invocation SessionContext.

        This is the preferred entrypoint for agents so accounting is always scoped
        to the current invocation context (not the agent instance).
        """
        response = self.call_provider(messages)
        prompt_tokens, completion_tokens, reasoning_tokens = self.extract_usage(response)
        context.add_usage_and_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            pricing=self.model_config.pricing,
        )
        try:
            serialized_messages = json.loads(json.dumps(messages, default=str))
            response_text = self.extract_content(response)
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "reasoning_tokens": int(reasoning_tokens),
            }
            cost = None
            if self.model_config.pricing is not None:
                input_cost_per_token = (
                    float(getattr(self.model_config.pricing, "input")) / 1_000_000
                )
                output_cost_per_token = (
                    float(getattr(self.model_config.pricing, "output")) / 1_000_000
                )
                prompt_cost = usage["prompt_tokens"] * input_cost_per_token
                completion_cost = usage["completion_tokens"] * output_cost_per_token
                reasoning_cost = usage["reasoning_tokens"] * output_cost_per_token
                cost = {
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "reasoning_cost": reasoning_cost if usage["reasoning_tokens"] > 0 else 0.0,
                    "total_cost": prompt_cost + completion_cost + reasoning_cost,
                }

            action_num = None
            try:
                action_num = int(context.game.action_counter) + 1
            except Exception:
                action_num = None

            record = ModelCallRecord(
                step_name=step_name,
                action_num=action_num,
                provider=self.model_config.provider,
                model=self.model_config.model_name,
                messages=serialized_messages,
                response=str(response_text) if response_text is not None else None,
                usage=usage,
                cost=cost,
                timestamp=datetime.now(timezone.utc),
            )
            context.append_model_call(record)
        except Exception:
            logger.exception("Failed to capture provider call history for checkpointing.")
        return response
