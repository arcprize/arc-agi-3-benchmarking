import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

import anthropic
from dotenv import load_dotenv

from arcagi3.schemas import (Attempt, AttemptMetadata, Choice,
                             CompletionTokensDetails, Cost, Message,
                             StreamResponse, Usage)
from arcagi3.utils.retry import retry_with_exponential_backoff

from .provider import ProviderAdapter

load_dotenv()

logger = logging.getLogger(__name__)

class AnthropicAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Anthropic model
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        return client
    
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Anthropic model and return an Attempt object

        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.now(timezone.utc)

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Check if streaming is enabled
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)

        if stream_enabled:
            response = self.chat_completion_stream(messages)
        else:
            response = self.chat_completion(messages)

        end_time = datetime.now(timezone.utc)

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000  # Convert from per 1M tokens
        output_cost_per_token = self.model_config.pricing.output / 1_000_000  # Convert from per 1M tokens
        
        prompt_cost = response.usage.input_tokens * input_cost_per_token
        completion_cost = response.usage.output_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=i,
                message=Message(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
            for i, msg in enumerate(messages)
        ]

        # Convert Anthropic response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role="assistant",
                    content=content.text if content.type == "text" else json.dumps(content.input)
                )
            )
            for content in response.content
            if content.type in ["text", "tool_use"]
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Thinking blocks from Anthropic
        reasoning_summary = self._get_reasoning_summary(response)

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,  # Use kwargs from model config
            reasoning_summary=reasoning_summary,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Anthropic doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.output_tokens,
                    rejected_prediction_tokens=0  # Anthropic doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            ),
            task_id=task_id,  # Add task_id to metadata
            pair_index=pair_index,  # Add pair_index to metadata
            test_id=test_id  # Add test_id to metadata
        )

        # Incase there is a thinking block
        answer = ""
        for content in response.content:
            if content.type == "text":
                answer = content.text
                break

        attempt = Attempt(
            metadata=metadata,
            answer=answer
        )

        return attempt

    def chat_completion(self, messages, tools=[]):
        """
        Make a raw API call to Anthropic and return the response
        """

        return self.client.messages.create(
            model=self.model_config.model_name,
            messages=messages,
            tools=tools,
            **self.model_config.kwargs
        )

    def chat_completion_stream(self, messages, tools=[]):
        """
        Make a streaming API call to Anthropic and return the final complete response.
        Only the final message is returned; intermediate deltas are ignored.
        """
        logger.debug(f"Starting streaming for Anthropic model: {self.model_config.model_name}")

        # Prepare kwargs for streaming, removing 'stream' to avoid duplication
        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}

        try:
            # Create the stream
            with self.client.messages.stream(
                model=self.model_config.model_name,
                messages=messages,
                tools=tools,
                **stream_kwargs
            ) as stream:
                # Accumulate the complete message
                # The stream context manager handles all the event processing
                # and gives us the final message when done
                final_message = stream.get_final_message()

            logger.debug(f"Streaming complete for message ID: {final_message.id}")
            return final_message

        except Exception as e:
            logger.error(f"Error during Anthropic streaming: {e}")
            raise

    def _get_reasoning_summary(self, response: Any) -> str:
        """Get the reasoning summary from the response."""
        reasoning_summary = None
        thinking_texts: List[str] = []
        try:
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "thinking" and hasattr(block, 'thinking'):
                        if isinstance(block.thinking, str): # Ensure it's a string
                            thinking_texts.append(block.thinking)
            if thinking_texts:
                reasoning_summary = "\n\n".join(thinking_texts)
        except Exception as e:
            logger.warning(f"Error extracting thinking blocks from Anthropic response: {e}", exc_info=True)
        return reasoning_summary

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        tools = [
            {
                "name": "extract_json",
                "description": "Extracts JSON from the response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "description": "A list of lists of integers extracted from the response."
                        }
                    },
                    "required": ["response"]
                }
            }
        ]

        text = f"Extract JSON of the test output from the following response: {input_response}"

        query = f"""
        <document>
        {text}
        </document>

        Use the extract_json tool.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": query}],
            tools=tools
        )

        json_response = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "extract_json":
                json_entities = content.input
                break

        if json_entities:
            return json_entities['response']
        else:
            return None
        
    def extract_usage(self, response):
        # Handle consumed streams
        if isinstance(response, StreamResponse):
            return response.prompt_tokens, response.completion_tokens, 0
        
        # Check if it's an unconsumed stream
        if 'Stream' in str(type(response)):
            # For streams, we can't get usage info
            # Return 0,0,0 for now - usage will need to be tracked differently
            return 0, 0, 0
        
        if hasattr(response, 'usage'):
            # Anthropic format (no separate reasoning tokens)
            if hasattr(response.usage, 'input_tokens'):
                return response.usage.input_tokens, response.usage.output_tokens, 0
        return 0, 0, 0
    
    def extract_content(self, response):
        if hasattr(response, 'content') and response.content:
            text_parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            return ''.join(text_parts)
        logger.warning(f"Unknown response format. Type: {type(response)}")
        return ""
    
    @retry_with_exponential_backoff(max_retries=3)
    def call_provider(self, messages):
        # Anthropic requires system messages as separate parameter, not in messages array
        # Extract system messages and filter them out
        system_messages = []
        filtered_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                # Anthropic system can be string or list of content blocks
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_messages.append(content)
                elif isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        system_messages.append("\n".join(text_parts))
            else:
                # Convert image blocks for Anthropic
                msg_copy = dict(msg)
                msg_copy["content"] = self._convert_image_blocks_for_anthropic(msg.get("content"))
                filtered_messages.append(msg_copy)
        
        # Combine system messages
        system_content = "\n".join(system_messages) if system_messages else None
        
        # Prepare kwargs
        anthropic_kwargs = dict(self.model_config.kwargs)
        if system_content:
            anthropic_kwargs["system"] = system_content

        # Stream call
        if anthropic_kwargs.get("stream", False):
            logger.info("Anthropic streaming enabled — consuming stream...")
            stream_kwargs = {k: v for k, v in anthropic_kwargs.items() if k != "stream"}

            with self.client.messages.stream(
                model=self.model_config.model_name,
                messages=filtered_messages,
                **stream_kwargs
            ) as stream:
                final_message = stream.get_final_message()

            logger.info("Anthropic final message received.")
            return final_message

        # Normal call
        return self.client.messages.create(
            model=self.model_config.model_name,
            messages=filtered_messages,
            **anthropic_kwargs
        )
    
    def _convert_image_blocks_for_anthropic(self, content: Any) -> Any:
        """Convert OpenAI-style image_url blocks to Anthropic format"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            converted = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        # Convert from OpenAI format to Anthropic format
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "")
                        
                        # Extract base64 data from data URL
                        if url.startswith("data:image/png;base64,"):
                            base64_data = url[len("data:image/png;base64,"):]
                            converted.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data
                                }
                            })
                        else:
                            # Keep as is if not base64
                            converted.append(block)
                    else:
                        converted.append(block)
                else:
                    converted.append(block)
            return converted
        return content
        
if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-3-5-sonnet-20240620")
    logger.debug(f"Extract JSON test result type: {type(adapter.extract_json_from_response('[[1, 2, 3], [4, 5, 6]]'))}")