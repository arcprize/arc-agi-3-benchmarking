import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from arcagi3.schemas import (Attempt, AttemptMetadata, Choice,
                             CompletionTokensDetails, Cost, Message,
                             StreamResponse, Usage)
from arcagi3.utils.retry import retry_with_exponential_backoff

from .provider import ProviderAdapter

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiAdapter(ProviderAdapter):
    def init_client(self):
        """Initialize the Gemini client."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.generation_config_dict = self.model_config.kwargs
        
        client = genai.Client(api_key=api_key)
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Gemini model and return an Attempt object.
        
        Args:
            prompt: The prompt to send to the model.
            task_id: Optional task ID.
            test_id: Optional test ID.
            pair_index: Optional index for paired data.
        """
        start_time = datetime.now(timezone.utc)
        
        messages = [{"role": "user", "content": prompt}] 
        response = self.chat_completion(messages)
        
        if response is None:
            logger.error(f"Failed to get response from chat_completion for task {task_id}")
            # Create a default Attempt object to signify failure
            default_usage = Usage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0, accepted_prediction_tokens=0, rejected_prediction_tokens=0
                )
            )
            default_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
            
            return Attempt(
                metadata=AttemptMetadata(
                    model=self.model_config.model_name,
                    provider=self.model_config.provider,
                    start_timestamp=start_time,
                    end_timestamp=datetime.now(timezone.utc),
                    choices=[], 
                    kwargs=self.model_config.kwargs, 
                    usage=default_usage,
                    cost=default_cost,
                    error_message="Failed to get valid response from provider",
                    task_id=task_id, pair_index=pair_index, test_id=test_id
                ),
                answer=""
            )

        end_time = datetime.now(timezone.utc)

        usage_metadata = getattr(response, 'usage_metadata', None)
        logger.debug(f"Response usage metadata: {usage_metadata}")
        
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        reasoning_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) if usage_metadata else 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0
        
        response_text = getattr(response, 'text', "")

        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000

        prompt_cost = input_tokens * input_cost_per_token
        completion_cost = output_tokens * output_cost_per_token
        reasoning_cost = reasoning_tokens * output_cost_per_token

        input_choices = [
            Choice(index=i, message=Message(role=msg["role"], content=msg["content"]))
            for i, msg in enumerate(messages)
        ]
        response_choices = [
            Choice(index=len(input_choices), message=Message(role="assistant", content=response_text))
        ]
        all_choices = input_choices + response_choices

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens,
                    accepted_prediction_tokens=output_tokens,
                    rejected_prediction_tokens=0
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                reasoning_cost=reasoning_cost,
                total_cost=prompt_cost + completion_cost + reasoning_cost
            ),
            task_id=task_id, pair_index=pair_index, test_id=test_id
        )
        attempt = Attempt(metadata=metadata, answer=response_text)
        return attempt

    def _convert_content_to_gemini_parts(self, content):
        """
        Convert OpenAI-style content blocks to Gemini Part objects.
        
        Handles:
        - String content -> Part(text=content)
        - List of blocks -> List of Part objects
        - image_url blocks -> Part(inline_data=Blob(...))
        
        Args:
            content: Can be a string or a list of content blocks
            
        Returns:
            List of types.Part objects
        """
        from google.genai.types import Blob
        parts = []
        
        if isinstance(content, str):
            # Simple string content
            parts.append(types.Part(text=content))
        elif isinstance(content, list):
            # Multimodal content (text + images)
            image_count = 0
            text_count = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(types.Part(text=block.get("text", "")))
                        text_count += 1
                    elif block.get("type") == "image_url":
                        # Convert OpenAI-style image_url to Gemini format
                        image_url = block.get("image_url", {})
                        if isinstance(image_url, dict):
                            base64_data = image_url.get("url") or image_url.get("data", "")
                            media_type = image_url.get("media_type", "image/png")
                        else:
                            base64_data = str(image_url)
                            media_type = "image/png"
                        
                        # Remove data URL prefix if present
                        if base64_data.startswith("data:"):
                            base64_data = base64_data.split(",", 1)[1]
                        
                        # Convert base64 string to bytes
                        try:
                            image_bytes = base64.b64decode(base64_data)
                            parts.append(types.Part(
                                inline_data=Blob(
                                    data=image_bytes,
                                    mime_type=media_type
                                )
                            ))
                            image_count += 1
                            logger.debug(f"Converted image block to Gemini Part ({len(image_bytes)} bytes, {media_type})")
                        except Exception as e:
                            logger.error(f"Failed to decode base64 image: {e}")
                    elif "inline_data" in block:
                        # Handle pre-converted inline_data format
                        inline_data = block["inline_data"]
                        if isinstance(inline_data, dict):
                            parts.append(types.Part(
                                inline_data=Blob(
                                    data=inline_data.get("data"),
                                    mime_type=inline_data.get("mime_type", "image/png")
                                )
                            ))
                            image_count += 1
                elif isinstance(block, str):
                    parts.append(types.Part(text=block))
                    text_count += 1
            
            if image_count > 0:
                logger.info(f"Converted {image_count} image(s) and {text_count} text block(s) to Gemini Parts")
        else:
            logger.warning(f"Unexpected content type: {type(content)}")
        
        return parts

    def chat_completion(self, messages: list):
        contents_list = []
        system_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                role = "model"  # Gemini uses 'model' for assistant responses
            
            if role in ["user", "model"]:
                # Check if content contains images
                has_images = False
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            has_images = True
                            break
                
                parts = self._convert_content_to_gemini_parts(content)
                if parts:
                    contents_list.append(types.Content(role=role, parts=parts))
                    if has_images:
                        logger.debug(f"Added {len(parts)} Parts to {role} message")
            elif role == "system" and content:
                # Extract system messages to be used as system_instruction
                if isinstance(content, str):
                    system_messages.append(content)
                elif isinstance(content, list):
                    # Handle multimodal system content
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        system_messages.append("\n".join(text_parts))

        config_params = self.generation_config_dict.copy()
        
        # Combine system messages and add to config if not already present
        if system_messages and 'system_instruction' not in config_params:
            system_content = "\n".join(system_messages)
            config_params['system_instruction'] = system_content

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name,
                contents=contents_list,
                config=types.GenerateContentConfig(**config_params)
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat_completion with google.genai: {e}")
            if hasattr(e, 'response') and e.response:
                 logger.error(f"API Error details: {e.response}")
            return None

    def extract_json_from_response(self, input_response: str) -> Optional[List[List[int]]]:
        prompt = f"""
        Extract only the JSON of the test output from the following response.
        Remove any markdown code blocks and return only valid JSON.

        Response:
        {input_response}

        The JSON should be in this format:
        {{
            "response": [
                [1, 2, 3],
                [4, 5, 6]
            ]
        }}
        """
        
        # Filter config for extraction, using common generation parameters.
        # System instructions are generally not needed for this type of extraction.
        extract_config_params = {
            k: v for k, v in self.generation_config_dict.items() 
            if k in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences']
        }

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name,
                contents=prompt, 
                config=types.GenerateContentConfig(**extract_config_params) if extract_config_params else None
            )
            content = response.text.strip()

            if content.startswith("```json"):
                content = content[7:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            try:
                json_data = json.loads(content)
                return json_data.get("response")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from extraction response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error in extract_json_from_response with google.genai: {e}")
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
        
        # Gemini format (no separate reasoning tokens)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
            return prompt_tokens, completion_tokens, 0
        return 0, 0, 0
    
    def extract_content(self, response):
        if hasattr(response, 'text'):
            text = response.text
            if text is None:
                logger.warning("Gemini returned None content")
                return ""
            return text
        logger.warning(f"Unknown response format. Type: {type(response)}")
        return ""
    
    @retry_with_exponential_backoff(max_retries=3)
    def call_provider(self, messages):
        # GeminiAdapter handles message conversion internally
        return self.chat_completion(messages)