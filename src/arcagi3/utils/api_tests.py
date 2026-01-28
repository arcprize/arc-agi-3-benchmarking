"""
API key testing functions for the --check command.
"""
import os
from typing import Callable, Dict, List, Optional, Tuple

from arcagi3.game_client import GameClient


def is_placeholder_key(api_key: str) -> bool:
    """Check if an API key is a placeholder/example value."""
    if not api_key:
        return True
    api_key_lower = api_key.lower().strip()
    placeholder_patterns = [
        "your",
        "example",
        "placeholder",
        "replace",
        "sk-0000",
        "sk-test",
        "xxx",
        "demo",
        "sample",
    ]
    return any(pattern in api_key_lower for pattern in placeholder_patterns)


def test_arc_api_key() -> Tuple[bool, str, Optional[List[Dict[str, str]]]]:
    """Test ARC API key by attempting to list games. Returns (status, message, games_list)."""
    api_key = os.environ.get("ARC_API_KEY")
    if not api_key:
        return False, "✗ Missing API key", None
    if is_placeholder_key(api_key):
        return None, "Not configured", None

    try:
        client = GameClient()
        games = client.list_games()
        return True, f"✓ Connected (found {len(games)} games)", games
    except ValueError as e:
        if "ARC_API_KEY" in str(e):
            return False, "✗ Missing API key", None
        return False, f"✗ {str(e)}", None
    except Exception as e:
        return False, f"✗ {str(e)[:50]}", None


def test_provider_api_key(
    provider_name: str, env_var: str, test_func: Callable[[], bool]
) -> Tuple[Optional[bool], str]:
    """Test a provider's API key."""
    api_key = os.environ.get(env_var)
    if not api_key:
        return None, "Not configured"
    if is_placeholder_key(api_key):
        return None, "Not configured"

    # Suppress verbose logging during tests
    import logging

    loggers_to_suppress = ["httpx", "google_genai", "openai", "anthropic"]
    old_levels = {}
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        old_levels[logger_name] = logger.level
        logger.setLevel(logging.ERROR)

    try:
        result = test_func()
        # Restore logging levels
        for logger_name, old_level in old_levels.items():
            logging.getLogger(logger_name).setLevel(old_level)
        if result:
            return True, "✓ Valid"
        else:
            return False, "✗ Invalid"
    except Exception as e:
        # Restore logging levels
        for logger_name, old_level in old_levels.items():
            logging.getLogger(logger_name).setLevel(old_level)
        error_msg = str(e)
        # Clean up common error messages
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, "✗ Invalid API key"
        elif "400" in error_msg or "bad request" in error_msg.lower():
            return False, "✗ Invalid request"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return False, "✗ Service unavailable"
        elif "not found" in error_msg.lower() or "missing" in error_msg.lower():
            return False, "✗ Missing key"
        # Truncate long error messages
        clean_msg = error_msg.replace("\n", " ").replace("\r", " ")
        if len(clean_msg) > 40:
            clean_msg = clean_msg[:37] + "..."
        return False, f"✗ {clean_msg}"


def test_openai() -> bool:
    """Test OpenAI API key."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # Simple test: list models (lightweight call)
    try:
        list(client.models.list(limit=1))
        return True
    except Exception:
        # Fallback: try a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}], max_tokens=1
        )
        return bool(response.choices)


def test_anthropic() -> bool:
    """Test Anthropic API key."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # Simple test: send a minimal message
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1,
        messages=[{"role": "user", "content": "test"}],
    )
    return bool(response.content)


def test_gemini() -> bool:
    """Test Google Gemini API key."""
    from google import genai

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    # Simple test: generate content
    response = client.models.generate_content(model="gemini-pro", contents="test")
    return bool(response.text)


def test_openrouter() -> bool:
    """Test OpenRouter API key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}], max_tokens=1
    )
    return bool(response.choices)


def test_fireworks() -> bool:
    """Test Fireworks API key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1",
    )
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3-8b-instruct",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1,
    )
    return bool(response.choices)


def test_groq() -> bool:
    """Test Groq API key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": "test"}], max_tokens=1
    )
    return bool(response.choices)


def test_deepseek() -> bool:
    """Test DeepSeek API key."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1"
    )
    response = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": "test"}], max_tokens=1
    )
    return bool(response.choices)


def test_xai() -> bool:
    """Test xAI API key."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-beta", messages=[{"role": "user", "content": "test"}], max_tokens=1
    )
    return bool(response.choices)


def test_huggingface() -> bool:
    """Test Hugging Face API key."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        provider="fireworks-ai", api_key=os.environ.get("HUGGING_FACE_API_KEY")
    )
    # Simple text generation test
    response = client.text_generation("test", max_new_tokens=1)
    return bool(response)
