from .provider import ProviderAdapter


# Lazy imports to avoid requiring all provider SDKs
def _lazy_import_adapter(adapter_name: str):
    """Lazy import adapter to avoid loading all provider SDKs upfront"""
    if adapter_name == "anthropic":
        from .anthropic import AnthropicAdapter
        return AnthropicAdapter
    elif adapter_name == "openai":
        from .open_ai import OpenAIAdapter
        return OpenAIAdapter
    elif adapter_name == "deepseek":
        from .deepseek import DeepseekAdapter
        return DeepseekAdapter
    elif adapter_name == "gemini":
        from .gemini import GeminiAdapter
        return GeminiAdapter
    elif adapter_name == "huggingfacefireworks":
        from .hugging_face_fireworks import HuggingFaceFireworksAdapter
        return HuggingFaceFireworksAdapter
    elif adapter_name == "fireworks":
        from .fireworks import FireworksAdapter
        return FireworksAdapter
    elif adapter_name == "grok":
        from .grok import GrokAdapter
        return GrokAdapter
    elif adapter_name == "openrouter":
        from .openrouter import OpenRouterAdapter
        return OpenRouterAdapter
    elif adapter_name == "xai":
        from .xai import XAIAdapter
        return XAIAdapter
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")


def create_provider(config: str) -> ProviderAdapter:
    """
    Factory function to create a provider adapter based on config name.
    
    Args:
        config: Model configuration name from models.yml
        
    Returns:
        Initialized provider adapter
    """
    from arcagi3.utils import read_models_config
    
    model_config = read_models_config(config)
    provider_name = model_config.provider
    
    # Get the adapter class and instantiate it
    adapter_class = _lazy_import_adapter(provider_name)
    return adapter_class(config)


__all__ = [
    "ProviderAdapter",
    "create_provider",
]