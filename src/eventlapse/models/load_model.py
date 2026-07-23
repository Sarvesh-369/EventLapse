from typing import Optional
from eventlapse.models.base import BaseVideoModel, ModelConfig
from eventlapse.models.adapters.gemini import GeminiAdapter
from eventlapse.models.adapters.openai import OpenAIAdapter
from eventlapse.models.adapters.anthropic import AnthropicAdapter
from eventlapse.models.adapters.bedrock import BedrockAdapter
from eventlapse.models.adapters.fireworks import FireworksAdapter

def load_model(provider: str, model_name: str, config: Optional[ModelConfig] = None) -> BaseVideoModel:
    if config is None:
        config = ModelConfig(provider=provider, model_name=model_name)
    else:
        config.provider = provider
        config.model_name = model_name

    provider_lower = provider.lower()

    if provider_lower == "google":
        return GeminiAdapter(config)
    elif provider_lower == "openai":
        return OpenAIAdapter(config)
    elif provider_lower == "anthropic":
        return AnthropicAdapter(config)
    elif provider_lower == "bedrock":
        return BedrockAdapter(config)
    elif provider_lower == "fireworks":
        return FireworksAdapter(config)
    else:
        raise ValueError(f"Unsupported model provider: '{provider}'. Supported providers: google, openai, anthropic, bedrock, fireworks.")
