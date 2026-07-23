import os
from typing import Optional, Tuple
from eventlapse.models.base import BaseVideoModel, ModelConfig
from eventlapse.models.adapters.gemini import GeminiAdapter
from eventlapse.models.adapters.openai import OpenAIAdapter
from eventlapse.models.adapters.anthropic import AnthropicAdapter
from eventlapse.models.adapters.bedrock import BedrockAdapter
from eventlapse.models.adapters.fireworks import FireworksAdapter
from eventlapse.models.adapters.vllm import VLLMAdapter

def parse_model_spec(provider_or_spec: str, model_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Parses provider and model_name from strings. Supports PropensityBench format:
    e.g. 'gemini/gemini-3.1-pro-preview', 'google/gemini-2.0-flash', 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20241022'
    """
    provider_alias_map = {
        "gemini": "google",
        "google": "google",
        "openai": "openai",
        "gpt": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "bedrock": "bedrock",
        "aws": "bedrock",
        "fireworks": "fireworks",
        "vllm": "vllm",
        "propensity": "vllm"
    }

    if provider_or_spec.lower() in provider_alias_map and model_name:
        resolved_provider = provider_alias_map[provider_or_spec.lower()]
        resolved_model = model_name
    elif "/" in provider_or_spec and (model_name is None or model_name == provider_or_spec):
        parts = provider_or_spec.split("/", 1)
        raw_p = parts[0]
        resolved_provider = provider_alias_map.get(raw_p.lower(), raw_p.lower())
        resolved_model = parts[1]
    elif model_name and "/" in model_name and provider_or_spec.lower() not in ["vllm", "google", "openai", "anthropic", "bedrock", "fireworks"]:
        parts = model_name.split("/", 1)
        raw_p = parts[0]
        resolved_provider = provider_alias_map.get(raw_p.lower(), raw_p.lower())
        resolved_model = parts[1]
    else:
        raw_p = provider_or_spec
        resolved_provider = provider_alias_map.get(raw_p.lower(), raw_p.lower())
        resolved_model = model_name if model_name else provider_or_spec

    return resolved_provider, resolved_model

def load_model(provider: str, model_name: Optional[str] = None, config: Optional[ModelConfig] = None) -> BaseVideoModel:
    """
    Loads and returns the requested model adapter. Supports both (provider, model_name) and single spec string '<provider>/<model_name>'.
    """
    resolved_provider, resolved_model_name = parse_model_spec(provider, model_name)

    if config is None:
        config = ModelConfig(provider=resolved_provider, model_name=resolved_model_name)
    else:
        config.provider = resolved_provider
        config.model_name = resolved_model_name

    provider_lower = resolved_provider.lower()

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
    elif provider_lower == "vllm":
        return VLLMAdapter(config)
    else:
        raise ValueError(f"Unsupported model provider: '{resolved_provider}'. Supported providers: google, openai, anthropic, bedrock, fireworks, vllm.")
