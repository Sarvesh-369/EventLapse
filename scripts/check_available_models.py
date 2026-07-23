#!/usr/bin/env python3
import os
import sys
import click
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.models.capabilities import PROVIDER_CAPABILITIES, get_provider_capabilities

@click.command()
@click.option("--provider", default="google", help="Provider to check (google, openai, anthropic, bedrock, fireworks)")
@click.option("--smoke-test", is_flag=True, help="Perform minimal model call")
def main(provider: str, smoke_test: bool):
    """
    Model discovery and capability inspector script.
    Checks environment API keys, capabilities, and performs minimal smoke test.
    """
    click.echo(f"=== Model Discovery & Capability Check: {provider.upper()} ===")

    env_var_map = {
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "fireworks": "FIREWORKS_API_KEY"
    }

    key_var = env_var_map.get(provider.lower(), "")
    has_key = bool(os.getenv(key_var)) if key_var else False

    click.echo(f"API Key Environment Variable ({key_var}): {'SET' if has_key else 'NOT SET'}")

    caps = get_provider_capabilities(provider)
    click.echo("\nCapabilities:")
    for cap, val in caps.items():
        click.echo(f"  - {cap}: {val}")

    if provider.lower() == "google" and has_key:
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            click.echo("\nListing available Gemini models:")
            for m in client.models.list():
                if "gemini" in m.name.lower():
                    click.echo(f"  - {m.name}")
        except Exception as e:
            click.echo(f"\nFailed to list models via API: {e}")

    if smoke_test:
        click.echo(f"\nRunning minimal smoke test for provider {provider}...")
        from eventlapse.models.load_model import load_model
        from eventlapse.models.base import ModelConfig

        model_name_map = {
            "google": "gemini-3.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "fireworks": "accounts/fireworks/models/qwen2-vl-72b-instruct"
        }
        m_name = model_name_map.get(provider.lower(), "default")
        cfg = ModelConfig(provider=provider, model_name=m_name)
        model = load_model(provider, m_name, cfg)
        click.echo(f"Successfully instantiated model loader for {model.config.model_name}")

if __name__ == "__main__":
    main()
