#!/usr/bin/env python3
import sys
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.models.load_model import load_model
from eventlapse.models.base import ModelConfig
from eventlapse.utils.paths import get_data_dir

@click.command()
@click.option("--provider", default="google")
@click.option("--model-name", default="gemini-3.5-flash")
@click.option("--video-path", default=None, help="Path to sample MP4 video")
def main(provider: str, model_name: str, video_path: str):
    """
    Minimal smoke test for model API adapter.
    """
    click.echo(f"Testing model adapter: {provider} / {model_name}")
    config = ModelConfig(provider=provider, model_name=model_name)
    model = load_model(provider, model_name, config)

    click.echo(f"Loaded model instance: {model.get_model_metadata()}")

    if video_path and Path(video_path).exists():
        click.echo(f"Querying model with video {video_path}...")
        resp = model.query_native_video(Path(video_path), "How many bounces occurred in this video?")
        click.echo(f"Response Latency: {resp.latency_sec}s")
        click.echo(f"Raw Response: {resp.raw_response_text[:200]}")
        click.echo(f"Error: {resp.error}")
    else:
        click.echo("No valid video provided. Skipping live query.")

if __name__ == "__main__":
    main()
