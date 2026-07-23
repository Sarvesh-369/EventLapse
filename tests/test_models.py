import pytest
from eventlapse.models.load_model import load_model, parse_model_spec
from eventlapse.models.base import ModelConfig

def test_load_model_dispatch():
    cfg = ModelConfig(provider="google", model_name="gemini-2.0-flash")
    model = load_model("google", "gemini-2.0-flash", cfg)
    assert model.supports_native_video is True
    assert model.supports_structured_output is True
    assert model.config.provider == "google"

def test_propensity_bench_format():
    p1, m1 = parse_model_spec("gemini/gemini-3.1-pro-preview")
    assert p1 == "google"
    assert m1 == "gemini-3.1-pro-preview"

    p2, m2 = parse_model_spec("openai/gpt-4o")
    assert p2 == "openai"
    assert m2 == "gpt-4o"

    p3, m3 = parse_model_spec("anthropic/claude-3-5-sonnet-20241022")
    assert p3 == "anthropic"
    assert m3 == "claude-3-5-sonnet-20241022"

    m = load_model("gemini/gemini-3.1-pro-preview")
    assert m.config.provider == "google"
    assert m.config.model_name == "gemini-3.1-pro-preview"

def test_vllm_model_dispatch():
    cfg = ModelConfig(provider="vllm", model_name="Qwen/Qwen2-VL-7B-Instruct")
    model = load_model("vllm", "Qwen/Qwen2-VL-7B-Instruct", cfg)
    assert model.supports_multiple_images is True
    assert model.supports_structured_output is True
    assert model.config.provider == "vllm"

def test_unsupported_provider():
    with pytest.raises(ValueError):
        load_model("unsupported_provider", "dummy-model")
