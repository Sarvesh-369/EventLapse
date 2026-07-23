import pytest
from eventlapse.models.load_model import load_model
from eventlapse.models.base import ModelConfig

def test_load_model_dispatch():
    cfg = ModelConfig(provider="google", model_name="gemini-2.0-flash")
    model = load_model("google", "gemini-2.0-flash", cfg)
    assert model.supports_native_video is True
    assert model.supports_structured_output is True
    assert model.config.provider == "google"

def test_vllm_model_dispatch():
    cfg = ModelConfig(provider="vllm", model_name="Qwen/Qwen2-VL-7B-Instruct")
    model = load_model("vllm", "Qwen/Qwen2-VL-7B-Instruct", cfg)
    assert model.supports_multiple_images is True
    assert model.supports_structured_output is True
    assert model.config.provider == "vllm"

def test_unsupported_provider():
    with pytest.raises(ValueError):
        load_model("unsupported_provider", "dummy-model")
