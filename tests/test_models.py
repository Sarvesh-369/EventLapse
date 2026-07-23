import pytest
from eventlapse.models.load_model import load_model
from eventlapse.models.base import ModelConfig

def test_load_model_dispatch():
    cfg = ModelConfig(provider="google", model_name="gemini-3.5-flash")
    model = load_model("google", "gemini-3.5-flash", cfg)
    assert model.supports_native_video is True
    assert model.supports_structured_output is True
    assert model.config.provider == "google"

def test_unsupported_provider():
    with pytest.raises(ValueError):
        load_model("unsupported_provider", "dummy-model")
