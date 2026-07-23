from typing import Dict, Any

PROVIDER_CAPABILITIES = {
    "google": {
        "native_video": True,
        "multiple_images": True,
        "structured_output": True,
        "thinking": True
    },
    "openai": {
        "native_video": False,
        "multiple_images": True,
        "structured_output": True,
        "thinking": True
    },
    "anthropic": {
        "native_video": False,
        "multiple_images": True,
        "structured_output": True,
        "thinking": True
    },
    "bedrock": {
        "native_video": False,
        "multiple_images": True,
        "structured_output": True,
        "thinking": False
    },
    "fireworks": {
        "native_video": False,
        "multiple_images": True,
        "structured_output": True,
        "thinking": False
    }
}

def get_provider_capabilities(provider: str) -> Dict[str, bool]:
    return PROVIDER_CAPABILITIES.get(provider.lower(), {
        "native_video": False,
        "multiple_images": False,
        "structured_output": False,
        "thinking": False
    })
