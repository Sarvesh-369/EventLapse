from typing import Dict, Any, Optional

def configure_thinking_budget(thinking_level: Optional[str] = "high") -> Dict[str, Any]:
    return {
        "thinking_level": thinking_level,
        "thinking_enabled": thinking_level is not None
    }
