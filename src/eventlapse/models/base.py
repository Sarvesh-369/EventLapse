import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

@dataclass
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    max_output_tokens: int = 2048
    thinking_level: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    raw_response_text: str
    parsed_json: Optional[Dict[str, Any]] = None
    token_usage: Dict[str, int] = field(default_factory=dict) # prompt_tokens, candidate_tokens, total_tokens
    latency_sec: float = 0.0
    model_version: str = ""
    finish_reason: str = ""
    retry_count: int = 0
    error: Optional[str] = None

class BaseVideoModel(abc.ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    @abc.abstractmethod
    def supports_native_video(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def supports_multiple_images(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def supports_structured_output(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def supports_thinking(self) -> bool:
        pass

    @abc.abstractmethod
    def query_native_video(
        self,
        video_path: Path,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        pass

    @abc.abstractmethod
    def query_frames(
        self,
        frame_paths: List[Path],
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        pass

    def start_conversation(self, **kwargs) -> Any:
        raise NotImplementedError("Conversational mode not implemented for this adapter")

    def continue_conversation(self, conversation_handle: Any, prompt: str, **kwargs) -> ModelResponse:
        raise NotImplementedError("Conversational mode not implemented for this adapter")

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "supports_native_video": self.supports_native_video,
            "supports_multiple_images": self.supports_multiple_images,
            "supports_structured_output": self.supports_structured_output,
            "supports_thinking": self.supports_thinking
        }
