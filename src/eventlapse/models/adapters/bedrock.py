from pathlib import Path
from typing import Dict, Any, List, Optional
from eventlapse.models.base import BaseVideoModel, ModelConfig, ModelResponse

class BedrockAdapter(BaseVideoModel):
    @property
    def supports_native_video(self) -> bool:
        return False

    @property
    def supports_multiple_images(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True

    @property
    def supports_thinking(self) -> bool:
        return False

    def query_native_video(self, video_path: Path, prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> ModelResponse:
        return ModelResponse(raw_response_text="", error="Amazon Bedrock adapter native video query stub called.")

    def query_frames(self, frame_paths: List[Path], prompt: str, response_schema: Optional[Dict[str, Any]] = None, **kwargs) -> ModelResponse:
        return ModelResponse(raw_response_text="", error="Amazon Bedrock adapter frame query stub called.")
