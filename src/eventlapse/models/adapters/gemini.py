import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from eventlapse.models.base import BaseVideoModel, ModelConfig, ModelResponse
from eventlapse.utils.caching import compute_file_checksum

logger = logging.getLogger("eventlapse.models.gemini")

class GeminiAdapter(BaseVideoModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._file_cache: Dict[str, Any] = {} # checksum -> uploaded file object

        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set.")

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        except Exception as e:
            logger.error(f"Failed to initialize google-genai client: {e}")
            self.client = None

    @property
    def supports_native_video(self) -> bool:
        return True

    @property
    def supports_multiple_images(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True

    @property
    def supports_thinking(self) -> bool:
        return True

    def _upload_video_with_cache(self, video_path: Path) -> Any:
        checksum = compute_file_checksum(video_path)
        if checksum in self._file_cache:
            file_ref = self._file_cache[checksum]
            logger.info(f"Using cached Gemini file reference for {video_path.name}")
            return file_ref

        if not self.client:
            raise RuntimeError("Gemini client not initialized (missing API key or SDK)")

        logger.info(f"Uploading {video_path.name} to Gemini Files API...")
        file_ref = self.client.files.upload(file=str(video_path))

        # Poll until active
        while file_ref.state.name == "PROCESSING":
            logger.info("Waiting for video processing to complete on Gemini...")
            time.sleep(3)
            file_ref = self.client.files.get(name=file_ref.name)

        if file_ref.state.name == "FAILED":
            raise RuntimeError(f"Gemini video processing failed: {file_ref.error.message}")

        self._file_cache[checksum] = file_ref
        return file_ref

    def query_native_video(
        self,
        video_path: Path,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        if not self.client:
            return ModelResponse(
                raw_response_text="",
                error="Gemini API client unavailable. Set GEMINI_API_KEY environment variable."
            )

        start_time = time.time()
        max_retries = kwargs.get("max_retries", 3)
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                video_file = self._upload_video_with_cache(video_path)

                gen_config = {}
                if self.config.temperature is not None:
                    gen_config["temperature"] = self.config.temperature
                if response_schema:
                    gen_config["response_mime_type"] = "application/json"
                    # Note: pass schema if dict or pydantic
                    gen_config["response_schema"] = response_schema

                contents = [video_file, prompt]

                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=gen_config if gen_config else None
                )

                latency = round(time.time() - start_time, 3)
                raw_text = response.text or ""

                parsed_json = None
                if response_schema or raw_text.strip().startswith("{"):
                    try:
                        parsed_json = json.loads(raw_text)
                    except Exception:
                        pass

                usage_dict = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_dict = {
                        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "candidate_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
                    }

                return ModelResponse(
                    raw_response_text=raw_text,
                    parsed_json=parsed_json,
                    token_usage=usage_dict,
                    latency_sec=latency,
                    model_version=self.config.model_name,
                    finish_reason="STOP",
                    retry_count=attempt
                )

            except Exception as e:
                logger.warning(f"Gemini request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return ModelResponse(
                        raw_response_text="",
                        latency_sec=round(time.time() - start_time, 3),
                        retry_count=attempt,
                        error=str(e)
                    )
                time.sleep(backoff ** (attempt + 1))

        return ModelResponse(raw_response_text="", error="Max retries reached")

    def query_frames(
        self,
        frame_paths: List[Path],
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        if not self.client:
            return ModelResponse(raw_response_text="", error="Gemini API client unavailable.")

        from PIL import Image
        images = [Image.open(p) for p in frame_paths if p.exists()]
        contents = images + [prompt]

        start_time = time.time()
        try:
            gen_config = {}
            if response_schema:
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = response_schema

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=gen_config if gen_config else None
            )

            latency = round(time.time() - start_time, 3)
            raw_text = response.text or ""
            parsed_json = None
            if raw_text.strip().startswith("{"):
                try:
                    parsed_json = json.loads(raw_text)
                except Exception:
                    pass

            return ModelResponse(
                raw_response_text=raw_text,
                parsed_json=parsed_json,
                latency_sec=latency,
                model_version=self.config.model_name,
                finish_reason="STOP"
            )
        except Exception as e:
            return ModelResponse(raw_response_text="", latency_sec=round(time.time() - start_time, 3), error=str(e))
