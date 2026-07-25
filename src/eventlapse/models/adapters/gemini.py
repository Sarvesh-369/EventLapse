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
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        except Exception as e:
            logger.error(f"Failed to initialize google-genai client: {e}")
            self.client = None
            self.types = None

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

    def _build_gen_config(
        self,
        system_instruction: Optional[str] = None,
        thinking_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        if not self.types:
            config_dict = {}
            if self.config.temperature is not None:
                config_dict["temperature"] = self.config.temperature
            if system_instruction:
                config_dict["system_instruction"] = system_instruction
            if response_schema:
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = response_schema
            return config_dict if config_dict else None

        kwargs = {}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if thinking_mode or "thinking" in self.config.model_name.lower():
            try:
                kwargs["thinking_config"] = self.types.ThinkingConfig(include_thoughts=True)
            except Exception:
                pass

        if response_schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema

        return self.types.GenerateContentConfig(**kwargs) if kwargs else None

    def query_native_video(
        self,
        video_path: Path,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        thinking_mode: bool = False,
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
        fps = kwargs.get("fps", None)

        for attempt in range(max_retries):
            try:
                gen_config = self._build_gen_config(
                    system_instruction=system_instruction,
                    thinking_mode=thinking_mode,
                    response_schema=response_schema
                )

                # For video files under 20MB, use inline_data with VideoMetadata
                if video_path.exists() and video_path.stat().st_size < 20 * 1024 * 1024 and self.types:
                    with open(video_path, "rb") as vf:
                        v_bytes = vf.read()

                    part_kwargs = {
                        "inline_data": self.types.Blob(data=v_bytes, mime_type="video/mp4")
                    }
                    if fps:
                        part_kwargs["video_metadata"] = self.types.VideoMetadata(fps=float(fps))

                    video_part = self.types.Part(**part_kwargs)
                    contents = [video_part, prompt]
                else:
                    video_file = self._upload_video_with_cache(video_path)
                    if fps and self.types:
                        try:
                            video_part = self.types.Part.from_uri(
                                file_uri=video_file.uri,
                                mime_type="video/mp4",
                                video_metadata=self.types.VideoMetadata(fps=float(fps))
                            )
                            contents = [video_part, prompt]
                        except Exception:
                            contents = [video_file, prompt]
                    else:
                        contents = [video_file, prompt]

                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=gen_config
                )

                latency = round(time.time() - start_time, 3)
                raw_text = response.text or ""

                usage_meta = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
                candidates_tokens = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
                total_tokens = getattr(usage_meta, "total_token_count", 0) if usage_meta else 0

                return ModelResponse(
                    raw_response_text=raw_text,
                    parsed_json=self._try_parse_json(raw_text),
                    token_usage={
                        "prompt_tokens": prompt_tokens,
                        "candidate_tokens": candidates_tokens,
                        "total_tokens": total_tokens
                    },
                    latency_sec=latency,
                    model_version=self.config.model_name
                )
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return ModelResponse(
                        raw_response_text="",
                        latency_sec=round(time.time() - start_time, 3),
                        error=str(e)
                    )
                time.sleep(backoff)
                backoff *= 2.0

        return ModelResponse(raw_response_text="", error="Gemini query failed after max retries")

    def query_frames(
        self,
        frame_paths: List[Path],
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        thinking_mode: bool = False,
        **kwargs
    ) -> ModelResponse:
        if not self.client:
            return ModelResponse(
                raw_response_text="",
                error="Gemini API client unavailable. Set GEMINI_API_KEY environment variable."
            )

        start_time = time.time()
        try:
            from PIL import Image
            contents = [prompt]
            for fp in frame_paths:
                try:
                    img = Image.open(fp)
                    contents.append(img)
                except Exception as ie:
                    logger.warning(f"Failed to load frame image {fp}: {ie}")

            gen_config = self._build_gen_config(
                system_instruction=system_instruction,
                thinking_mode=thinking_mode,
                response_schema=response_schema
            )

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=gen_config
            )

            latency = round(time.time() - start_time, 3)
            raw_text = response.text or ""

            usage_meta = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
            candidates_tokens = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
            total_tokens = getattr(usage_meta, "total_token_count", 0) if usage_meta else 0

            return ModelResponse(
                raw_response_text=raw_text,
                parsed_json=self._try_parse_json(raw_text),
                token_usage={
                    "prompt_tokens": prompt_tokens,
                    "candidate_tokens": candidates_tokens,
                    "total_tokens": total_tokens
                },
                latency_sec=latency,
                model_version=self.config.model_name
            )
        except Exception as e:
            return ModelResponse(
                raw_response_text="",
                latency_sec=round(time.time() - start_time, 3),
                error=f"Gemini query_frames error: {str(e)}"
            )

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return json.loads(text.strip())
        except Exception:
            return None
