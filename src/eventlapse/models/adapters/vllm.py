import os
import time
import base64
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from eventlapse.models.base import BaseVideoModel, ModelConfig, ModelResponse

class VLLMAdapter(BaseVideoModel):
    """
    Adapter for models hosted using vLLM's OpenAI-compatible API endpoint (e.g. vLLM vLM server).
    Supports open-source VLMs like Qwen2-VL, LLaVA-NeXT-Video, InternVL2, etc.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        # vLLM local server does not require API authentication.
        # If you deploy behind a proxy with auth, set VLLM_API_KEY in your environment.
        self.api_key = os.environ.get("VLLM_API_KEY", None)

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
        return False

    def query_native_video(
        self,
        video_path: Path,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Sends native video or base64 video payload to vLLM OpenAI-compatible endpoint.
        """
        start_t = time.time()
        endpoint = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with open(video_path, "rb") as vf:
                b64_video = base64.b64encode(vf.read()).decode("utf-8")
            video_url = f"data:video/mp4;base64,{b64_video}"

            content = [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_url}}
            ]

            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens
            }

            resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            latency = round(time.time() - start_t, 2)

            if resp.status_code == 200:
                res_json = resp.json()
                text = res_json["choices"][0]["message"]["content"]
                usage = res_json.get("usage", {})
                return ModelResponse(
                    raw_response_text=text,
                    parsed_json=self._try_parse_json(text),
                    token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "candidate_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
                    latency_sec=latency,
                    model_version=self.config.model_name
                )
            else:
                # If native video_url endpoint is not supported by the vLLM model server, return error
                return ModelResponse(
                    raw_response_text="",
                    latency_sec=latency,
                    error=f"vLLM server returned HTTP {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            return ModelResponse(
                raw_response_text="",
                latency_sec=round(time.time() - start_t, 2),
                error=f"vLLM query_native_video error: {str(e)}"
            )

    def query_frames(
        self,
        frame_paths: List[Path],
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Sends frame sequence as base64 images to vLLM OpenAI-compatible endpoint.
        """
        start_t = time.time()
        endpoint = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"


        for fp in frame_paths:
            try:
                with open(fp, "rb") as img_f:
                    b64_img = base64.b64encode(img_f.read()).decode("utf-8")
                img_url = f"data:image/jpeg;base64,{b64_img}"
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            except Exception as e:
                pass

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            latency = round(time.time() - start_t, 2)

            if resp.status_code == 200:
                res_json = resp.json()
                text = res_json["choices"][0]["message"]["content"]
                usage = res_json.get("usage", {})
                return ModelResponse(
                    raw_response_text=text,
                    parsed_json=self._try_parse_json(text),
                    token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "candidate_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
                    latency_sec=latency,
                    model_version=self.config.model_name
                )
            else:
                return ModelResponse(
                    raw_response_text="",
                    latency_sec=latency,
                    error=f"vLLM server returned HTTP {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            return ModelResponse(
                raw_response_text="",
                latency_sec=round(time.time() - start_t, 2),
                error=f"vLLM query_frames error: {str(e)}"
            )

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            # Look for ```json ... ``` blocks
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            return json.loads(text.strip())
        except Exception:
            return None
