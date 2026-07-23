import os
import time
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from eventlapse.models.base import BaseVideoModel, ModelConfig, ModelResponse

try:
    import litellm
except ImportError:
    litellm = None

class PropensityClientAdapter(BaseVideoModel):
    """
    Model adapter integrating PropensityBench (scaleapi/propensity-evaluation) gateway pattern.
    Uses LiteLLM / API proxy with rate limiting (RATE_PM, RATE_LIMIT) and key pooling support.
    Supports model format '<provider>/<original_model_name>' e.g. gemini/gemini-2.0-flash, openai/gpt-4o.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.rate_limit_enabled = os.environ.get("RATE_LIMIT", "false").lower() in ("true", "1")
        self.rate_pm = int(os.environ.get("RATE_PM", "60"))
        self.last_call_time = 0.0
        self.min_interval = 60.0 / max(1, self.rate_pm) if self.rate_limit_enabled else 0.0

        # Construct litellm / propensity model spec
        if "/" in self.config.model_name:
            self.full_model_spec = self.config.model_name
        else:
            p_map = {"google": "gemini", "openai": "openai", "anthropic": "anthropic", "vllm": "vllm"}
            p_prefix = p_map.get(self.config.provider.lower(), self.config.provider.lower())
            self.full_model_spec = f"{p_prefix}/{self.config.model_name}"

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

    def _enforce_rate_limit(self):
        if self.rate_limit_enabled and self.min_interval > 0:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()

    def query_native_video(
        self,
        video_path: Path,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        self._enforce_rate_limit()
        start_t = time.time()

        if litellm is None:
            # Fallback to direct HTTP / OpenAI compatible format if litellm is not installed
            return self._query_http_fallback(video_path=video_path, prompt=prompt)

        try:
            with open(video_path, "rb") as vf:
                b64_video = base64.b64encode(vf.read()).decode("utf-8")
            video_url = f"data:video/mp4;base64,{b64_video}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_url}}
                    ]
                }
            ]

            res = litellm.completion(
                model=self.full_model_spec,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens
            )

            latency = round(time.time() - start_t, 2)
            raw_text = res.choices[0].message.content or ""
            usage = getattr(res, "usage", {})
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            comp_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

            return ModelResponse(
                raw_response_text=raw_text,
                parsed_json=self._try_parse_json(raw_text),
                token_usage={"prompt_tokens": prompt_tokens, "candidate_tokens": comp_tokens, "total_tokens": total_tokens},
                latency_sec=latency,
                model_version=self.full_model_spec
            )
        except Exception as e:
            return ModelResponse(
                raw_response_text="",
                latency_sec=round(time.time() - start_t, 2),
                error=f"PropensityClient litellm query_native_video error: {str(e)}"
            )

    def query_frames(
        self,
        frame_paths: List[Path],
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        self._enforce_rate_limit()
        start_t = time.time()

        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            try:
                with open(fp, "rb") as img_f:
                    b64_img = base64.b64encode(img_f.read()).decode("utf-8")
                img_url = f"data:image/jpeg;base64,{b64_img}"
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            except Exception:
                pass

        messages = [{"role": "user", "content": content}]

        if litellm is None:
            return self._query_frames_http_fallback(messages=messages, start_t=start_t)

        try:
            res = litellm.completion(
                model=self.full_model_spec,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens
            )

            latency = round(time.time() - start_t, 2)
            raw_text = res.choices[0].message.content or ""
            usage = getattr(res, "usage", {})
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            comp_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

            return ModelResponse(
                raw_response_text=raw_text,
                parsed_json=self._try_parse_json(raw_text),
                token_usage={"prompt_tokens": prompt_tokens, "candidate_tokens": comp_tokens, "total_tokens": total_tokens},
                latency_sec=latency,
                model_version=self.full_model_spec
            )
        except Exception as e:
            return ModelResponse(
                raw_response_text="",
                latency_sec=round(time.time() - start_t, 2),
                error=f"PropensityClient litellm query_frames error: {str(e)}"
            )

    def _query_http_fallback(self, video_path: Path, prompt: str) -> ModelResponse:
        import requests
        start_t = time.time()
        gateway_url = os.environ.get("PROPENSITY_GATEWAY_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        endpoint = f"{gateway_url}/chat/completions"

        try:
            with open(video_path, "rb") as vf:
                b64_video = base64.b64encode(vf.read()).decode("utf-8")
            video_url = f"data:video/mp4;base64,{b64_video}"

            content = [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_url}}
            ]
            payload = {
                "model": self.full_model_spec,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens
            }

            resp = requests.post(endpoint, json=payload, timeout=120)
            latency = round(time.time() - start_t, 2)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return ModelResponse(
                    raw_response_text=text,
                    parsed_json=self._try_parse_json(text),
                    token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "candidate_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
                    latency_sec=latency,
                    model_version=self.full_model_spec
                )
            else:
                return ModelResponse(raw_response_text="", latency_sec=latency, error=f"Gateway returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            return ModelResponse(raw_response_text="", latency_sec=round(time.time() - start_t, 2), error=f"HTTP fallback query_native_video error: {str(e)}")

    def _query_frames_http_fallback(self, messages: List[Dict[str, Any]], start_t: float) -> ModelResponse:
        import requests
        gateway_url = os.environ.get("PROPENSITY_GATEWAY_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        endpoint = f"{gateway_url}/chat/completions"

        payload = {
            "model": self.full_model_spec,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=180)
            latency = round(time.time() - start_t, 2)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return ModelResponse(
                    raw_response_text=text,
                    parsed_json=self._try_parse_json(text),
                    token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "candidate_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
                    latency_sec=latency,
                    model_version=self.full_model_spec
                )
            else:
                return ModelResponse(raw_response_text="", latency_sec=latency, error=f"Gateway returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            return ModelResponse(raw_response_text="", latency_sec=round(time.time() - start_t, 2), error=f"HTTP fallback query_frames error: {str(e)}")

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
