import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from eventlapse.models.base import BaseVideoModel, ModelResponse
from eventlapse.inference.prompts import get_prompt_for_condition
from eventlapse.inference.parser import parse_model_response
from eventlapse.interventions.frame_extraction import extract_frames_at_fps

logger = logging.getLogger("eventlapse.inference.runner")

def run_single_inference(
    model: BaseVideoModel,
    video_path: Path,
    question: str,
    input_mode: str = "native_video",
    prompt_condition: str = "structured_trace",
    response_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    prompt = get_prompt_for_condition(prompt_condition, question)

    start_time = time.time()

    if input_mode == "native_video":
        if not model.supports_native_video:
            raise ValueError(f"Model provider {model.config.provider} does not support native video input.")
        response = model.query_native_video(video_path, prompt, response_schema=response_schema)
    elif input_mode in ["frames_2fps", "frames_10fps"]:
        target_fps = 2 if input_mode == "frames_2fps" else 10
        frames_dir = video_path.parent / f"frames_{input_mode}_{video_path.stem}"
        frame_paths = extract_frames_at_fps(video_path, target_fps=target_fps, output_dir=frames_dir)
        response = model.query_frames(frame_paths, prompt, response_schema=response_schema)
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")

    pred_answer, trace_dict, is_valid = parse_model_response(response.raw_response_text, response.parsed_json)

    return {
        "raw_response": response.raw_response_text,
        "parsed_json": trace_dict,
        "predicted_answer": pred_answer,
        "is_valid": is_valid,
        "latency_sec": response.latency_sec,
        "token_usage": response.token_usage,
        "error": response.error,
        "input_mode": input_mode,
        "prompt_condition": prompt_condition
    }
