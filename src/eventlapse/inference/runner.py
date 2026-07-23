import time
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from eventlapse.models.base import BaseVideoModel, ModelResponse
from eventlapse.inference.prompts import get_prompt_for_condition
from eventlapse.inference.parser import parse_model_response
from eventlapse.interventions.frame_extraction import extract_frames_at_fps
from eventlapse.interventions.oracle_evidence import extract_oracle_frames
from eventlapse.utils.cost_calculator import calculate_api_cost

logger = logging.getLogger("eventlapse.inference.runner")

def get_native_video_num_frames(video_path: Path, fps: int = 30) -> int:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", str(video_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(res.stdout.strip())
        return int(round(dur * fps))
    except Exception:
        return 720 # Fallback 24s @ 30fps

def run_single_inference(
    model: BaseVideoModel,
    video_path: Path,
    question: str,
    input_mode: str = "native_video",
    prompt_condition: str = "structured_trace",
    response_schema: Optional[Dict[str, Any]] = None,
    trace_events: Optional[list] = None
) -> Dict[str, Any]:
    prompt = get_prompt_for_condition(prompt_condition, question)

    system_instruction = None
    if prompt_condition == "role_prompting":
        system_instruction = "You are an expert video analytics systems auditor specializing in fine-grained temporal event verification."

    thinking_mode = (prompt_condition == "thinking")

    num_frames = 0
    start_time = time.time()

    query_kwargs = {}
    if system_instruction:
        query_kwargs["system_instruction"] = system_instruction
    if thinking_mode:
        query_kwargs["thinking_mode"] = True

    if input_mode == "native_video":
        if not model.supports_native_video:
            raise ValueError(f"Model provider {model.config.provider} does not support native video input.")
        num_frames = get_native_video_num_frames(video_path)
        response = model.query_native_video(video_path, prompt, response_schema=response_schema, **query_kwargs)

    elif input_mode.startswith("frames_"):
        # Support frames_1fps, frames_2fps, frames_4fps, frames_8fps, frames_10fps, frames_16fps, etc.
        try:
            fps_str = input_mode.replace("frames_", "").replace("fps", "")
            target_fps = int(fps_str)
        except ValueError:
            target_fps = 2

        frames_dir = video_path.parent / f"frames_{input_mode}_{video_path.stem}"
        frame_paths = extract_frames_at_fps(video_path, target_fps=target_fps, output_dir=frames_dir)
        num_frames = len(frame_paths)
        response = model.query_frames(frame_paths, prompt, response_schema=response_schema, **query_kwargs)

    elif input_mode == "oracle_evidence":
        # Extract oracle evidence frames around ground truth event timestamps
        if trace_events is None:
            trace_json_path = video_path.parent.parent.parent / "traces" / video_path.parent.name / f"{video_path.stem}_trace.json"
            if not trace_json_path.exists():
                trace_json_path = video_path.parent.parent / "traces" / video_path.parent.name / f"{video_path.stem}_trace.json"

            if trace_json_path.exists():
                with open(trace_json_path, "r") as tf:
                    t_data = json.load(tf)
                    trace_events = t_data.get("events", [])
            else:
                trace_events = []

        oracle_dir = video_path.parent / f"oracle_{video_path.stem}"
        oracle_frame_paths = extract_oracle_frames(video_path, trace_events=trace_events, epsilon=0.1, output_dir=oracle_dir)
        num_frames = len(oracle_frame_paths)

        if len(oracle_frame_paths) > 0 and model.supports_multiple_images:
            response = model.query_frames(oracle_frame_paths, prompt, response_schema=response_schema, **query_kwargs)
        elif model.supports_native_video:
            num_frames = get_native_video_num_frames(video_path)
            oracle_prompt = f"[ORACLE EVIDENCE AVAILABLE: Key event windows extracted] {prompt}"
            response = model.query_native_video(video_path, oracle_prompt, response_schema=response_schema, **query_kwargs)
        else:
            raise ValueError(f"Oracle evidence execution failed: no extracted frames and model does not support native video.")

    else:
        raise ValueError(f"Unknown input mode: {input_mode}")

    pred_answer, trace_dict, is_valid = parse_model_response(response.raw_response_text, response.parsed_json)

    token_usage = response.token_usage or {}
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("candidate_tokens", token_usage.get("completion_tokens", 0))
    total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

    cost_usd = calculate_api_cost(
        model_name=model.config.model_name,
        prompt_tokens=prompt_tokens,
        candidate_tokens=completion_tokens,
        provider=model.config.provider
    )

    return {
        "raw_response": response.raw_response_text,
        "parsed_json": trace_dict,
        "predicted_answer": pred_answer,
        "is_valid": is_valid,
        "latency_sec": response.latency_sec,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "num_frames": num_frames,
        "error": response.error,
        "input_mode": input_mode,
        "prompt_condition": prompt_condition
    }
