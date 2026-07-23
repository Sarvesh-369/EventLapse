#!/usr/bin/env python3
import sys
import json
import time
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_data_dir, get_outputs_dir
from eventlapse.utils.logging import logger
from eventlapse.models.load_model import load_model
from eventlapse.models.base import ModelConfig
from eventlapse.inference.runner import run_single_inference
from eventlapse.evaluation.exact_match import compute_exact_match

@click.command()
@click.option("--provider", default="google", help="Model provider: google, openai, anthropic, bedrock, fireworks, vllm")
@click.option("--model-name", default="gemini-2.0-flash", help="Model identifier (e.g. gemini-2.0-flash, gpt-4o, Qwen/Qwen2-VL-7B-Instruct)")
@click.option("--input-mode", default="native_video", help="Input mode: native_video, frames_1fps, frames_2fps, frames_4fps, frames_8fps, frames_16fps, oracle_evidence")
@click.option("--prompt-condition", default="structured_trace", help="Prompt condition: direct, structured_trace, thinking")
@click.option("--task", default="all", help="Target task or 'all'")
@click.option("--vllm-url", default=None, help="Optional vLLM server base URL override (e.g. http://localhost:8000/v1)")
@click.option("--resume/--overwrite", default=True, help="Resume existing experiment run")
@click.option("--dry-run", is_flag=True, help="Dry run without executing API calls")
def main(
    provider: str,
    model_name: str,
    input_mode: str,
    prompt_condition: str,
    task: str,
    vllm_url: str,
    resume: bool,
    dry_run: bool
):
    """
    Executes N x F parametric matrix evaluation sweep across models, tracking input/output tokens,
    API cost ($USD), latency (sec), supplied frame counts, and MORSE exact match results.
    """
    data_dir = get_data_dir()
    manifest_path = data_dir / "manifest.jsonl"

    if not manifest_path.exists():
        # Fallback to sample_data manifest if data/ manifest does not exist
        sample_manifest = Path("sample_data/manifest.jsonl")
        if sample_manifest.exists():
            manifest_path = sample_manifest
        else:
            logger.error(f"Manifest path not found at {manifest_path} or sample_data/manifest.jsonl. Run generate_dataset.py first.")
            sys.exit(1)

    outputs_dir = get_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    clean_model_id = model_name.replace("/", "_").replace(":", "_")
    run_id = f"matrix_{provider}_{clean_model_id}_{input_mode}_{prompt_condition}"
    output_file = outputs_dir / f"results_{run_id}.jsonl"

    existing_sample_ids = set()
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_sample_ids.add(r.get("sample_id"))
        logger.info(f"Resuming matrix sweep {run_id}. Found {len(existing_sample_ids)} already completed samples.")

    # Load dataset items matching task filter
    samples = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if task == "all" or item.get("task_name") == task:
                    samples.append(item)

    logger.info(f"Loaded {len(samples)} samples for task target '{task}'.")

    if dry_run:
        logger.info("[DRY RUN MODE] Simulating N x F matrix sweep execution.")
        for item in samples[:5]:
            logger.info(f"Would process: {item['sample_id']} ({item['task_name']}) | Mode: {input_mode} | Condition: {prompt_condition}")
        return

    if vllm_url and provider == "vllm":
        import os
        os.environ["VLLM_BASE_URL"] = vllm_url

    config = ModelConfig(provider=provider, model_name=model_name)
    model = load_model(provider, model_name, config)

    executed_calls = 0

    with open(output_file, "a" if resume else "w") as out_f:
        for item in samples:
            sid = item["sample_id"]
            if resume and sid in existing_sample_ids:
                continue

            video_path = Path(item["video_path"])
            logger.info(f"Running matrix evaluation on {sid} ({item['task_name']})...")

            res = run_single_inference(
                model=model,
                video_path=video_path,
                question=item["question"],
                input_mode=input_mode,
                prompt_condition=prompt_condition
            )
            executed_calls += 1

            exact_match = compute_exact_match(res["predicted_answer"], item["exact_answer"])

            row = {
                "run_id": run_id,
                "sample_id": sid,
                "timestamp": time.time(),
                "task": item["task_name"],
                "control_parameter": item["control_parameter_name"],
                "control_value": item["control_parameter_value"],
                "seed": item["seed"],
                "provider": provider,
                "requested_model": model_name,
                "resolved_model": model_name,
                "input_mode": input_mode,
                "prompt_condition": prompt_condition,
                "num_frames": res["num_frames"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["total_tokens"],
                "estimated_cost_usd": res["estimated_cost_usd"],
                "latency_sec": res["latency_sec"],
                "question": item["question"],
                "ground_truth_answer": item["exact_answer"],
                "raw_model_response": res["raw_response"],
                "parsed_response": res["parsed_json"],
                "predicted_answer": res["predicted_answer"],
                "parser_validity": res["is_valid"],
                "exact_match_result": exact_match,
                "token_usage": res["token_usage"],
                "error_message": res["error"]
            }

            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    logger.info(f"Matrix sweep finished. Executed {executed_calls} model calls. Output written to {output_file}")

if __name__ == "__main__":
    main()
