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
@click.option("--provider", default="google", help="Model provider")
@click.option("--model-name", default="gemini-2.0-flash", help="Model identifier")
@click.option("--input-mode", default="native_video", help="Input mode: native_video, frames_1fps, frames_2fps, frames_4fps, frames_8fps, frames_10fps, frames_16fps, oracle_evidence")
@click.option("--prompt-condition", default="structured_trace", help="Prompt condition: direct, structured_trace, multi_turn_verification, thinking")
@click.option("--experiment-group", default="group_1", help="Experiment group identifier")
@click.option("--resume/--overwrite", default=True, help="Resume existing experiment run without rerunning completed calls")
@click.option("--max-calls", default=None, type=int, help="Maximum API calls to execute")
@click.option("--concurrency", default=1, type=int, help="Concurrency level")
@click.option("--dry-run", is_flag=True, help="Dry run without calling model API")
@click.option("--seed-start", default=0, help="Filter start seed")
@click.option("--seed-end", default=29, help="Filter end seed")
def main(
    provider: str,
    model_name: str,
    input_mode: str,
    prompt_condition: str,
    experiment_group: str,
    resume: bool,
    max_calls: int,
    concurrency: int,
    dry_run: bool,
    seed_start: int,
    seed_end: int
):
    """
    Main experiment runner executing models on generated dataset samples and storing per-request JSONL outputs.
    """
    data_dir = get_data_dir()
    manifest_path = data_dir / "manifest.jsonl"

    if not manifest_path.exists():
        logger.error(f"Manifest path not found: {manifest_path}. Run generate_dataset.py first.")
        sys.exit(1)

    outputs_dir = get_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{provider}_{model_name}_{input_mode}_{prompt_condition}"
    output_file = outputs_dir / f"results_{run_id}.jsonl"

    existing_sample_ids = set()
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_sample_ids.add(r.get("sample_id"))
        logger.info(f"Resuming experiment run {run_id}. Found {len(existing_sample_ids)} already completed samples.")

    # Load dataset items
    samples = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if seed_start <= item.get("seed", 0) <= seed_end:
                    samples.append(item)

    logger.info(f"Loaded {len(samples)} candidate samples matching seed range [{seed_start}, {seed_end}].")

    if dry_run:
        logger.info("[DRY RUN MODE] Simulating execution without calling Model APIs.")
        for item in samples[:5]:
            logger.info(f"Would process sample: {item['sample_id']} ({item['task_name']}) | Mode: {input_mode} | Condition: {prompt_condition}")
        return

    config = ModelConfig(provider=provider, model_name=model_name)
    model = load_model(provider, model_name, config)

    executed_calls = 0

    with open(output_file, "a" if resume else "w") as out_f:
        for item in samples:
            sid = item["sample_id"]
            if resume and sid in existing_sample_ids:
                continue

            if max_calls is not None and executed_calls >= max_calls:
                logger.info(f"Reached max_calls limit of {max_calls}. Stopping experiment execution.")
                break

            video_path = Path(item["video_path"])
            logger.info(f"Running inference on {sid} ({item['task_name']})...")

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

    logger.info(f"Experiment execution finished. Executed {executed_calls} model calls. Results written to {output_file}")

if __name__ == "__main__":
    main()
