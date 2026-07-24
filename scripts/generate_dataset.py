#!/usr/bin/env python3
import sys
import json
import click
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_data_dir, ensure_directories
from eventlapse.utils.logging import logger
from eventlapse.utils.caching import compute_file_checksum

from eventlapse.generation.bounce_ball import BounceBallGenerator
from eventlapse.generation.blinking import BlinkingGenerator
from eventlapse.generation.state_machine import StateMachineGenerator

# Exact 2D N x F matrix parameter grids:
COUNT_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
FREQUENCY_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

TASK_GENERATOR_MAP = {
    "bounce_ball": BounceBallGenerator(),
    "blinking": BlinkingGenerator(),
    "state_machine": StateMachineGenerator(),
}

@click.command()
@click.option("--num-seeds", default=20, help="Number of seeds to generate per (N, F) configuration (default 20)")
@click.option("--seed-start", default=0, help="Starting seed index")
@click.option("--seed-end", default=None, type=int, help="Ending seed index (exclusive)")
@click.option("--tasks", default="all", help="Comma-separated task names or 'all'")
@click.option("--output-dir", default=None, help="Output dataset directory")
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip re-rendering existing video files")
def main(num_seeds: int, seed_start: int, seed_end: int, tasks: str, output_dir: str, skip_existing: bool):
    """
    Generate synthetic task videos across 2D N x F matrix grid (Count N x Frequency F), executable traces JSON, ground-truth metadata, and questions.
    For N=0 (zero events), frequency is invariant, so only F=1.0 Hz is sampled.
    """
    ensure_directories()
    data_dir = Path(output_dir) if output_dir else get_data_dir()

    if seed_end is None:
        seed_end = seed_start + num_seeds

    seeds = list(range(seed_start, seed_end))
    selected_tasks = list(TASK_GENERATOR_MAP.keys()) if tasks == "all" else [t.strip() for t in tasks.split(",")]

    manifest_entries = []

    for task_key in selected_tasks:
        if task_key not in TASK_GENERATOR_MAP:
            logger.warning(f"Skipping unknown task: {task_key}")
            continue

        generator = TASK_GENERATOR_MAP[task_key]
        logger.info(f"Processing task '{task_key}' across 2D N x F grid for seeds {seeds}...")

        for n_val in COUNT_GRID:
            # For N=0 (no events), frequency is invariant. Sample only F=1.0 Hz to prevent duplicate generation.
            active_freqs = [1.0] if n_val == 0 else FREQUENCY_GRID

            for f_val in active_freqs:
                for s in seeds:
                    sample_id_f = f"{task_key}_N{n_val}_F{f_val}_seed{s}"
                    target_video_f = data_dir / "videos" / task_key / f"{sample_id_f}.mp4"

                    sample_id_legacy = f"{task_key}_N{n_val}_seed{s}"
                    target_video_legacy = data_dir / "videos" / task_key / f"{sample_id_legacy}.mp4"

                    existing_video = None
                    active_sample_id = sample_id_f

                    if skip_existing:
                        if target_video_f.exists() and target_video_f.stat().st_size > 0:
                            existing_video = target_video_f
                            active_sample_id = sample_id_f
                        elif f_val == 1.0 and target_video_legacy.exists() and target_video_legacy.stat().st_size > 0:
                            existing_video = target_video_legacy
                            active_sample_id = sample_id_legacy

                    if existing_video:
                        logger.info(f"Skipping re-render for existing video: {existing_video.name}")
                        gt_json_path = data_dir / "gt" / task_key / f"{active_sample_id}_gt.json"

                        question = "How many events occurred in the video?"
                        exact_answer = str(n_val)

                        if gt_json_path.exists():
                            try:
                                with open(gt_json_path, "r") as gf:
                                    gt_d = json.load(gf)
                                    question = gt_d.get("question", question)
                                    exact_answer = str(gt_d.get("exact_answer", exact_answer))
                            except Exception:
                                pass

                        checksum = compute_file_checksum(existing_video)
                        rel_video_path = f"videos/{task_key}/{existing_video.name}"

                        manifest_entries.append({
                            "sample_id": active_sample_id,
                            "task_name": task_key,
                            "control_parameter_name": "N",
                            "control_parameter_value": n_val,
                            "frequency_hz": f_val,
                            "seed": s,
                            "video_path": rel_video_path,
                            "question": question,
                            "exact_answer": exact_answer,
                            "duration": 24.0,
                            "checksum": checksum
                        })
                    else:
                        sample = generator.generate_sample(
                            control_value=n_val,
                            frequency=f_val,
                            seed=s,
                            output_dir=data_dir
                        )
                        logger.info(f"Generated sample {sample.sample_id} -> {sample.video_path.name}")

                        rel_video_path = f"videos/{sample.task_name}/{sample.video_path.name}"

                        manifest_entries.append({
                            "sample_id": sample.sample_id,
                            "task_name": sample.task_name,
                            "control_parameter_name": sample.control_parameter_name,
                            "control_parameter_value": sample.control_parameter_value,
                            "frequency_hz": f_val,
                            "seed": sample.seed,
                            "video_path": rel_video_path,
                            "question": sample.question,
                            "exact_answer": sample.exact_answer,
                            "duration": sample.duration,
                            "checksum": sample.checksum
                        })

    manifest_path = data_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Dataset generation complete ({len(manifest_entries)} samples in manifest). Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
