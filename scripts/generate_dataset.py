#!/usr/bin/env python3
import sys
import json
import click
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_data_dir, ensure_directories
from eventlapse.utils.logging import logger

from eventlapse.generation.bounce_ball import BounceBallGenerator
from eventlapse.generation.blinking import BlinkingGenerator
from eventlapse.generation.state_machine import StateMachineGenerator

# Exact count grid matching Qwen3-VL evaluation matrix: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
COUNT_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]

TASK_GENERATOR_MAP = {
    "bounce_ball": (BounceBallGenerator(), COUNT_GRID),
    "blinking": (BlinkingGenerator(), COUNT_GRID),
    "state_machine": (StateMachineGenerator(), COUNT_GRID),
}

@click.command()
@click.option("--num-seeds", default=20, help="Number of seeds to generate per control value (default 20)")
@click.option("--seed-start", default=0, help="Starting seed index")
@click.option("--seed-end", default=None, type=int, help="Ending seed index (exclusive)")
@click.option("--tasks", default="all", help="Comma-separated task names or 'all'")
@click.option("--output-dir", default=None, help="Output dataset directory")
def main(num_seeds: int, seed_start: int, seed_end: int, tasks: str, output_dir: str):
    """
    Generate synthetic task videos, executable traces JSON, ground-truth metadata, and questions.
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

        generator, control_values = TASK_GENERATOR_MAP[task_key]
        logger.info(f"Generating task '{task_key}' across counts {control_values} for seeds {seeds}...")

        for val in control_values:
            for s in seeds:
                sample = generator.generate_sample(
                    control_value=val,
                    seed=s,
                    output_dir=data_dir
                )
                logger.info(f"Generated sample {sample.sample_id} -> {sample.video_path.name}")

                manifest_entries.append({
                    "sample_id": sample.sample_id,
                    "task_name": sample.task_name,
                    "control_parameter_name": sample.control_parameter_name,
                    "control_parameter_value": sample.control_parameter_value,
                    "seed": sample.seed,
                    "video_path": str(sample.video_path),
                    "question": sample.question,
                    "exact_answer": sample.exact_answer,
                    "duration": sample.duration,
                    "checksum": sample.checksum
                })

    manifest_path = data_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Dataset generation complete. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
