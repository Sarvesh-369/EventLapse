#!/usr/bin/env python3
import sys
import json
import click
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_data_dir
from eventlapse.utils.logging import logger

@click.command()
@click.option("--data-dir", default=None, help="Dataset directory to validate")
def main(data_dir: str):
    """
    Validates dataset integrity and rejects malformed samples.
    """
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    manifest_path = base_dir / "manifest.jsonl"

    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        sys.exit(1)

    seen_sample_ids = set()
    errors = []
    sample_count = 0

    with open(manifest_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample_count += 1
            item = json.loads(line)

            sid = item.get("sample_id")
            if sid in seen_sample_ids:
                errors.append(f"Line {line_num}: Duplicate sample ID '{sid}'")
            seen_sample_ids.add(sid)

            video_path = Path(item.get("video_path", ""))
            if not video_path.exists():
                errors.append(f"Sample '{sid}': Missing video file {video_path}")
            elif video_path.stat().st_size == 0:
                errors.append(f"Sample '{sid}': Empty video file {video_path}")

            question_file = base_dir / "questions" / item.get("task_name", "") / f"{sid}.txt"
            if not question_file.exists():
                errors.append(f"Sample '{sid}': Missing question file {question_file}")

            trace_file = base_dir / "traces" / item.get("task_name", "") / f"{sid}_trace.json"
            if not trace_file.exists():
                errors.append(f"Sample '{sid}': Missing trace file {trace_file}")

            cot_file = base_dir / "reasoning_traces" / item.get("task_name", "") / f"{sid}.txt"
            if not cot_file.exists():
                errors.append(f"Sample '{sid}': Missing CoT reasoning file {cot_file}")

            gt_file = base_dir / "gt" / item.get("task_name", "") / f"{sid}_gt.json"
            if not gt_file.exists():
                errors.append(f"Sample '{sid}': Missing ground-truth file {gt_file}")

    logger.info(f"Validated {sample_count} dataset samples.")
    if errors:
        logger.error(f"Validation FAILED with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)
    else:
        logger.info("Dataset Validation PASSED cleanly with 0 errors!")

if __name__ == "__main__":
    main()
