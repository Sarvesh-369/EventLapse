import shutil
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Type
from manim import config, Scene, Text, Write, UP

from eventlapse.utils.caching import compute_file_checksum

def render_question_card(scene: Scene, question: str, format_instruction: str = ""):
    """
    Renders the question text on screen at the end of the video clip (similar to MORSE benchmark).
    """
    scene.clear()
    scene.wait(0.2)
    lines = [question]
    if format_instruction:
        lines.extend(["", format_instruction])

    question_texts = []
    start_y = 1.2
    line_height = 0.8
    for i, line in enumerate(lines):
        if line:
            t = Text(line, font_size=28 if i == 0 else 22)
            t.move_to(UP * (start_y - i * line_height))
            question_texts.append(t)

    scene.play(*[Write(t) for t in question_texts], run_time=1.0)
    scene.wait(2.5)

def render_manim_scene(
    scene_class: Type[Scene],
    output_filename: str,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30,
    **scene_kwargs
) -> Tuple[Path, Scene, Path]:
    temp_dir = Path(tempfile.mkdtemp())
    config.media_dir = str(temp_dir / "media")
    config.video_dir = str(temp_dir)
    config.output_file = output_filename
    config.verbosity = "WARNING"
    config.pixel_width = resolution[0]
    config.pixel_height = resolution[1]
    config.frame_rate = fps
    config.preview = False

    scene = scene_class(**scene_kwargs)
    scene.render()

    rendered_mp4 = temp_dir / f"{output_filename}.mp4"
    if not rendered_mp4.exists():
        found = list(temp_dir.glob("*.mp4"))
        if found:
            rendered_mp4 = found[0]

    return rendered_mp4, scene, temp_dir

def save_sample_outputs(
    sample_id: str,
    task_name: str,
    rendered_mp4_path: Path,
    trace_data: Dict[str, Any],
    cot_text: str,
    gt_data: Dict[str, Any],
    base_output_dir: Path
) -> Tuple[Path, Path, Path, Path]:
    videos_dir = base_output_dir / "videos" / task_name
    traces_dir = base_output_dir / "traces" / task_name
    reasoning_dir = base_output_dir / "reasoning_traces" / task_name
    gt_dir = base_output_dir / "gt" / task_name

    videos_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    dest_video_path = videos_dir / f"{sample_id}.mp4"
    dest_trace_path = traces_dir / f"{sample_id}_trace.json"
    dest_cot_path = reasoning_dir / f"{sample_id}.txt"
    dest_gt_path = gt_dir / f"{sample_id}_gt.json"

    if rendered_mp4_path.exists():
        shutil.copy(str(rendered_mp4_path), str(dest_video_path))

    with open(dest_trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    with open(dest_cot_path, "w") as f:
        f.write(cot_text)

    with open(dest_gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    return dest_video_path, dest_trace_path, dest_cot_path, dest_gt_path
