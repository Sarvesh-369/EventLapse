import random
import math
import hashlib
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Line, Circle, WHITE

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "teal", "pink", "lime", "brown", "gold", "silver", "maroon", "navy"]
FIXED_TASK_DURATION = 18.0

class TemporalOrderingScene(Scene):
    def __init__(self, L: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.L = int(L)
        self.seed = seed
        self.crossing_events = []
        self.queried_k = math.ceil(self.L / 2)
        self.correct_object_color = ""
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"eventlapse_ordering_{self.seed}_{self.L}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        colors = get_nuisance_colors(self.seed, self.L)

        # Continuous angle rotation theta in [0, 2*pi)
        rotation_angle = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(rotation_angle), math.sin(rotation_angle), 0]) # motion direction
        v = np.array([-math.sin(rotation_angle), math.cos(rotation_angle), 0]) # finish line direction

        finish_dist = 3.0
        finish_line = Line(u * finish_dist + v * 3.5, u * finish_dist - v * 3.5, color=WHITE, stroke_width=4)
        self.add(finish_line)

        order_indices = list(range(self.L))
        rng.shuffle(order_indices)

        object_metadata = []
        for idx in range(self.L):
            c_hex = colors[idx]
            c_name = COLOR_NAMES[idx % len(COLOR_NAMES)]
            object_metadata.append({"id": idx, "hex": c_hex, "color_name": c_name})

        lane_heights = [3.0 - i * (6.0 / max(1, self.L - 1)) for i in range(self.L)] if self.L > 1 else [0.0]

        interval = 0.8
        current_time = 0.5

        for rank, obj_idx in enumerate(order_indices):
            obj_meta = object_metadata[obj_idx]
            lane_offset = lane_heights[obj_idx]

            start_pos = -u * 6.0 + v * lane_offset
            target_pos = u * 4.5 + v * lane_offset

            shape = Circle(radius=0.25, color=obj_meta["hex"], fill_opacity=1).move_to(start_pos)
            self.add(shape)

            self.play(shape.animate.move_to(target_pos), run_time=1.2)
            crossing_time = round(current_time + 0.6, 2)
            current_time += interval

            self.crossing_events.append({
                "rank": rank + 1,
                "timestamp": crossing_time,
                "color_name": obj_meta["color_name"],
                "color_hex": obj_meta["hex"],
                "object_id": obj_idx
            })

            if (rank + 1) == self.queried_k:
                self.correct_object_color = obj_meta["color_name"]

        # Calculate remaining wait time to enforce FIXED_TASK_DURATION (18.0s)
        question_duration = 3.7
        remaining_wait = max(0.2, FIXED_TASK_DURATION - current_time - question_duration)
        self.wait(remaining_wait)
        current_time += remaining_wait

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question=f"Which object crossed the finish line in position {self.queried_k}?",
            format_instruction="Answer with the color name (e.g. red)."
        )
        current_time += question_duration
        self.actual_duration = round(current_time, 2)

class TemporalOrderingGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "temporal_ordering"

    @property
    def control_parameter_name(self) -> str:
        return "L"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        L = int(control_value)
        sample_id = f"ordering_L{L}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            TemporalOrderingScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            L=L,
            seed=seed
        )

        question = f"Which object crossed the finish line in position {scene.queried_k}?"
        exact_answer = scene.correct_object_color

        trace_data = {
            "steps": [
                {
                    "state": {"rank": e["rank"], "color": e["color_name"]},
                    "event": {"type": "finish_line_crossing", "timestamp": e["timestamp"], "color": e["color_name"]},
                    "operation": {"action": "record_crossing_rank", "rank": e["rank"]}
                } for e in scene.crossing_events
            ],
            "sequence_length": L,
            "queried_rank": scene.queried_k,
            "correct_color": exact_answer,
            "events": scene.crossing_events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Sequence of L={L} colored objects crossing the finish line.",
            "",
            "### Step 1: Record Finish Line Crossings"
        ]
        for e in scene.crossing_events:
            cot_lines.append(f"- Position {e['rank']} at {e['timestamp']:.2f}s: {e['color_name']} object")

        cot_lines.extend([
            "",
            "### Step 2: Identify Position {scene.queried_k}",
            f"Object at position {scene.queried_k}: {exact_answer}.",
            "",
            f"\\boxed{{{exact_answer}}}"
        ])
        cot_text = "\n".join(cot_lines)

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": L,
            "seed": seed
        }

        dest_video, dest_question, dest_trace, dest_cot, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, cot_text, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)

        rendered_duration = scene.actual_duration
        if dest_video.exists():
            try:
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", str(dest_video)]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                rendered_duration = round(float(res.stdout.strip()), 2)
            except Exception:
                pass

        shutil.rmtree(temp_dir, ignore_errors=True)

        return SyntheticSample(
            sample_id=sample_id,
            task_name=self.task_name,
            control_parameter_name=self.control_parameter_name,
            control_parameter_value=L,
            seed=seed,
            video_path=dest_video,
            question=question,
            exact_answer=exact_answer,
            executable_trace=trace_data,
            cot_text=cot_text,
            generation_config={"resolution": resolution, "fps": fps},
            duration=rendered_duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
