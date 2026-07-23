import random
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Line, Circle, LEFT, RIGHT, UP, DOWN, WHITE

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "teal", "pink", "lime", "brown", "gold", "silver", "maroon", "navy"]

class TemporalOrderingScene(Scene):
    def __init__(self, L: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.L = int(L)
        self.seed = seed
        self.crossing_events = []
        self.queried_k = math.ceil(self.L / 2)
        self.correct_object_color = ""

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, self.L)

        finish_x = 3.0
        finish_line = Line(UP * 3.5 + RIGHT * finish_x, DOWN * 3.5 + RIGHT * finish_x, color=WHITE, stroke_width=4)
        self.add(finish_line)

        rng = random.Random(self.seed)
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
            lane_y = lane_heights[obj_idx]
            shape = Circle(radius=0.25, color=obj_meta["hex"], fill_opacity=1)
            shape.move_to(LEFT * 6.0 + UP * lane_y)
            self.add(shape)

            self.play(shape.animate.move_to(RIGHT * 4.5 + UP * lane_y), run_time=1.2)
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

        self.wait(0.5)

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question=f"Which object crossed the finish line in position {self.queried_k}?",
            format_instruction="Answer with the color name (e.g. red)."
        )

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

        k = scene.queried_k
        question = f"Which object crossed the finish line in position {k}?"
        exact_answer = scene.correct_object_color

        trace_data = {
            "steps": [
                {
                    "state": {"finish_line_crossings": e["rank"]},
                    "event": {"type": "finish_line_crossing", "timestamp": e["timestamp"], "color": e["color_name"]},
                    "operation": {"action": "record_crossing_order", "rank": e["rank"], "color": e["color_name"]}
                } for e in scene.crossing_events
            ],
            "queried_k": k,
            "ordered_crossings": [e["color_name"] for e in scene.crossing_events],
            "events": scene.crossing_events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Objects crossing finish line sequentially (sequence length L={L}).",
            "",
            "### Step 1: Record Crossing Order"
        ]
        for e in scene.crossing_events:
            cot_lines.append(f"- Position {e['rank']} at {e['timestamp']:.2f}s: {e['color_name']} object")

        cot_lines.extend([
            "",
            f"### Step 2: Extract Queried Position k={k}",
            f"Object crossing in position {k} is {exact_answer}.",
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

        dest_video, dest_trace, dest_cot, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, cot_text, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(0.5 + L * 1.2 + 3.7, 2)

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
            duration=duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
