import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Square, LEFT, RIGHT, UP, DOWN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green", "yellow"]

class FuturePredictionScene(Scene):
    def __init__(self, h: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.h = int(h)
        self.seed = seed
        self.observed_sequence = []
        self.future_sequence = []
        self.target_future_color = ""

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 4)

        rng = random.Random(self.seed)
        color_order = list(range(4))
        rng.shuffle(color_order)

        positions = [
            UP * 1.5 + LEFT * 1.5,
            UP * 1.5 + RIGHT * 1.5,
            DOWN * 1.5 + RIGHT * 1.5,
            DOWN * 1.5 + LEFT * 1.5
        ]

        squares = []
        square_metadata = []
        for i in range(4):
            c_hex = colors[color_order[i]]
            c_name = COLOR_NAMES[color_order[i] % len(COLOR_NAMES)]
            sq = Square(side_length=1.2, color=c_hex, fill_opacity=0.3).move_to(positions[i])
            squares.append(sq)
            square_metadata.append({"id": i, "color_hex": c_hex, "color_name": c_name, "square": sq})
            self.add(sq)

        self.wait(0.3)
        current_time = 0.3

        num_observed_flashes = 8
        flash_dur = 0.4
        interval = 0.3

        start_offset = rng.randint(0, 3)

        for step in range(num_observed_flashes):
            idx = (start_offset + step) % 4
            meta = square_metadata[idx]
            sq = meta["square"]

            self.play(sq.animate.set_fill(meta["color_hex"], opacity=1.0).scale(1.1), run_time=flash_dur / 2)
            self.play(sq.animate.set_fill(meta["color_hex"], opacity=0.3).scale(1.0 / 1.1), run_time=flash_dur / 2)

            flash_time = round(current_time + flash_dur / 2, 2)
            current_time += flash_dur + interval

            self.observed_sequence.append({
                "step": step + 1,
                "timestamp": flash_time,
                "color_name": meta["color_name"],
                "color_hex": meta["color_hex"],
                "square_index": idx
            })

            self.wait(interval)

        for step in range(1, self.h + 1):
            future_idx = (start_offset + num_observed_flashes + step - 1) % 4
            future_meta = square_metadata[future_idx]
            self.future_sequence.append({
                "horizon_step": step,
                "color_name": future_meta["color_name"]
            })
            if step == self.h:
                self.target_future_color = future_meta["color_name"]

        self.wait(0.2)

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question=f"Which colored square will flash {self.h} steps after the video ends?",
            format_instruction="Answer with the color name (e.g. red)."
        )

class FuturePredictionGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "future_prediction"

    @property
    def control_parameter_name(self) -> str:
        return "h"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        h = int(control_value)
        sample_id = f"prediction_h{h}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            FuturePredictionScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            h=h,
            seed=seed
        )

        question = f"Which colored square will flash {h} steps after the video ends?"
        exact_answer = scene.target_future_color

        trace_data = {
            "steps": [
                {
                    "state": {"active_flash_index": e["square_index"]},
                    "event": {"type": "flash", "timestamp": e["timestamp"], "color": e["color_name"]},
                    "operation": {"action": "observe_pattern_step", "step": e["step"]}
                } for e in scene.observed_sequence
            ],
            "prediction_horizon_h": h,
            "observed_sequence": [e["color_name"] for e in scene.observed_sequence],
            "inferred_cyclic_rule": "0 -> 1 -> 2 -> 3 -> 0",
            "future_sequence": scene.future_sequence,
            "target_future_color": exact_answer
        }

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": h,
            "seed": seed
        }

        dest_video, dest_trace, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(0.5 + 8 * 0.7 + 3.9, 2)

        shutil.rmtree(temp_dir, ignore_errors=True)

        return SyntheticSample(
            sample_id=sample_id,
            task_name=self.task_name,
            control_parameter_name=self.control_parameter_name,
            control_parameter_value=h,
            seed=seed,
            video_path=dest_video,
            question=question,
            exact_answer=exact_answer,
            executable_trace=trace_data,
            generation_config={"resolution": resolution, "fps": fps},
            duration=duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
