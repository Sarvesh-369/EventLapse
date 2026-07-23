import random
import math
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Line, Circle, LEFT, RIGHT, UP, DOWN, ValueTracker, linear

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

class EventFrequencyScene(Scene):
    def __init__(self, f_slower: float, seed: int, duration: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.f_slower = f_slower
        self.seed = seed
        self.clip_duration = duration
        self.events = []
        self.faster_side = ""

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 3)
        left_color, right_color = colors[0], colors[1]

        rng = random.Random(self.seed)
        faster_is_left = rng.choice([True, False])
        self.faster_side = "left" if faster_is_left else "right"

        f_left = self.f_slower * (1.5 if faster_is_left else 1.0)
        f_right = self.f_slower * (1.0 if faster_is_left else 1.5)

        initial_phase_left = rng.uniform(0, math.pi / 4)
        initial_phase_right = rng.uniform(0, math.pi / 4)
        amplitude = rng.uniform(0.6, 0.9)

        pivot_left = LEFT * 3 + UP * 2
        pivot_right = RIGHT * 3 + UP * 2
        arm_len = 3.0

        bob_left = Circle(radius=0.3, color=left_color, fill_opacity=1)
        bob_right = Circle(radius=0.3, color=right_color, fill_opacity=1)

        rod_left = Line(pivot_left, pivot_left + DOWN * arm_len, color=left_color, stroke_width=4)
        rod_right = Line(pivot_right, pivot_right + DOWN * arm_len, color=right_color, stroke_width=4)

        self.add(rod_left, rod_right, bob_left, bob_right)

        time_tracker = ValueTracker(0)

        def update_left(m):
            t = time_tracker.get_value()
            angle = amplitude * math.sin(2 * math.pi * f_left * t + initial_phase_left)
            end_pos = pivot_left + np.array([arm_len * math.sin(angle), -arm_len * math.cos(angle), 0])
            rod_left.put_start_and_end_on(pivot_left, end_pos)
            bob_left.move_to(end_pos)

        def update_right(m):
            t = time_tracker.get_value()
            angle = amplitude * math.sin(2 * math.pi * f_right * t + initial_phase_right)
            end_pos = pivot_right + np.array([arm_len * math.sin(angle), -arm_len * math.cos(angle), 0])
            rod_right.put_start_and_end_on(pivot_right, end_pos)
            bob_right.move_to(end_pos)

        bob_left.add_updater(update_left)
        bob_right.add_updater(update_right)

        self.play(time_tracker.animate.set_value(self.clip_duration), run_time=self.clip_duration, rate_func=linear)
        self.wait(0.2)

        bob_left.remove_updater(update_left)
        bob_right.remove_updater(update_right)

        for side, f in [("left", f_left), ("right", f_right)]:
            period = 1.0 / f
            num_cycles = int(self.clip_duration / period)
            for c in range(num_cycles):
                t_cycle = round((c + 1) * period, 2)
                if t_cycle <= self.clip_duration:
                    self.events.append({
                        "side": side,
                        "frequency": f,
                        "completed_cycle": c + 1,
                        "timestamp": t_cycle
                    })

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Which pendulum oscillated faster?",
            format_instruction="Answer with 'left' or 'right'."
        )

class EventFrequencyGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "event_frequency"

    @property
    def control_parameter_name(self) -> str:
        return "f"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        f_slower = float(control_value)
        sample_id = f"frequency_f{f_slower:.1f}_seed{seed}"
        duration = 4.0

        rendered_file, scene, temp_dir = render_manim_scene(
            EventFrequencyScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            f_slower=f_slower,
            seed=seed,
            duration=duration
        )

        question = "Which pendulum oscillated faster?"
        exact_answer = scene.faster_side

        trace_data = {
            "steps": [
                {
                    "state": {"left_frequency": f_slower if exact_answer == "right" else f_slower*1.5, "right_frequency": f_slower if exact_answer == "left" else f_slower*1.5},
                    "event": {"type": "cycle_completion", "timestamp": e["timestamp"], "side": e["side"]},
                    "operation": {"action": "track_oscillation_rate", "completed_cycles": e["completed_cycle"]}
                } for e in scene.events
            ],
            "faster_side": exact_answer,
            "events": scene.events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            "Left pendulum and Right pendulum oscillating at different frequencies.",
            "",
            "### Step 1: Track Cycles",
        ]
        for e in scene.events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: {e['side'].capitalize()} pendulum completed cycle {e['completed_cycle']} (freq={e['frequency']:.2f} Hz)")

        cot_lines.extend([
            "",
            "### Step 2: Compare Oscillation Frequencies",
            f"Left pendulum rate vs Right pendulum rate comparison shows '{exact_answer}' side completed cycles faster.",
            "",
            "### Step 3: Derive Answer",
            f"Faster pendulum: {exact_answer}",
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
            "control_value": f_slower,
            "seed": seed
        }

        dest_video, dest_trace, dest_cot, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, cot_text, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)

        shutil.rmtree(temp_dir, ignore_errors=True)

        return SyntheticSample(
            sample_id=sample_id,
            task_name=self.task_name,
            control_parameter_name=self.control_parameter_name,
            control_parameter_value=f_slower,
            seed=seed,
            video_path=dest_video,
            question=question,
            exact_answer=exact_answer,
            executable_trace=trace_data,
            cot_text=cot_text,
            generation_config={"resolution": resolution, "fps": fps},
            duration=duration + 3.9,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
