import random
import math
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Star, Triangle

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

FIXED_TASK_DURATION = 20.0

class BlinkingScene(Scene):
    def __init__(self, N: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.seed = seed
        self.blink_events = []
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"eventlapse_blinking_{self.seed}_{self.N}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        colors = get_nuisance_colors(self.seed, 2)
        obj_color = colors[0]

        shape_choice = rng.choice(["Circle", "Square", "Star", "Triangle"])
        if shape_choice == "Circle":
            obj = Circle(radius=1.0, color=obj_color, fill_opacity=0.2)
        elif shape_choice == "Square":
            obj = Square(side_length=2.0, color=obj_color, fill_opacity=0.2)
        elif shape_choice == "Star":
            obj = Star(n=5, outer_radius=1.2, color=obj_color, fill_opacity=0.2)
        else:
            obj = Triangle(color=obj_color, fill_opacity=0.2).scale(1.5)

        self.add(obj)
        self.wait(0.5)
        current_time = 0.5

        blink_interval = 1.0

        for i in range(self.N):
            # Blink ON (high opacity and bright scale)
            self.play(obj.animate.set_fill(obj_color, opacity=1.0).scale(1.1), run_time=0.2)
            current_time += 0.2

            self.blink_events.append({
                "blink_index": i + 1,
                "timestamp": round(current_time, 2),
                "running_count": i + 1
            })

            # Blink OFF
            self.play(obj.animate.set_fill(obj_color, opacity=0.2).scale(1.0 / 1.1), run_time=0.2)
            current_time += 0.2

            if i < self.N - 1:
                self.wait(blink_interval)
                current_time += blink_interval

        question_duration = 3.7
        remaining_wait = max(0.2, FIXED_TASK_DURATION - current_time - question_duration)
        self.wait(remaining_wait)
        current_time += remaining_wait

        render_question_card(
            self,
            question="How many times did the object blink?",
            format_instruction="Answer with a single integer (e.g. 4)."
        )
        current_time += question_duration
        self.actual_duration = round(current_time, 2)

class BlinkingGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "blinking"

    @property
    def control_parameter_name(self) -> str:
        return "N"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        N = int(control_value)
        sample_id = f"blinking_N{N}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            BlinkingScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            N=N,
            seed=seed
        )

        question = "How many times did the object blink?"
        exact_answer = str(N)

        trace_data = {
            "steps": [
                {
                    "state": {"object_state": "blinking", "running_count": e["running_count"]},
                    "event": {"type": "blink_pulse", "timestamp": e["timestamp"]},
                    "operation": {"action": "increment_counter", "count": e["running_count"]}
                } for e in scene.blink_events
            ],
            "final_count": N,
            "events": scene.blink_events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            "An object pulsing ON and OFF."
        ]
        for e in scene.blink_events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: Object blinked ON (count={e['running_count']})")

        cot_lines.extend([
            "",
            "### Step 1: Track Blink Pulses",
            f"Total blinks detected: {N}.",
            "",
            f"\\boxed{{{N}}}"
        ])
        cot_text = "\n".join(cot_lines)

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": N,
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
            control_parameter_value=N,
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
