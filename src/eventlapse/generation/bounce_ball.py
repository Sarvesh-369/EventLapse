import random
import math
import hashlib
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Rectangle

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

FIXED_TASK_DURATION = 24.0

class BounceBallScene(Scene):
    def __init__(self, N: int, F: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.F = float(F)
        self.seed = seed
        self.contact_events = []
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"eventlapse_bounce_{self.seed}_{self.N}_{self.F}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        colors = get_nuisance_colors(self.seed, 3)
        ball_color = colors[0]
        wall_color = colors[1]

        rotation_angle = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(rotation_angle), math.sin(rotation_angle), 0])

        ball_radius = rng.uniform(0.25, 0.35)
        ball = Circle(radius=ball_radius, color=ball_color, fill_opacity=1)

        wall_dist = rng.uniform(2.5, 3.5)
        wall1 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1).move_to(-u * wall_dist).rotate(rotation_angle)
        wall2 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1).move_to(u * wall_dist).rotate(rotation_angle)

        start_offset = rng.uniform(-0.3 * wall_dist, 0.3 * wall_dist)
        ball.move_to(u * start_offset)

        self.add(wall1, wall2, ball)

        one_way_duration = max(0.1, 1.0 / self.F)
        if self.N == 0:
            est_active_time = 2.0
        else:
            est_active_time = (self.N + 1) * one_way_duration

        total_idle = max(0.4, FIXED_TASK_DURATION - est_active_time)
        pre_wait = rng.uniform(0.3, max(0.3, total_idle - 0.3))

        self.wait(pre_wait)
        current_time = pre_wait

        if self.N == 0:
            end_offset = rng.uniform(-0.3 * wall_dist, 0.3 * wall_dist)
            self.play(ball.animate.move_to(u * end_offset), run_time=2.0)
            current_time += 2.0
        else:
            initial_target_positive = rng.choice([True, False])
            first_target = (u if initial_target_positive else -u) * (wall_dist - ball_radius - 0.1)
            dist_first = abs(np.linalg.norm(first_target - (u * start_offset)))

            first_leg_duration = max(0.1, (dist_first / (2 * wall_dist)) * one_way_duration)

            self.play(ball.animate.move_to(first_target), run_time=first_leg_duration)
            current_time += first_leg_duration

            wall_id = "wall_positive" if initial_target_positive else "wall_negative"

            self.contact_events.append({
                "contact_index": 1,
                "timestamp": round(current_time, 2),
                "wall_identity": wall_id,
                "running_count": 1
            })

            current_target_positive = not initial_target_positive

            for i in range(1, self.N):
                target_pos = (u if current_target_positive else -u) * (wall_dist - ball_radius - 0.1)

                self.play(ball.animate.move_to(target_pos), run_time=one_way_duration)
                current_time += one_way_duration

                wall_id = "wall_positive" if current_target_positive else "wall_negative"

                self.contact_events.append({
                    "contact_index": i + 1,
                    "timestamp": round(current_time, 2),
                    "wall_identity": wall_id,
                    "running_count": i + 1
                })

                current_target_positive = not current_target_positive

            end_offset = rng.uniform(-0.3 * wall_dist, 0.3 * wall_dist)
            end_target = u * end_offset
            last_wall_val = (u if not current_target_positive else -u) * (wall_dist - ball_radius - 0.1)
            dist_end = abs(np.linalg.norm(end_target - last_wall_val))

            final_leg_duration = max(0.1, (dist_end / (2 * wall_dist)) * one_way_duration)
            self.play(ball.animate.move_to(end_target), run_time=final_leg_duration)
            current_time += final_leg_duration

        post_wait = max(0.1, FIXED_TASK_DURATION - current_time)
        self.wait(post_wait)
        current_time += post_wait
        self.actual_duration = round(current_time, 2)

class BounceBallGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "bounce_ball"

    @property
    def control_parameter_name(self) -> str:
        return "N"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        frequency: float = 1.0,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        N = int(control_value)
        F = float(frequency)
        sample_id = f"bounce_N{N}_F{F}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            BounceBallScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            N=N,
            F=F,
            seed=seed
        )

        question = "How many times did the ball contact the walls?"
        exact_answer = str(N)

        trace_data = {
            "steps": [
                {
                    "state": {"ball_position": "in_motion", "running_count": e["running_count"]},
                    "event": {"type": "wall_contact", "timestamp": e["timestamp"], "wall": e["wall_identity"]},
                    "operation": {"action": "increment_counter", "count": e["running_count"]}
                } for e in scene.contact_events
            ],
            "final_count": N,
            "events": scene.contact_events,
            "frequency_hz": F
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let me analyze the video step by step.",
            "",
            "### Scene Description",
            f"Ball bouncing between two walls at frequency {F} Hz."
        ]
        for e in scene.contact_events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: Ball contacted {e['wall_identity']} (count={e['running_count']})")

        cot_lines.extend([
            "",
            "### Step 1: Track Contact Events",
            f"Total wall contact events detected: {N}.",
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
            "frequency_hz": F,
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
            generation_config={"resolution": resolution, "fps": fps, "frequency": F},
            duration=rendered_duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
