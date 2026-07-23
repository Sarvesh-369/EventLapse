import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Rectangle, RIGHT, LEFT, UP, DOWN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

FIXED_TASK_DURATION = 20.0

class EventCountingScene(Scene):
    def __init__(self, N: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.seed = seed
        self.contact_events = []
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 3)
        ball_color = colors[0]
        wall_color = colors[1]

        rng = random.Random(self.seed)
        orientation = self.seed % 2
        ball_radius = rng.uniform(0.25, 0.35)
        ball = Circle(radius=ball_radius, color=ball_color, fill_opacity=1)

        wall_dist = rng.uniform(2.5, 3.5)
        wall1 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1)
        wall2 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1)

        # Pick random interior start position away from both walls
        start_offset = rng.uniform(-0.5 * wall_dist, 0.5 * wall_dist)
        initial_target_positive = rng.choice([True, False])

        if orientation == 0:
            wall1.move_to(LEFT * wall_dist)
            wall2.move_to(RIGHT * wall_dist)
            ball.move_to(RIGHT * start_offset)
        else:
            wall1.rotate(1.5708)
            wall2.rotate(1.5708)
            wall1.move_to(DOWN * wall_dist)
            wall2.move_to(UP * wall_dist)
            ball.move_to(UP * start_offset)

        self.add(wall1, wall2, ball)
        self.wait(0.2)
        current_time = 0.2

        # Initial move from interior start position to first wall
        if orientation == 0:
            first_target = (RIGHT if initial_target_positive else LEFT) * (wall_dist - ball_radius - 0.1)
            dist_first = abs(first_target[0] - start_offset)
        else:
            first_target = (UP if initial_target_positive else DOWN) * (wall_dist - ball_radius - 0.1)
            dist_first = abs(first_target[1] - start_offset)

        first_leg_duration = max(0.3, dist_first / 3.0)
        self.play(ball.animate.move_to(first_target), run_time=first_leg_duration)
        current_time += first_leg_duration

        wall_id = "wall_right" if initial_target_positive else "wall_left"
        dir_before = "right" if initial_target_positive else "left"
        dir_after = "left" if initial_target_positive else "right"

        self.contact_events.append({
            "contact_index": 1,
            "timestamp": round(current_time, 2),
            "wall_identity": wall_id,
            "direction_before": dir_before,
            "direction_after": dir_after,
            "running_count": 1
        })

        current_target_positive = not initial_target_positive
        one_way_duration = 1.0

        for i in range(1, self.N):
            if orientation == 0:
                target_pos = (RIGHT if current_target_positive else LEFT) * (wall_dist - ball_radius - 0.1)
            else:
                target_pos = (UP if current_target_positive else DOWN) * (wall_dist - ball_radius - 0.1)

            self.play(ball.animate.move_to(target_pos), run_time=one_way_duration)
            current_time += one_way_duration

            wall_id = "wall_right" if current_target_positive else "wall_left"
            dir_before = "right" if current_target_positive else "left"
            dir_after = "left" if current_target_positive else "right"

            self.contact_events.append({
                "contact_index": i + 1,
                "timestamp": round(current_time, 2),
                "wall_identity": wall_id,
                "direction_before": dir_before,
                "direction_after": dir_after,
                "running_count": i + 1
            })

            current_target_positive = not current_target_positive

        # Calculate remaining wait time to enforce FIXED_TASK_DURATION (20.0s)
        question_duration = 3.7
        remaining_wait = max(0.2, FIXED_TASK_DURATION - current_time - question_duration)
        self.wait(remaining_wait)
        current_time += remaining_wait

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="How many times did the ball contact the walls?",
            format_instruction="Answer with a single integer (e.g. 4)."
        )
        current_time += question_duration
        self.actual_duration = round(current_time, 2)

class EventCountingGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "event_counting"

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
        sample_id = f"counting_N{N}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            EventCountingScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            N=N,
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
            "events": scene.contact_events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            "The following events occurred in the video:"
        ]
        if scene.contact_events:
            for e in scene.contact_events:
                cot_lines.append(f"- At {e['timestamp']:.2f}s: Ball contacted {e['wall_identity']} (count={e['running_count']})")
        else:
            cot_lines.append("- No wall contact events occurred.")

        cot_lines.extend([
            "",
            "### Step 1: Understand the Goal",
            "We need to count the total number of times the ball contacts the walls.",
            "",
            "### Step 2: Analyze Contact Events",
            f"Scanning event log: {len(scene.contact_events)} wall contact events detected.",
            "",
            "### Step 3: Derive the Answer",
            f"Total counted contact events: {N}",
            "",
            "### Step 4: Verification",
            f"Event schedule analysis confirmed {N} contacts. Matches.",
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
