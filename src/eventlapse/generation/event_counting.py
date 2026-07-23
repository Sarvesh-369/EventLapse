import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Rectangle, RIGHT, LEFT, UP, DOWN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

class EventCountingScene(Scene):
    def __init__(self, N: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.seed = seed
        self.contact_events = []

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 3)
        ball_color = colors[0]
        wall_color = colors[1]

        orientation = self.seed % 2
        ball_radius = random.uniform(0.25, 0.35)
        ball = Circle(radius=ball_radius, color=ball_color, fill_opacity=1)

        wall_dist = random.uniform(2.5, 3.5)
        wall1 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1)
        wall2 = Rectangle(width=0.2, height=3.0, color=wall_color, fill_opacity=1)

        if orientation == 0:
            wall1.move_to(LEFT * wall_dist)
            wall2.move_to(RIGHT * wall_dist)
            ball.move_to(LEFT * (wall_dist - ball_radius - 0.1))
            current_target_right = True
        else:
            wall1.rotate(1.5708)
            wall2.rotate(1.5708)
            wall1.move_to(DOWN * wall_dist)
            wall2.move_to(UP * wall_dist)
            ball.move_to(DOWN * (wall_dist - ball_radius - 0.1))
            current_target_right = True

        self.add(wall1, wall2, ball)
        self.wait(0.2)
        current_time = 0.2

        one_way_duration = 1.0

        for i in range(self.N):
            if orientation == 0:
                target_pos = (RIGHT if current_target_right else LEFT) * (wall_dist - ball_radius - 0.1)
            else:
                target_pos = (UP if current_target_right else DOWN) * (wall_dist - ball_radius - 0.1)

            self.play(ball.animate.move_to(target_pos), run_time=one_way_duration)
            current_time += one_way_duration

            wall_id = "wall_right" if current_target_right else "wall_left"
            dir_before = "right" if current_target_right else "left"
            dir_after = "left" if current_target_right else "right"

            self.contact_events.append({
                "contact_index": i + 1,
                "timestamp": round(current_time, 2),
                "wall_identity": wall_id,
                "direction_before": dir_before,
                "direction_after": dir_after,
                "running_count": i + 1
            })

            current_target_right = not current_target_right

        self.wait(0.5)

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

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": N,
            "seed": seed
        }

        dest_video, dest_trace, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(0.7 + N * 1.0, 2)

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
            generation_config={"resolution": resolution, "fps": fps},
            duration=duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
