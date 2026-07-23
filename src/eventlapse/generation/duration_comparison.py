import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Rectangle, LEFT, RIGHT, UP, DOWN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

class DurationComparisonScene(Scene):
    def __init__(self, r: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.ratio = r
        self.seed = seed
        self.shorter_duration = 2.0
        self.longer_duration = 2.0 * r
        self.events = []
        self.longer_object = ""

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 4)
        top_obj_color, btm_obj_color = colors[0], colors[1]
        top_zone_color, btm_zone_color = colors[2], colors[3]

        rng = random.Random(self.seed)
        top_is_longer = rng.choice([True, False])

        dur_top = self.longer_duration if top_is_longer else self.shorter_duration
        dur_btm = self.shorter_duration if top_is_longer else self.longer_duration
        self.longer_object = "top object" if top_is_longer else "bottom object"

        zone_top = Rectangle(width=1.5, height=1.5, color=top_zone_color, fill_opacity=0.3).move_to(UP * 1.5)
        zone_btm = Rectangle(width=1.5, height=1.5, color=btm_zone_color, fill_opacity=0.3).move_to(DOWN * 1.5)

        obj_top = Circle(radius=0.4, color=top_obj_color, fill_opacity=1).move_to(LEFT * 5.0 + UP * 1.5)
        obj_btm = Square(side_length=0.8, color=btm_obj_color, fill_opacity=1).move_to(LEFT * 5.0 + DOWN * 1.5)

        self.add(zone_top, zone_btm, obj_top, obj_btm)

        entry_time = 1.0
        self.wait(0.5)
        self.play(
            obj_top.animate.move_to(UP * 1.5),
            obj_btm.animate.move_to(DOWN * 1.5),
            run_time=entry_time
        )
        t_enter = 0.5 + entry_time

        self.events.append({"object": "top object", "event": "entry", "timestamp": round(t_enter, 2)})
        self.events.append({"object": "bottom object", "event": "entry", "timestamp": round(t_enter, 2)})

        if dur_top < dur_btm:
            self.wait(dur_top)
            t_exit_top = t_enter + dur_top
            self.play(obj_top.animate.move_to(RIGHT * 5.0 + UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "exit", "timestamp": round(t_exit_top, 2), "dwell": dur_top})

            diff = dur_btm - dur_top
            self.wait(max(0.1, diff - 1.0))
            t_exit_btm = t_enter + dur_btm
            self.play(obj_btm.animate.move_to(RIGHT * 5.0 + DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "exit", "timestamp": round(t_exit_btm, 2), "dwell": dur_btm})
        else:
            self.wait(dur_btm)
            t_exit_btm = t_enter + dur_btm
            self.play(obj_btm.animate.move_to(RIGHT * 5.0 + DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "exit", "timestamp": round(t_exit_btm, 2), "dwell": dur_btm})

            diff = dur_top - dur_btm
            self.wait(max(0.1, diff - 1.0))
            t_exit_top = t_enter + dur_top
            self.play(obj_top.animate.move_to(RIGHT * 5.0 + UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "exit", "timestamp": round(t_exit_top, 2), "dwell": dur_top})

        self.wait(0.5)

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Which object remained stopped longer?",
            format_instruction="Answer with 'top object' or 'bottom object'."
        )

class DurationComparisonGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "duration_comparison"

    @property
    def control_parameter_name(self) -> str:
        return "r"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        r = float(control_value)
        sample_id = f"duration_r{r:.2f}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            DurationComparisonScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            r=r,
            seed=seed
        )

        question = "Which object remained stopped longer?"
        exact_answer = scene.longer_object

        trace_data = {
            "steps": [
                {
                    "state": {"top_dwell": scene.longer_duration if exact_answer=="top object" else scene.shorter_duration, "bottom_dwell": scene.shorter_duration if exact_answer=="top object" else scene.longer_duration},
                    "event": {"type": e["event"], "object": e["object"], "timestamp": e["timestamp"]},
                    "operation": {"action": "measure_duration", "dwell_sec": e.get("dwell", None)}
                } for e in scene.events
            ],
            "duration_ratio": r,
            "longer_object": exact_answer,
            "events": scene.events
        }

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": r,
            "seed": seed
        }

        dest_video, dest_trace, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(3.5 + 2.0 * r + 3.7, 2)

        shutil.rmtree(temp_dir, ignore_errors=True)

        return SyntheticSample(
            sample_id=sample_id,
            task_name=self.task_name,
            control_parameter_name=self.control_parameter_name,
            control_parameter_value=r,
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
