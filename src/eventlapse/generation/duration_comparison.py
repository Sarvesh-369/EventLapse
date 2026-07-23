import random
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Rectangle, LEFT, RIGHT, UP, DOWN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

FIXED_TASK_DURATION = 13.0

class DurationComparisonScene(Scene):
    def __init__(self, r: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.ratio = r
        self.seed = seed
        self.shorter_duration = 2.0
        self.longer_duration = 2.0 * r
        self.events = []
        self.longer_object = ""
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"eventlapse_duration_{self.seed}_{self.ratio:.2f}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        colors = get_nuisance_colors(self.seed, 4)
        top_obj_color, btm_obj_color = colors[0], colors[1]
        top_zone_color, btm_zone_color = colors[2], colors[3]

        top_is_longer = rng.choice([True, False])
        dur_top = self.longer_duration if top_is_longer else self.shorter_duration
        dur_btm = self.shorter_duration if top_is_longer else self.longer_duration
        self.longer_object = "top object" if top_is_longer else "bottom object"

        # Stagger entry times so that entry order is also randomized
        stagger_top_first = rng.choice([True, False])
        stagger_delay = rng.uniform(0.1, 0.4)

        t_enter_top = 1.5 + (0.0 if stagger_top_first else stagger_delay)
        t_enter_btm = 1.5 + (stagger_delay if stagger_top_first else 0.0)

        zone_top = Rectangle(width=1.5, height=1.5, color=top_zone_color, fill_opacity=0.3).move_to(UP * 1.5)
        zone_btm = Rectangle(width=1.5, height=1.5, color=btm_zone_color, fill_opacity=0.3).move_to(DOWN * 1.5)

        obj_top = Circle(radius=0.4, color=top_obj_color, fill_opacity=1).move_to(LEFT * 5.0 + UP * 1.5)
        obj_btm = Square(side_length=0.8, color=btm_obj_color, fill_opacity=1).move_to(LEFT * 5.0 + DOWN * 1.5)

        self.add(zone_top, zone_btm, obj_top, obj_btm)
        self.wait(0.5)

        # Animate entries according to randomized stagger
        if stagger_top_first:
            self.play(obj_top.animate.move_to(UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "entry", "timestamp": round(t_enter_top, 2)})
            self.wait(stagger_delay)
            self.play(obj_btm.animate.move_to(DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "entry", "timestamp": round(t_enter_btm, 2)})
        else:
            self.play(obj_btm.animate.move_to(DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "entry", "timestamp": round(t_enter_btm, 2)})
            self.wait(stagger_delay)
            self.play(obj_top.animate.move_to(UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "entry", "timestamp": round(t_enter_top, 2)})

        t_exit_top = t_enter_top + dur_top
        t_exit_btm = t_enter_btm + dur_btm

        # Animate exits in order of absolute exit timestamp
        if t_exit_top < t_exit_btm:
            wait_1 = max(0.1, t_exit_top - (t_enter_btm if not stagger_top_first else t_enter_top + 1.0))
            self.wait(wait_1)
            self.play(obj_top.animate.move_to(RIGHT * 5.0 + UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "exit", "timestamp": round(t_exit_top, 2), "dwell": round(dur_top, 2)})

            wait_2 = max(0.1, t_exit_btm - t_exit_top - 1.0)
            self.wait(wait_2)
            self.play(obj_btm.animate.move_to(RIGHT * 5.0 + DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "exit", "timestamp": round(t_exit_btm, 2), "dwell": round(dur_btm, 2)})
            current_time = t_exit_btm + 1.0
        else:
            wait_1 = max(0.1, t_exit_btm - (t_enter_top if stagger_top_first else t_enter_btm + 1.0))
            self.wait(wait_1)
            self.play(obj_btm.animate.move_to(RIGHT * 5.0 + DOWN * 1.5), run_time=1.0)
            self.events.append({"object": "bottom object", "event": "exit", "timestamp": round(t_exit_btm, 2), "dwell": round(dur_btm, 2)})

            wait_2 = max(0.1, t_exit_top - t_exit_btm - 1.0)
            self.wait(wait_2)
            self.play(obj_top.animate.move_to(RIGHT * 5.0 + UP * 1.5), run_time=1.0)
            self.events.append({"object": "top object", "event": "exit", "timestamp": round(t_exit_top, 2), "dwell": round(dur_top, 2)})
            current_time = t_exit_top + 1.0

        # Calculate remaining wait time to enforce FIXED_TASK_DURATION (13.0s)
        question_duration = 3.7
        remaining_wait = max(0.2, FIXED_TASK_DURATION - current_time - question_duration)
        self.wait(remaining_wait)
        current_time += remaining_wait

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Which object remained stopped longer?",
            format_instruction="Answer with 'top object' or 'bottom object'."
        )
        current_time += question_duration
        self.actual_duration = round(current_time, 2)

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

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            "Two objects entering and leaving marked stopping zones.",
            "",
            "### Step 1: Track Entry and Exit Timestamps"
        ]
        for e in scene.events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: {e['object']} {e['event']}")

        cot_lines.extend([
            "",
            "### Step 2: Calculate Dwell Durations",
            f"Top object dwell duration vs Bottom object dwell duration (ratio r={r:.2f}).",
            "",
            "### Step 3: Compare Durations",
            f"Longer stopped object: {exact_answer}.",
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
            "control_value": r,
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
            control_parameter_value=r,
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
