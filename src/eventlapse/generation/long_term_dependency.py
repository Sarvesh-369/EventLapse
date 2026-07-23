import random
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Rectangle, Star, VGroup, RIGHT, UP, DOWN, YELLOW, GREEN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green"]
FIXED_TASK_DURATION = 17.0

class LongTermDependencyScene(Scene):
    def __init__(self, D: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.D = int(D)
        self.seed = seed
        self.swap_events = []
        self.marked_parcel_id = 0
        self.final_parcel_in_bin_id = -1
        self.entered_delivery_bin = False
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 3)

        rng = random.Random(self.seed)
        self.marked_parcel_id = 0

        x_positions = [-3.0, 0.0, 3.0]
        y_pos = 1.0

        parcels = []
        parcel_positions = [0, 1, 2]

        marker = None
        for i in range(3):
            c_hex = colors[i]
            c_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            shape = Circle(radius=0.4, color=c_hex, fill_opacity=1).move_to(RIGHT * x_positions[i] + UP * y_pos)

            if i == 0:
                marker = Star(n=5, outer_radius=0.2, color=YELLOW, fill_opacity=1).move_to(RIGHT * x_positions[0] + UP * (y_pos + 0.6))
                mobject = VGroup(shape, marker)
            else:
                mobject = shape

            parcels.append({"id": i, "color_name": c_name, "hex": c_hex, "mobject": mobject})
            self.add(mobject)

        if marker is not None:
            self.play(marker.animate.scale(1.2), run_time=0.3)
            self.play(marker.animate.scale(1/1.2), run_time=0.3)

        bin_box = Rectangle(width=1.8, height=1.5, color=GREEN, fill_opacity=0.3).move_to(RIGHT * 3.0 + DOWN * 2.0)
        self.add(bin_box)

        self.wait(0.4)
        current_time = 1.0

        for s in range(self.D):
            p_a, p_b = rng.sample([0, 1, 2], 2)

            pos_a_idx = parcel_positions[p_a]
            pos_b_idx = parcel_positions[p_b]

            target_x_a = x_positions[pos_b_idx]
            target_x_b = x_positions[pos_a_idx]

            self.play(
                parcels[p_a]["mobject"].animate.move_to(RIGHT * target_x_a + UP * y_pos),
                parcels[p_b]["mobject"].animate.move_to(RIGHT * target_x_b + UP * y_pos),
                run_time=0.6
            )

            parcel_positions[p_a], parcel_positions[p_b] = pos_b_idx, pos_a_idx
            current_time += 0.6

            self.swap_events.append({
                "swap_index": s + 1,
                "timestamp": round(current_time, 2),
                "swapped_parcels": [parcels[p_a]["color_name"], parcels[p_b]["color_name"]],
                "positions": list(parcel_positions)
            })

        # Ensure 50% positive (marked parcel 0) and 50% negative (unmarked parcel 1 or 2)
        # across control parameter values D even for a single seed (like seed 0).
        sample_hash = int(hashlib.md5(f"{self.seed}_{self.D}".encode()).hexdigest(), 16)
        should_marked_enter = (sample_hash % 2 == 0)

        if should_marked_enter:
            entering_parcel_idx = 0
        else:
            entering_parcel_idx = 1 if (sample_hash // 2) % 2 == 0 else 2

        self.final_parcel_in_bin_id = entering_parcel_idx
        self.entered_delivery_bin = (entering_parcel_idx == self.marked_parcel_id)

        winning_mobject = parcels[entering_parcel_idx]["mobject"]
        self.play(winning_mobject.animate.move_to(RIGHT * 3.0 + DOWN * 2.0), run_time=0.8)
        current_time += 0.8

        # Calculate remaining wait time to enforce FIXED_TASK_DURATION (17.0s)
        question_duration = 3.7
        remaining_wait = max(0.2, FIXED_TASK_DURATION - current_time - question_duration)
        self.wait(remaining_wait)
        current_time += remaining_wait

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Did the parcel marked at the beginning enter the delivery bin?",
            format_instruction="Answer with 'yes' or 'no'."
        )
        current_time += question_duration
        self.actual_duration = round(current_time, 2)

class LongTermDependencyGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "long_term_dependency"

    @property
    def control_parameter_name(self) -> str:
        return "D"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        D = int(control_value)
        sample_id = f"dependency_D{D}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            LongTermDependencyScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            D=D,
            seed=seed
        )

        question = "Did the parcel marked at the beginning enter the delivery bin?"
        exact_answer = "yes" if scene.entered_delivery_bin else "no"

        trace_data = {
            "steps": [
                {
                    "state": {"marked_parcel_id": scene.marked_parcel_id, "positions": e["positions"]},
                    "event": {"type": "parcel_swap", "timestamp": e["timestamp"], "swapped": e["swapped_parcels"]},
                    "operation": {"action": "track_swap", "swap_index": e["swap_index"]}
                } for e in scene.swap_events
            ],
            "num_intervening_swaps": D,
            "marked_parcel_color": COLOR_NAMES[scene.marked_parcel_id],
            "final_bin_parcel_id": scene.final_parcel_in_bin_id,
            "entered_delivery_bin": scene.entered_delivery_bin,
            "events": scene.swap_events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Initial marking on parcel '{COLOR_NAMES[0]}', followed by D={D} position swaps and delivery bin entry.",
            "",
            "### Step 1: Track Intervening Swaps"
        ]
        for e in scene.swap_events:
            cot_lines.append(f"- Swap {e['swap_index']} at {e['timestamp']:.2f}s: swapped {e['swapped_parcels'][0]} and {e['swapped_parcels'][1]}")

        cot_lines.extend([
            "",
            "### Step 2: Track Final Bin Parcel Identity",
            f"Parcel entering delivery bin ID: {scene.final_parcel_in_bin_id} ({COLOR_NAMES[scene.final_parcel_in_bin_id]}).",
            "",
            "### Step 3: Match Marked Parcel",
            f"Did marked parcel enter bin? {exact_answer.upper()}.",
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
            "control_value": D,
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
            control_parameter_value=D,
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
