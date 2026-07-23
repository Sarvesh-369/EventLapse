import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Rectangle, Star, RIGHT, UP, DOWN, YELLOW, GREEN

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green"]

class LongTermDependencyScene(Scene):
    def __init__(self, D: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.D = int(D)
        self.seed = seed
        self.swap_events = []
        self.marked_parcel_id = 0
        self.final_parcel_in_bin_id = -1
        self.entered_delivery_bin = False

    def construct(self):
        set_seed(self.seed)
        colors = get_nuisance_colors(self.seed, 3)

        rng = random.Random(self.seed)
        self.marked_parcel_id = 0

        x_positions = [-3.0, 0.0, 3.0]
        y_pos = 1.0

        parcels = []
        parcel_positions = [0, 1, 2] # current slot index for parcel 0, 1, 2

        for i in range(3):
            c_hex = colors[i]
            c_name = COLOR_NAMES[i % len(COLOR_NAMES)]
            shape = Circle(radius=0.4, color=c_hex, fill_opacity=1).move_to(RIGHT * x_positions[i] + UP * y_pos)
            parcels.append({"id": i, "color_name": c_name, "hex": c_hex, "shape": shape})
            self.add(shape)

        # Mark parcel 0 with a yellow star
        marker = Star(n=5, outer_radius=0.2, color=YELLOW, fill_opacity=1).move_to(parcels[0]["shape"].get_center() + UP * 0.6)
        self.add(marker)

        self.play(marker.animate.scale(1.2), run_time=0.3)
        self.play(marker.animate.scale(1/1.2), run_time=0.3)

        # Delivery bin at bottom right
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
                parcels[p_a]["shape"].animate.move_to(RIGHT * target_x_a + UP * y_pos),
                parcels[p_b]["shape"].animate.move_to(RIGHT * target_x_b + UP * y_pos),
                run_time=0.6
            )
            if p_a == 0:
                marker.move_to(RIGHT * target_x_a + UP * (y_pos + 0.6))
            elif p_b == 0:
                marker.move_to(RIGHT * target_x_b + UP * (y_pos + 0.6))

            parcel_positions[p_a], parcel_positions[p_b] = pos_b_idx, pos_a_idx
            current_time += 0.6

            self.swap_events.append({
                "swap_index": s + 1,
                "timestamp": round(current_time, 2),
                "swapped_parcels": [parcels[p_a]["color_name"], parcels[p_b]["color_name"]],
                "positions": list(parcel_positions)
            })

        # 50% positive cases: marked parcel (0) enters bin
        # 50% negative cases: unmarked parcel (1 or 2) enters bin
        should_marked_enter = (self.seed % 2 == 0)

        if should_marked_enter:
            entering_parcel_idx = 0
        else:
            # Deterministically select parcel 1 or 2
            entering_parcel_idx = 1 if (self.seed // 2) % 2 == 0 else 2

        self.final_parcel_in_bin_id = entering_parcel_idx
        self.entered_delivery_bin = (entering_parcel_idx == self.marked_parcel_id)

        winning_shape = parcels[entering_parcel_idx]["shape"]
        self.play(winning_shape.animate.move_to(RIGHT * 3.0 + DOWN * 2.0), run_time=0.8)
        if entering_parcel_idx == 0:
            self.play(marker.animate.move_to(RIGHT * 3.0 + DOWN * 1.4), run_time=0.8)

        current_time += 0.8
        self.wait(0.5)

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Did the parcel marked at the beginning enter the delivery bin?",
            format_instruction="Answer with 'yes' or 'no'."
        )

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

        gt_data = {
            "sample_id": sample_id,
            "question": question,
            "exact_answer": exact_answer,
            "task_name": self.task_name,
            "control_parameter": self.control_parameter_name,
            "control_value": D,
            "seed": seed
        }

        dest_video, dest_trace, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(1.4 + D * 0.6 + 5.0, 2)

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
            generation_config={"resolution": resolution, "fps": fps},
            duration=duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
