import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Rectangle, RIGHT, LEFT, UP, DOWN, GREY, YELLOW

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green", "purple", "orange", "cyan"]

class CausalAttributionScene(Scene):
    def __init__(self, C: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.C = int(C)
        self.seed = seed
        self.causal_graph = {"nodes": [], "edges": []}
        self.correct_cause_color = ""
        self.events = []

    def construct(self):
        set_seed(self.seed)
        num_chains = 3
        colors = get_nuisance_colors(self.seed, num_chains + 1)

        rng = random.Random(self.seed)
        true_cause_idx = rng.randint(0, num_chains - 1)
        self.correct_cause_color = COLOR_NAMES[true_cause_idx % len(COLOR_NAMES)]

        lamp = Circle(radius=0.5, color=GREY, fill_opacity=0.5).move_to(RIGHT * 5.0)
        self.add(lamp)

        y_positions = [2.0, 0.0, -2.0]
        chain_objects = []

        for idx in range(num_chains):
            col_hex = colors[idx]
            col_name = COLOR_NAMES[idx % len(COLOR_NAMES)]
            y = y_positions[idx]

            init_obj = Square(side_length=0.6, color=col_hex, fill_opacity=1).move_to(LEFT * 5.0 + UP * y)
            self.add(init_obj)

            depth = self.C if idx == true_cause_idx else max(1, self.C - 1)
            chain_objects.append({
                "idx": idx,
                "color_name": col_name,
                "is_true_cause": (idx == true_cause_idx),
                "init_obj": init_obj,
                "depth": depth,
                "y": y
            })

            self.causal_graph["nodes"].append(f"object_{col_name}")

        current_time = 0.5

        for item in chain_objects:
            col_name = item["color_name"]
            y = item["y"]
            depth = item["depth"]
            is_true = item["is_true_cause"]

            self.play(item["init_obj"].animate.shift(RIGHT * 1.5), run_time=0.6)
            current_time += 0.6
            self.events.append({
                "type": "initiation",
                "object": col_name,
                "timestamp": round(current_time, 2)
            })

            curr_x = -3.5 + 1.5
            prev_node = f"object_{col_name}"

            for d in range(depth):
                next_node = f"step_{col_name}_{d+1}"
                self.causal_graph["nodes"].append(next_node)
                self.causal_graph["edges"].append([prev_node, next_node])

                block = Rectangle(width=0.3, height=0.6, color="#FFFFFF", fill_opacity=1).move_to(RIGHT * (curr_x + 0.8) + UP * y)
                self.add(block)
                self.play(block.animate.scale(1.2).set_color(YELLOW), run_time=0.4)
                current_time += 0.4

                self.events.append({
                    "type": "intermediate_activation",
                    "step": d + 1,
                    "object": col_name,
                    "timestamp": round(current_time, 2)
                })
                curr_x += 0.8
                prev_node = next_node

            if is_true:
                self.causal_graph["edges"].append([prev_node, "lamp"])
                self.play(lamp.animate.set_color(YELLOW).set_fill(YELLOW, opacity=1.0), run_time=0.5)
                current_time += 0.5
                self.events.append({
                    "type": "lamp_activation",
                    "object": col_name,
                    "timestamp": round(current_time, 2)
                })
            else:
                self.wait(0.3)
                current_time += 0.3

        self.wait(0.5)

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Which colored object caused the lamp to turn on?",
            format_instruction="Answer with the color name (e.g. red)."
        )

class CausalAttributionGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "causal_attribution"

    @property
    def control_parameter_name(self) -> str:
        return "C"

    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        C = int(control_value)
        sample_id = f"causal_C{C}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            CausalAttributionScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            C=C,
            seed=seed
        )

        question = "Which colored object caused the lamp to turn on?"
        exact_answer = scene.correct_cause_color

        trace_data = {
            "steps": [
                {
                    "state": {"active_chain": e["object"]},
                    "event": {"type": e["type"], "object": e["object"], "timestamp": e["timestamp"]},
                    "operation": {"action": "evaluate_causal_propagation"}
                } for e in scene.events
            ],
            "causal_depth": C,
            "causal_graph": scene.causal_graph,
            "root_cause_object": exact_answer,
            "events": scene.events
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Parallel scripted causal chains (depth C={C}) activating toward lamp.",
            "",
            "### Step 1: Trace Causal Graph Chain Activations"
        ]
        for e in scene.events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: [{e['type']}] by {e['object']} object")

        cot_lines.extend([
            "",
            "### Step 2: Identify Root Cause",
            f"The chain initiated by the {exact_answer} object successfully triggered the lamp activation event.",
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
            "control_value": C,
            "seed": seed
        }

        dest_video, dest_trace, dest_cot, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, cot_text, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(1.0 + 3 * (0.6 + C * 0.4) + 3.7, 2)

        shutil.rmtree(temp_dir, ignore_errors=True)

        return SyntheticSample(
            sample_id=sample_id,
            task_name=self.task_name,
            control_parameter_name=self.control_parameter_name,
            control_parameter_value=C,
            seed=seed,
            video_path=dest_video,
            question=question,
            exact_answer=exact_answer,
            executable_trace=trace_data,
            cot_text=cot_text,
            generation_config={"resolution": resolution, "fps": fps},
            duration=duration,
            fps=fps,
            resolution=resolution,
            checksum=checksum
        )
