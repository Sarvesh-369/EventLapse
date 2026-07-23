import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Line, Text, RIGHT, LEFT, UP, DOWN, GREY, YELLOW

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

COLOR_NAMES = ["red", "blue", "green", "purple", "orange", "cyan"]
HEX_MAP = {
    "red": "#FF4444",
    "blue": "#4444FF",
    "green": "#44FF44",
    "purple": "#AA44FF",
    "orange": "#FF8844",
    "cyan": "#44FFFF"
}

class CausalAttributionScene(Scene):
    def __init__(self, C: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.C = int(C)
        self.seed = seed
        self.correct_cause_color = ""
        self.causal_events = []
        self.chain_metadata = []

    def construct(self):
        set_seed(self.seed)
        rng = random.Random(self.seed)
        num_chains = 3

        available_colors = list(COLOR_NAMES)
        rng.shuffle(available_colors)
        colors = available_colors[:num_chains]

        true_cause_row = rng.randint(0, num_chains - 1)
        self.correct_cause_color = colors[true_cause_row]

        y_positions = [2.0, 0.0, -2.0]

        lamp = Circle(radius=0.5, color=GREY, fill_opacity=0.3).move_to(RIGHT * 5.0)
        lamp_label = Text("LAMP", font_size=18).move_to(RIGHT * 5.0)
        self.add(lamp, lamp_label)

        chain_elements = []

        step_x = 9.0 / (self.C + 1)

        for row in range(num_chains):
            y = y_positions[row]
            c_name = colors[row]
            c_hex = HEX_MAP.get(c_name, "#FFFFFF")

            ball = Circle(radius=0.35, color=c_hex, fill_opacity=1.0).move_to(LEFT * 5.5 + UP * y)
            self.add(ball)

            blocks = []
            lines = []
            num_blocks = self.C if row == true_cause_row else rng.randint(1, max(1, self.C - 1))

            prev_pos = LEFT * 5.5 + UP * y
            for b_idx in range(num_blocks):
                bx = -5.5 + (b_idx + 1) * step_x
                block = Square(side_length=0.4, color=GREY, fill_opacity=0.5).move_to(RIGHT * bx + UP * y)
                line = Line(prev_pos, RIGHT * bx + UP * y, color=GREY, stroke_width=2)
                self.add(line, block)
                blocks.append(block)
                lines.append(line)
                prev_pos = RIGHT * bx + UP * y

            if row == true_cause_row:
                final_line = Line(prev_pos, RIGHT * 4.5 + UP * y, color=GREY, stroke_width=2)
                self.add(final_line)
                lines.append(final_line)

            chain_elements.append({
                "row": row,
                "color": c_name,
                "is_true": (row == true_cause_row),
                "ball": ball,
                "blocks": blocks,
                "lines": lines
            })

        self.wait(0.5)

        for step in range(self.C):
            anims = []
            for ch in chain_elements:
                if step < len(ch["blocks"]):
                    anims.append(ch["blocks"][step].animate.set_color(YELLOW).set_fill(YELLOW, opacity=0.8))
                    anims.append(ch["lines"][step].animate.set_color(YELLOW).set_stroke(width=4))

                    if ch["is_true"]:
                        cause_desc = f"{ch['color']}_ball_initiates_step_{step+1}" if step == 0 else f"step_{step}_activates_step_{step+1}"
                        effect_desc = f"step_{step+1}_activated"
                        self.causal_events.append({
                            "step": step + 1,
                            "cause": cause_desc,
                            "effect": effect_desc
                        })

            if anims:
                self.play(*anims, run_time=0.6)

        true_ch = [ch for ch in chain_elements if ch["is_true"]][0]
        self.play(
            true_ch["lines"][-1].animate.set_color(YELLOW).set_stroke(width=4),
            lamp.animate.set_color(YELLOW).set_fill(YELLOW, opacity=1.0),
            run_time=0.6
        )
        self.causal_events.append({
            "step": self.C + 1,
            "cause": f"step_{self.C}_activates_lamp",
            "effect": "lamp_turns_on"
        })

        self.wait(1.0)

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
                    "state": {"active_step": ev["step"]},
                    "event": {"type": "causal_propagation", "cause": ev["cause"], "effect": ev["effect"]},
                    "operation": {"action": "trace_causal_chain", "step": ev["step"]}
                } for ev in scene.causal_events
            ],
            "root_cause": exact_answer,
            "causal_events": scene.causal_events,
            "causal_depth": C,
            "answer": exact_answer
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Three parallel colored cause-and-effect chains (depth C={C}) terminating at a lamp.",
            "",
            "### Step 1: Trace Parallel Causal Chains"
        ]
        for ev in scene.causal_events:
            cot_lines.append(f"- Step {ev['step']}: [{ev['cause']}] -> [{ev['effect']}]")

        cot_lines.extend([
            "",
            "### Step 2: Evaluate Lamp Activation",
            f"Only the chain initiated by the {exact_answer} ball reached and activated the lamp.",
            "",
            "### Step 3: Identify Root Cause",
            f"Root cause object: {exact_answer}",
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

        dest_video, dest_question, dest_trace, dest_cot, dest_gt = save_sample_outputs(
            sample_id, self.task_name, rendered_file, trace_data, cot_text, gt_data, output_dir
        )
        checksum = compute_file_checksum(dest_video)
        duration = round(0.5 + C * 0.6 + 2.2 + 3.7, 2)

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
