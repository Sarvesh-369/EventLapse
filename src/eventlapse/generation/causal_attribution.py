import random
import hashlib
import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Line, Text, RIGHT, LEFT, UP, DOWN, GREY, YELLOW, RED

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed

COLOR_NAMES = ["red", "blue", "green"]
HEX_MAP = {
    "red": "#FF4444",
    "blue": "#4444FF",
    "green": "#44FF44"
}

class CausalAttributionScene(Scene):
    def __init__(self, C: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.C = int(C)
        self.seed = seed
        self.true_cause_color = ""
        self.successful_trial_index = -1
        self.trials_trace = []
        self.actual_duration = 0.0

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"{self.seed}_{self.C}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        candidate_colors = list(COLOR_NAMES)
        trial_order = [0, 1, 2]
        rng.shuffle(trial_order)

        self.successful_trial_index = rng.randint(0, 2)
        self.true_cause_color = candidate_colors[trial_order[self.successful_trial_index]]

        y_homes = [2.0, 0.0, -2.0]
        ball_objs = []
        for i in range(3):
            c_name = candidate_colors[i]
            b = Circle(radius=0.35, color=HEX_MAP[c_name], fill_opacity=1.0).move_to(LEFT * 6.0 + UP * y_homes[i])
            ball_objs.append(b)
            self.add(b)

        # Shared neutral causal chain
        trigger_x = -4.0
        trigger = Circle(radius=0.3, color=GREY, fill_opacity=0.5).move_to(LEFT * 4.0)
        self.add(trigger)

        num_inter_comps = self.C - 1
        comp_nodes = []
        comp_lines = []

        prev_mobj = trigger
        step_x = 8.5 / (num_inter_comps + 1) if num_inter_comps > 0 else 8.5

        for idx in range(num_inter_comps):
            cx = trigger_x + (idx + 1) * step_x
            comp = Square(side_length=0.4, color=GREY, fill_opacity=0.5).move_to(RIGHT * cx)
            line = Line(prev_mobj.get_right(), comp.get_left(), color=GREY, stroke_width=3)
            self.add(line, comp)
            comp_nodes.append(comp)
            comp_lines.append(line)
            prev_mobj = comp

        lamp = Circle(radius=0.5, color=GREY, fill_opacity=0.3).move_to(RIGHT * 5.5)
        lamp_label = Text("LAMP", font_size=18).move_to(RIGHT * 5.5)
        final_line = Line(prev_mobj.get_right(), lamp.get_left(), color=GREY, stroke_width=3)
        self.add(final_line, lamp, lamp_label)
        comp_lines.append(final_line)

        self.wait(0.3)
        current_time = 0.3

        for trial_idx in range(3):
            cand_idx = trial_order[trial_idx]
            cand_color = candidate_colors[cand_idx]
            cand_ball = ball_objs[cand_idx]
            home_pos = LEFT * 6.0 + UP * y_homes[cand_idx]
            is_succ = (trial_idx == self.successful_trial_index)

            # Step 1: Ball touches trigger
            target_y = 0.6 if y_homes[cand_idx] > 0 else (-0.6 if y_homes[cand_idx] < 0 else 0.0)
            self.play(cand_ball.animate.move_to(LEFT * 4.0 + UP * target_y), run_time=0.4)
            current_time += 0.4

            t_touch = round(current_time, 2)
            self.play(
                cand_ball.animate.move_to(home_pos),
                trigger.animate.set_color(YELLOW).set_fill(YELLOW, opacity=0.8),
                run_time=0.4
            )
            current_time += 0.4

            events = []

            if self.C == 1:
                # Direct transition to outcome (exactly 1 transition)
                if is_succ:
                    self.play(
                        final_line.animate.set_color(YELLOW).set_stroke(width=5),
                        lamp.animate.set_color(YELLOW).set_fill(YELLOW, opacity=1.0),
                        run_time=0.5
                    )
                    current_time += 0.5
                    events.append({
                        "transition_index": 1,
                        "timestamp": round(current_time, 2),
                        "cause": f"{cand_color}_ball_contacts_trigger",
                        "effect": "lamp_turns_on"
                    })
                else:
                    self.play(
                        final_line.animate.set_color(RED).set_stroke(width=5),
                        run_time=0.5
                    )
                    current_time += 0.5
                    events.append({
                        "transition_index": 1,
                        "timestamp": round(current_time, 2),
                        "cause": f"{cand_color}_ball_contacts_trigger",
                        "effect": "blocked_at_lamp"
                    })
            else:
                # Transition 1: Trigger -> Comp 1
                self.play(
                    comp_lines[0].animate.set_color(YELLOW).set_stroke(width=5),
                    comp_nodes[0].animate.set_color(YELLOW).set_fill(YELLOW, opacity=0.8),
                    run_time=0.4
                )
                current_time += 0.4
                events.append({
                    "transition_index": 1,
                    "timestamp": round(current_time, 2),
                    "cause": f"{cand_color}_ball_contacts_trigger",
                    "effect": "component_1_activates"
                })

                # Transitions 2 .. C-1: Comp s -> Comp s+1
                for step_idx in range(1, num_inter_comps):
                    self.play(
                        comp_lines[step_idx].animate.set_color(YELLOW).set_stroke(width=5),
                        comp_nodes[step_idx].animate.set_color(YELLOW).set_fill(YELLOW, opacity=0.8),
                        run_time=0.4
                    )
                    current_time += 0.4
                    events.append({
                        "transition_index": step_idx + 1,
                        "timestamp": round(current_time, 2),
                        "cause": f"component_{step_idx}_activates",
                        "effect": f"component_{step_idx+1}_activates"
                    })

                # Transition C: Final Comp -> Lamp
                if is_succ:
                    self.play(
                        final_line.animate.set_color(YELLOW).set_stroke(width=5),
                        lamp.animate.set_color(YELLOW).set_fill(YELLOW, opacity=1.0),
                        run_time=0.5
                    )
                    current_time += 0.5
                    events.append({
                        "transition_index": self.C,
                        "timestamp": round(current_time, 2),
                        "cause": f"component_{num_inter_comps}_activates",
                        "effect": "lamp_turns_on"
                    })
                else:
                    self.play(
                        final_line.animate.set_color(RED).set_stroke(width=5),
                        run_time=0.5
                    )
                    current_time += 0.5
                    events.append({
                        "transition_index": self.C,
                        "timestamp": round(current_time, 2),
                        "cause": f"component_{num_inter_comps}_activates",
                        "effect": "blocked_at_lamp"
                    })

            self.wait(0.4)
            current_time += 0.4

            # Reset chain before next trial
            reset_anims = [
                trigger.animate.set_color(GREY).set_fill(GREY, opacity=0.5),
                lamp.animate.set_color(GREY).set_fill(GREY, opacity=0.3)
            ]
            for node in comp_nodes:
                reset_anims.append(node.animate.set_color(GREY).set_fill(GREY, opacity=0.5))
            for line in comp_lines:
                reset_anims.append(line.animate.set_color(GREY).set_stroke(width=3))

            self.play(*reset_anims, run_time=0.3)
            current_time += 0.3

            self.trials_trace.append({
                "trial_index": trial_idx,
                "initiator_color": cand_color,
                "is_successful": is_succ,
                "events": events,
                "outcome": "lamp_activated" if is_succ else "blocked",
                "lamp_activated": is_succ
            })

        self.wait(0.5)
        current_time += 0.5

        # Render question text overlay at the end of the video
        render_question_card(
            self,
            question="Which colored object caused the lamp to turn on?",
            format_instruction="Answer with the color name (e.g. red)."
        )
        current_time += 3.7
        self.actual_duration = round(current_time, 2)

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
        exact_answer = scene.true_cause_color

        # Build steps for execution trace compatibility
        flat_steps = []
        for tr in scene.trials_trace:
            for ev in tr["events"]:
                flat_steps.append({
                    "state": {"trial_index": tr["trial_index"], "initiator_color": tr["initiator_color"], "active_step": ev["transition_index"]},
                    "event": {"type": "causal_transition", "cause": ev["cause"], "effect": ev["effect"], "timestamp": ev["timestamp"]},
                    "operation": {"action": "evaluate_trial_step", "trial_index": tr["trial_index"]}
                })

        trace_data = {
            "steps": flat_steps,
            "causal_depth": C,
            "true_cause_color": exact_answer,
            "successful_trial_index": scene.successful_trial_index,
            "trials": scene.trials_trace,
            "answer": exact_answer
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let's analyze the video step by step.",
            "",
            "### Scene Description",
            f"Three candidate colored balls (red, blue, green) testing a shared causal chain (depth C={C}) sequentially.",
            "",
            "### Step 1: Track Candidate Trials"
        ]
        for tr in scene.trials_trace:
            cot_lines.append(f"- Trial {tr['trial_index']+1} ({tr['initiator_color']} ball): {len(tr['events'])} transitions -> Outcome: {tr['outcome']}")

        cot_lines.extend([
            "",
            "### Step 2: Identify Successful Initiator",
            f"Only Trial {scene.successful_trial_index+1} ({exact_answer} ball) activated the lamp.",
            "",
            "### Step 3: Derive Root Cause",
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

        # Get actual rendered video duration via ffprobe or scene.actual_duration
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
            control_parameter_value=C,
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
