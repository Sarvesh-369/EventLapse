import random
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from manim import Scene, Circle, Square, Triangle, Star, Text, UP, DOWN, LEFT, RIGHT, Transform

from eventlapse.generation.base import BaseTaskGenerator, SyntheticSample
from eventlapse.generation.renderer import render_manim_scene, save_sample_outputs, render_question_card
from eventlapse.utils.caching import compute_file_checksum
from eventlapse.utils.seeds import set_seed, get_nuisance_colors

FIXED_TASK_DURATION = 24.0

class StateMachineScene(Scene):
    def __init__(self, N: int, F: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.F = float(F)
        self.seed = seed
        self.transition_events = []
        self.actual_duration = FIXED_TASK_DURATION

    def construct(self):
        set_seed(self.seed)
        sample_hash = int(hashlib.md5(f"eventlapse_state_{self.seed}_{self.N}_{self.F}".encode()).hexdigest(), 16)
        rng = random.Random(sample_hash)

        colors = get_nuisance_colors(self.seed, 4)

        states = [
            {"id": "A", "shape": Circle(radius=0.9, color=colors[0], fill_opacity=0.8), "pos": LEFT * 2.5 + UP * 1.5},
            {"id": "B", "shape": Square(side_length=1.8, color=colors[1], fill_opacity=0.8), "pos": RIGHT * 2.5 + UP * 1.5},
            {"id": "C", "shape": Star(n=5, outer_radius=1.0, color=colors[2], fill_opacity=0.8), "pos": RIGHT * 2.5 + DOWN * 1.5},
            {"id": "D", "shape": Triangle(color=colors[3], fill_opacity=0.8).scale(1.3), "pos": LEFT * 2.5 + DOWN * 1.5},
        ]

        current_idx = rng.randint(0, 3)
        curr_state = states[current_idx]

        active_obj = curr_state["shape"].copy().move_to(curr_state["pos"])
        label = Text(f"State {curr_state['id']}", font_size=32).next_to(active_obj, UP)

        self.add(active_obj, label)

        transition_duration = min(0.4, 0.8 / self.F)
        dwell_time = max(0.05, (1.0 / self.F) - transition_duration)

        if self.N == 0:
            est_active_time = 0.0
        else:
            est_active_time = self.N * transition_duration + max(0, self.N - 1) * dwell_time

        total_idle = max(0.4, FIXED_TASK_DURATION - est_active_time)
        pre_wait = rng.uniform(0.3, max(0.3, total_idle - 0.3))

        self.wait(pre_wait)
        current_time = pre_wait

        for i in range(self.N):
            next_idx = (current_idx + rng.choice([1, 2, 3])) % 4
            next_state = states[next_idx]

            new_obj = next_state["shape"].copy().move_to(next_state["pos"])
            new_label = Text(f"State {next_state['id']}", font_size=32).next_to(new_obj, UP)

            self.play(
                Transform(active_obj, new_obj),
                Transform(label, new_label),
                run_time=transition_duration
            )
            current_time += transition_duration

            self.transition_events.append({
                "transition_index": i + 1,
                "timestamp": round(current_time, 2),
                "from_state": curr_state["id"],
                "to_state": next_state["id"],
                "running_count": i + 1
            })

            current_idx = next_idx
            curr_state = next_state

            if i < self.N - 1:
                self.wait(dwell_time)
                current_time += dwell_time

        post_wait = max(0.1, FIXED_TASK_DURATION - current_time)
        self.wait(post_wait)
        current_time += post_wait
        self.actual_duration = round(current_time, 2)

class StateMachineGenerator(BaseTaskGenerator):
    @property
    def task_name(self) -> str:
        return "state_machine"

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
        sample_id = f"state_N{N}_F{F}_seed{seed}"

        rendered_file, scene, temp_dir = render_manim_scene(
            StateMachineScene,
            output_filename=sample_id,
            resolution=resolution,
            fps=fps,
            N=N,
            F=F,
            seed=seed
        )

        question = "How many state transitions occurred in the video?"
        exact_answer = str(N)

        trace_data = {
            "steps": [
                {
                    "state": {"current_state": e["to_state"], "running_count": e["running_count"]},
                    "event": {"type": "state_transition", "timestamp": e["timestamp"], "from": e["from_state"], "to": e["to_state"]},
                    "operation": {"action": "increment_counter", "count": e["running_count"]}
                } for e in scene.transition_events
            ],
            "final_count": N,
            "events": scene.transition_events,
            "frequency_hz": F
        }

        cot_lines = [
            f"**Question:** {question} Show your reasoning and put the final answer in \\boxed{{}}",
            "",
            "Let me analyze the video step by step.",
            "",
            "### Scene Description",
            f"Visual state machine transitions at frequency {F} Hz."
        ]
        for e in scene.transition_events:
            cot_lines.append(f"- At {e['timestamp']:.2f}s: Transitioned from State {e['from_state']} to State {e['to_state']} (count={e['running_count']})")

        cot_lines.extend([
            "",
            "### Step 1: Track State Transitions",
            f"Total state transitions detected: {N}.",
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
