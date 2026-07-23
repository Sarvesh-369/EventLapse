#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.generation.causal_attribution import CausalAttributionGenerator
from eventlapse.utils.logging import logger

VALIDATION_CASES = [
    (1, 0),
    (2, 0),
    (3, 0),
    (5, 0),
    (6, 0),
    (5, 1),
    (5, 2)
]

def main():
    generator = CausalAttributionGenerator()
    out_dir = Path(__file__).resolve().parent.parent / "data"
    frames_dir = out_dir / "validation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []

    print("=" * 80)
    print("EVENTLAPSE CAUSAL ATTRIBUTION VALIDATION SUITE REPORT")
    print("=" * 80)

    for C, seed in VALIDATION_CASES:
        sample = generator.generate_sample(control_value=C, seed=seed, output_dir=out_dir)
        trace = sample.executable_trace

        # Extract frames at key timestamps with ffmpeg
        video_path = sample.video_path
        sample_id = sample.sample_id

        # Extract 3 key timestamps: initiation (t=0.5s), mid-propagation, and outcome
        timestamps = [0.5, round(sample.duration / 2.0, 1), round(max(0.5, sample.duration - 3.8), 1)]
        extracted_frames = []

        for idx, ts in enumerate(timestamps):
            frame_png = frames_dir / f"{sample_id}_t{ts:.1f}s.png"
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ss", f"{ts:.3f}", "-vframes", "1", str(frame_png)]
            subprocess.run(cmd, capture_output=True, check=False)
            if frame_png.exists():
                extracted_frames.append(frame_png)

        num_intermediate_comps = C - 1
        num_transitions = C
        true_cause = trace["true_cause_color"]
        trial_order = [t["initiator_color"] for t in trace["trials"]]
        succ_pos = trace["successful_trial_index"]
        rendered_dur = sample.duration

        info = {
            "sample_id": sample_id,
            "C": C,
            "seed": seed,
            "num_intermediate_components": num_intermediate_comps,
            "num_transitions_per_trial": num_transitions,
            "true_cause_color": true_cause,
            "candidate_trial_order": trial_order,
            "successful_trial_position": f"Trial {succ_pos + 1} ({true_cause})",
            "actual_rendered_duration": f"{rendered_dur:.2f}s"
        }
        report_lines.append(info)

        print(f"\n--- [SAMPLE: {sample_id}] ---")
        print(f"C (Transitions): {C} | Seed: {seed}")
        print(f"Intermediate Components: {num_intermediate_comps}")
        print(f"Candidate Trial Order: {trial_order}")
        print(f"Successful Trial Position: Trial {succ_pos + 1} ({true_cause})")
        print(f"Actual Rendered Duration: {rendered_dur:.2f}s")
        print("EXECUTABLE TRACE JSON:")
        print(json.dumps(trace, indent=2))
        print("-" * 60)

    print("\n" + "=" * 80)
    print("SUMMARY METRICS TABLE")
    print("=" * 80)
    print(f"{'Sample ID':<22} | {'C':<3} | {'Comp':<4} | {'Trans':<5} | {'True Cause':<10} | {'Trial Order':<20} | {'Succ Pos':<10} | {'Duration':<8}")
    print("-" * 105)
    for info in report_lines:
        order_str = "->".join(info['candidate_trial_order'])
        print(f"{info['sample_id']:<22} | {info['C']:<3} | {info['num_intermediate_components']:<4} | {info['num_transitions_per_trial']:<5} | {info['true_cause_color']:<10} | {order_str:<20} | {info['successful_trial_position']:<10} | {info['actual_rendered_duration']:<8}")
    print("=" * 80)

if __name__ == "__main__":
    main()
