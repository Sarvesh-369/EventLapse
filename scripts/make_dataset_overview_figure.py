#!/usr/bin/env python3
import json
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, BoxStyle
from PIL import Image
from pathlib import Path

def extract_frame_at_timestamp(video_path: Path, timestamp: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(tmp_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        img = Image.open(tmp_path).convert("RGB")
        arr = np.array(img)
        tmp_path.unlink(missing_ok=True)
        return arr
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

def build_dataset_overview_figure(output_path: Path):
    demo_dir = Path("data/figure_demo")

    tasks_info = [
        {
            "domain": "Bounce Ball",
            "question": "Question: How many times did the ball contact the walls?",
            "video_path": demo_dir / "videos/bounce_ball/bounce_N2_F1.0_seed0.mp4",
            "sample_ts": [5.50, 6.49, 7.00, 7.49, 8.00, 8.50],
            "event_indices": [1, 3],
            "color": "#1f77b4",
            "cot_lines": [
                "Scene Description:",
                "Ball bouncing between walls at 1.0 Hz.",
                "• At 6.49s: Contact wall_negative (c=1)",
                "• At 7.49s: Contact wall_positive (c=2)",
                "",
                "Step 1: Track Contact Events",
                "Total wall contact events detected: 2.",
                "",
                "Final Answer: 2"
            ]
        },
        {
            "domain": "Blinking",
            "question": "Question: How many times did the object blink?",
            "video_path": demo_dir / "videos/blinking/blinking_N2_F1.0_seed0.mp4",
            "sample_ts": [5.70, 6.72, 7.20, 7.72, 8.20, 8.70],
            "event_indices": [1, 3],
            "color": "#d95f02",
            "cot_lines": [
                "Scene Description:",
                "An object pulsing ON and OFF at 1.0 Hz.",
                "• At 6.72s: Object blinked ON (c=1)",
                "• At 7.72s: Object blinked ON (c=2)",
                "",
                "Step 1: Track Blink Pulses",
                "Total blinks detected: 2.",
                "",
                "Final Answer: 2"
            ]
        },
        {
            "domain": "State Machine",
            "question": "Question: How many state transitions occurred in the video?",
            "video_path": demo_dir / "videos/state_machine/state_N2_F1.0_seed1.mp4",
            "sample_ts": [2.50, 3.09, 3.60, 4.09, 4.60, 5.10],
            "event_indices": [1, 3],
            "color": "#2ca02c",
            "cot_lines": [
                "Scene Description:",
                "Visual state transitions at 1.0 Hz.",
                "• At 3.09s: Transition State D ➔ State C (c=1)",
                "• At 4.09s: Transition State C ➔ State A (c=2)",
                "",
                "Step 1: Track State Transitions",
                "Total state transitions detected: 2.",
                "",
                "Final Answer: 2"
            ]
        }
    ]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig = plt.figure(figsize=(18, 9.8), dpi=300, facecolor="white")
    col_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

    for c_idx, tinfo in enumerate(tasks_info):
        col_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=col_gs[c_idx],
                                                   height_ratios=[0.75, 5.4, 3.6], hspace=0.015)

        # --- 1. Header Box ---
        ax_header = fig.add_subplot(col_sub[0])
        ax_header.axis("off")

        header_patch = FancyBboxPatch((0.005, 0.02), 0.99, 0.96, transform=ax_header.transAxes,
                                      facecolor="#ffffff", edgecolor=tinfo["color"], linewidth=2.2,
                                      boxstyle=BoxStyle("Round", pad=0.01, rounding_size=0.04),
                                      clip_on=False)
        ax_header.add_patch(header_patch)

        ax_header.text(0.5, 0.68, tinfo["domain"], transform=ax_header.transAxes,
                       fontsize=17, fontweight="bold", color=tinfo["color"], ha="center", va="center")
        ax_header.text(0.5, 0.28, tinfo["question"], transform=ax_header.transAxes,
                       fontsize=11.5, fontstyle="italic", color="#111111", ha="center", va="center")

        # --- 2. 6 Frames in Original 16:9 Aspect Ratio ---
        extracted = [(ts, extract_frame_at_timestamp(tinfo["video_path"], ts)) for ts in tinfo["sample_ts"]]

        ax_frames_container = fig.add_subplot(col_sub[1])
        ax_frames_container.axis("off")

        frame_rows_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=col_sub[1], hspace=0.01, wspace=0.01)

        for k, (ts, img) in enumerate(extracted):
            r_idx = k // 2
            c_idx_sub = k % 2

            ax_f = fig.add_subplot(frame_rows_gs[r_idx, c_idx_sub])
            ax_f.imshow(img)
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            is_event = k in tinfo["event_indices"]
            event_num = 1 if k == 1 else (2 if k == 3 else None)

            border_color = "#e63946" if is_event else "#444444"
            border_width = 3.5 if is_event else 1.2

            for spine in ax_f.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Top timestamp badge
            ax_f.text(0.04, 0.82, f"t = {ts:.2f}s", transform=ax_f.transAxes,
                      fontsize=11.5, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111", edgecolor="none", alpha=0.88))

            # Event badge
            if is_event:
                ax_f.text(0.96, 0.82, f"★ Event #{event_num}", transform=ax_f.transAxes,
                          fontsize=11.5, fontweight="bold", color="white", ha="right",
                          bbox=dict(boxstyle="round,pad=0.25", facecolor="#e63946", edgecolor="none", alpha=0.95))

        # --- 3. Trace Box ---
        ax_trace = fig.add_subplot(col_sub[2])
        ax_trace.axis("off")

        trace_patch = FancyBboxPatch((0.005, 0.01), 0.99, 0.98, transform=ax_trace.transAxes,
                                     facecolor="#ffffff", edgecolor="#333333", linewidth=2.2,
                                     boxstyle=BoxStyle("Round", pad=0.01, rounding_size=0.04),
                                     clip_on=False)
        ax_trace.add_patch(trace_patch)

        ax_trace.text(0.05, 0.91, "Executable Trace", transform=ax_trace.transAxes,
                      fontsize=15.0, fontweight="bold", color=tinfo["color"], va="center")

        y_pos = 0.79
        for line in tinfo["cot_lines"]:
            if not line:
                y_pos -= 0.03
                continue

            if line.endswith(":"):
                ax_trace.text(0.05, y_pos, line, transform=ax_trace.transAxes,
                              fontsize=13.0, fontweight="bold", color="#111111", va="center")
            elif line.startswith("Final Answer:"):
                ax_trace.text(0.05, y_pos, line, transform=ax_trace.transAxes,
                              fontsize=13.5, fontweight="bold", color="#111111", va="center")
            else:
                ax_trace.text(0.05, y_pos, line, transform=ax_trace.transAxes,
                              fontsize=11.8, color="#222222", fontfamily="monospace", va="center")

            y_pos -= 0.088

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.01)
    plt.close()
    print(f"Saved dataset overview figure to {output_path}")

if __name__ == "__main__":
    out_file = Path("outputs/dataset_overview_figure.png")
    build_dataset_overview_figure(out_file)
