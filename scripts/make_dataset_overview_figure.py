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
            "domain": "Bounce Ball Domain",
            "subtitle": "Physical Motion & Wall Contact Tracking",
            "question": "Q: \"How many times did the ball contact the walls?\"",
            "video_path": demo_dir / "videos/bounce_ball/bounce_N2_F1.0_seed0.mp4",
            "gt_path": demo_dir / "gt/bounce_ball/bounce_N2_F1.0_seed0_gt.json",
            "trace_path": demo_dir / "traces/bounce_ball/bounce_N2_F1.0_seed0_trace.json",
            "sample_ts": [5.50, 6.49, 7.00, 7.49, 8.50],
            "event_indices": [1, 3], # 0-indexed position in sample_ts that are events
            "color": "#1f77b4",
        },
        {
            "domain": "Blinking Domain",
            "subtitle": "Luminance & Opacity Pulse Tracking",
            "question": "Q: \"How many times did the object blink?\"",
            "video_path": demo_dir / "videos/blinking/blinking_N2_F1.0_seed0.mp4",
            "gt_path": demo_dir / "gt/blinking/blinking_N2_F1.0_seed0_gt.json",
            "trace_path": demo_dir / "traces/blinking/blinking_N2_F1.0_seed0_trace.json",
            "sample_ts": [5.70, 6.72, 7.20, 7.72, 8.70],
            "event_indices": [1, 3],
            "color": "#e65100",
        },
        {
            "domain": "State Machine Domain",
            "subtitle": "Discrete Visual State Transitions",
            "question": "Q: \"How many state transitions occurred in the video?\"",
            "video_path": demo_dir / "videos/state_machine/state_N2_F1.0_seed0.mp4",
            "gt_path": demo_dir / "gt/state_machine/state_N2_F1.0_seed0_gt.json",
            "trace_path": demo_dir / "traces/state_machine/state_N2_F1.0_seed0_trace.json",
            "sample_ts": [5.10, 6.16, 6.65, 7.16, 8.10],
            "event_indices": [1, 3],
            "color": "#2e7d32",
        }
    ]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig = plt.figure(figsize=(18, 11), dpi=300, facecolor="white")
    
    # 3 Main Columns
    col_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.18, left=0.03, right=0.97, top=0.92, bottom=0.03)

    for c_idx, tinfo in enumerate(tasks_info):
        # Sub-gridspec for each column (Header, Frames, Trace)
        col_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=col_gs[c_idx],
                                                   height_ratios=[0.9, 4.2, 2.5], hspace=0.18)

        # --- 1. Header Box (Title & Prompt) ---
        ax_header = fig.add_subplot(col_sub[0])
        ax_header.axis("off")

        header_patch = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=ax_header.transAxes,
                                      facecolor="#f8f9fa", edgecolor=tinfo["color"], linewidth=2.0,
                                      boxstyle=BoxStyle("Round", pad=0.02))
        ax_header.add_patch(header_patch)

        ax_header.text(0.5, 0.72, tinfo["domain"], transform=ax_header.transAxes,
                       fontsize=13, fontweight="bold", color=tinfo["color"], ha="center", va="center")
        ax_header.text(0.5, 0.44, tinfo["subtitle"], transform=ax_header.transAxes,
                       fontsize=8.5, fontweight="bold", color="#555555", ha="center", va="center")
        ax_header.text(0.5, 0.18, tinfo["question"], transform=ax_header.transAxes,
                       fontsize=8.0, fontstyle="italic", color="#222222", ha="center", va="center")

        # --- 2. Frames Grid (5 Big Frames Centered Around Events) ---
        # Extract frames
        extracted = [(ts, extract_frame_at_timestamp(tinfo["video_path"], ts)) for ts in tinfo["sample_ts"]]

        ax_frames_container = fig.add_subplot(col_sub[1])
        ax_frames_container.axis("off")

        # Grid of 5 frames stacked vertically (or 5 frames vertically in each column)
        frame_sub = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=col_sub[1], hspace=0.12)

        with open(tinfo["trace_path"], "r") as tf:
            tr_data = json.load(tf)
            events = tr_data.get("events", [])

        for k, (ts, img) in enumerate(extracted):
            ax_f = fig.add_subplot(frame_sub[k])
            ax_f.imshow(img)
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            is_event = k in tinfo["event_indices"]
            event_num = 1 if k == 1 else (2 if k == 3 else None)

            border_color = "#e63946" if is_event else "#cccccc"
            border_width = 2.5 if is_event else 1.0

            for spine in ax_f.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Left badge: timestamp
            ax_f.text(0.03, 0.82, f"t = {ts:.2f}s", transform=ax_f.transAxes,
                      fontsize=8, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", edgecolor="none", alpha=0.75))

            # Right badge: Event indicator
            if is_event:
                ax_f.text(0.97, 0.82, f"★ Event #{event_num}", transform=ax_f.transAxes,
                          fontsize=8, fontweight="bold", color="white", ha="right",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#e63946", edgecolor="none", alpha=0.95))

        # --- 3. Actual Generated Trace Example ---
        ax_trace = fig.add_subplot(col_sub[2])
        ax_trace.axis("off")

        trace_patch = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=ax_trace.transAxes,
                                     facecolor="#ffffff", edgecolor="#cccccc", linewidth=1.5,
                                     boxstyle=BoxStyle("Round", pad=0.02))
        ax_trace.add_patch(trace_patch)

        ax_trace.text(0.05, 0.88, "Ground-Truth MORSE Executable Trace", transform=ax_trace.transAxes,
                      fontsize=9.5, fontweight="bold", color="#111111", va="center")

        y_pos = 0.72
        for idx, e in enumerate(events):
            if "wall_identity" in e:
                detail = f"Ball contacted '{e['wall_identity']}'"
            elif "blink_index" in e:
                detail = f"Object blinked ON (pulse)"
            elif "from_state" in e:
                detail = f"Transitioned State {e['from_state']} ➔ State {e['to_state']}"
            else:
                detail = "Event detected"

            line_str = f"• [{e['timestamp']:.2f}s] {detail}  (count={e['running_count']})"
            ax_trace.text(0.05, y_pos, line_str, transform=ax_trace.transAxes,
                          fontsize=7.8, color="#333333", fontfamily="monospace", va="center")
            y_pos -= 0.18

        ax_trace.text(0.05, 0.22, "Final Answer:", transform=ax_trace.transAxes,
                      fontsize=9, fontweight="bold", color="#111111", va="center")
        ax_trace.text(0.42, 0.22, "\\boxed{2}", transform=ax_trace.transAxes,
                      fontsize=10, fontweight="bold", color="white", va="center", ha="center",
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="#e63946", edgecolor="none"))

    fig.suptitle("EventLapse Benchmark Dataset: N=2 Event Count Samples (F=1.0 Hz)",
                 fontsize=15, fontweight="bold", y=0.97, color="#111111")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved dataset overview figure to {output_path}")

if __name__ == "__main__":
    out_file = Path("outputs/dataset_overview_figure.png")
    build_dataset_overview_figure(out_file)
