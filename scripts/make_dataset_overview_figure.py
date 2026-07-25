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
            "subtitle": "Physical Motion & Wall Contact",
            "question": "How many times did the ball contact the walls?",
            "video_path": demo_dir / "videos/bounce_ball/bounce_N3_F1.5_seed0.mp4",
            "gt_path": demo_dir / "gt/bounce_ball/bounce_N3_F1.5_seed0_gt.json",
            "color": "#1f77b4",
        },
        {
            "domain": "Blinking",
            "subtitle": "Luminance & Opacity Pulse",
            "question": "How many times did the object blink?",
            "video_path": demo_dir / "videos/blinking/blinking_N3_F1.5_seed0.mp4",
            "gt_path": demo_dir / "gt/blinking/blinking_N3_F1.5_seed0_gt.json",
            "color": "#ff7f0e",
        },
        {
            "domain": "State Machine",
            "subtitle": "Discrete Visual State Transitions",
            "question": "How many state transitions occurred in the video?",
            "video_path": demo_dir / "videos/state_machine/state_N3_F1.5_seed0.mp4",
            "gt_path": demo_dir / "gt/state_machine/state_N3_F1.5_seed0_gt.json",
            "color": "#2ca02c",
        }
    ]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38)

    for i, tinfo in enumerate(tasks_info):
        # Load GT metadata
        with open(tinfo["gt_path"], "r") as gf:
            gt_data = json.load(gf)

        # Load trace json if exists
        trace_json_path = demo_dir / "traces" / tinfo["video_path"].parent.name / f"{tinfo['video_path'].stem}_trace.json"
        events = []
        if trace_json_path.exists():
            with open(trace_json_path, "r") as tf:
                tr = json.load(tf)
                events = tr.get("events", [])

        # Get event timestamps or generate 5 representative timestamps
        if events:
            ev_ts = [e["timestamp"] for e in events]
            min_t = max(0.2, ev_ts[0] - 0.5)
            max_t = ev_ts[-1] + 0.5
            sample_ts = [min_t] + ev_ts + [max_t]
            sample_ts = sample_ts[:5]
        else:
            sample_ts = [0.5, 1.5, 2.5, 3.5, 4.5]

        extracted = [(ts, extract_frame_at_timestamp(tinfo["video_path"], ts)) for ts in sample_ts]

        row_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[1.8, 5.2, 3.0], wspace=0.15)

        # --- Subpanel 1: Task Label & Question ---
        ax_label = fig.add_subplot(row_gs[0])
        ax_label.axis("off")

        rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.90, transform=ax_label.transAxes,
                              facecolor="#f8f9fa", edgecolor=tinfo["color"], linewidth=2.5,
                              boxstyle=BoxStyle("Round", pad=0.08))
        ax_label.add_patch(rect)

        ax_label.text(0.08, 0.72, f"{tinfo['domain']}", transform=ax_label.transAxes,
                      fontsize=15, fontweight="bold", color=tinfo["color"], va="center")
        ax_label.text(0.08, 0.50, f"{tinfo['subtitle']}", transform=ax_label.transAxes,
                      fontsize=9, fontweight="bold", color="#555555", va="center")
        ax_label.text(0.08, 0.25, f"Q: \"{tinfo['question']}\"", transform=ax_label.transAxes,
                      fontsize=8.5, fontstyle="italic", color="#222222", va="center", wrap=True)

        # --- Subpanel 2: 5 Sampled Video Keyframes ---
        ax_frames = fig.add_subplot(row_gs[1])
        ax_frames.axis("off")

        frame_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=row_gs[1], wspace=0.08)

        for k, (ts, img) in enumerate(extracted):
            ax_f = fig.add_subplot(frame_gs[k])
            ax_f.imshow(img)
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            # Highlight frames corresponding to events
            is_event_frame = False
            event_idx = None
            for idx, e in enumerate(events):
                if abs(e["timestamp"] - ts) < 0.2:
                    is_event_frame = True
                    event_idx = idx + 1
                    break

            border_color = "#e63946" if is_event_frame else "#cccccc"
            border_width = 3.0 if is_event_frame else 1.0

            for spine in ax_f.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Top label: Timestamp
            ax_f.set_title(f"t = {ts:.2f}s", fontsize=9, fontweight="bold", pad=4, color="#333333")

            # Bottom badge if event frame
            if is_event_frame:
                ax_f.text(0.5, 0.12, f"Event #{event_idx}", transform=ax_f.transAxes,
                          fontsize=8, fontweight="bold", color="white", ha="center",
                          bbox=dict(boxstyle="round,pad=0.25", facecolor="#e63946", edgecolor="none", alpha=0.95))

        # --- Subpanel 3: MORSE Executable Trace / Ledger ---
        ax_trace = fig.add_subplot(row_gs[2])
        ax_trace.axis("off")

        trace_rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.90, transform=ax_trace.transAxes,
                                    facecolor="#1e1e1e", edgecolor="#444444", linewidth=1.5,
                                    boxstyle=BoxStyle("Round", pad=0.06))
        ax_trace.add_patch(trace_rect)

        ax_trace.text(0.08, 0.82, "MORSE Executable Trace Ledger", transform=ax_trace.transAxes,
                      fontsize=10, fontweight="bold", color="#4cc9f0", va="center")

        y_pos = 0.65
        for idx, e in enumerate(events):
            if "wall_identity" in e:
                detail = f"Contact '{e['wall_identity']}'"
            elif "blink_index" in e:
                detail = f"Blink pulse ON"
            elif "from_state" in e:
                detail = f"Transition {e['from_state']}➔{e['to_state']}"
            else:
                detail = "Event detected"

            line_str = f"• [{e['timestamp']:.2f}s] {detail} (c={e['running_count']})"
            ax_trace.text(0.08, y_pos, line_str, transform=ax_trace.transAxes,
                          fontsize=8, color="#f8f9fa", fontfamily="monospace", va="center")
            y_pos -= 0.15

        ax_trace.text(0.08, y_pos, f"Final Answer: {gt_data['exact_answer']}", transform=ax_trace.transAxes,
                      fontsize=10, fontweight="bold", color="#ffffff", fontfamily="monospace", va="center",
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="#e63946", edgecolor="none", alpha=0.9))

    fig.suptitle("EventLapse Benchmark: Synthetic Task Domains & MORSE Ground-Truth Traces",
                 fontsize=16, fontweight="bold", y=0.98, color="#111111")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset overview figure to {output_path}")

if __name__ == "__main__":
    out_file = Path("outputs/dataset_overview_figure.png")
    build_dataset_overview_figure(out_file)
