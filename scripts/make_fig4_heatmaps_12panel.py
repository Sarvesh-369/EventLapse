#!/usr/bin/env python3
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_n_and_f(sample_id: str):
    n_match = re.search(r"_N(\d+)_", str(sample_id))
    f_match = re.search(r"_F([\d\.]+)_", str(sample_id))
    n_val = int(n_match.group(1)) if n_match else 0
    f_val = float(f_match.group(1)) if f_match else 0.0
    return n_val, f_val

def load_gt_events(sample_id, domain, root_dir):
    trace_path = root_dir / f"data/traces/{domain}/{sample_id}_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            data = json.load(f)
            return [e["timestamp"] for e in data.get("events", [])]
    return []

def main():
    root_dir = Path(__file__).resolve().parent.parent
    outputs_dir = root_dir / "outputs"
    paper_dir = outputs_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # 12 Interventions matching exact user screenshot
    heatmap_interventions = [
        ("Native Video (Baseline)", "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"),
        ("Sampling: 1 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_1fps_structured_trace.jsonl"),
        ("Sampling: 2 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_2fps_structured_trace.jsonl"),
        ("Sampling: 4 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_4fps_structured_trace.jsonl"),
        ("Sampling: 8 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_8fps_structured_trace.jsonl"),
        ("Sampling: 10 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_10fps_structured_trace.jsonl"),
        ("Sampling: 16 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_16fps_structured_trace.jsonl"),
        ("Keyframe Evidence", "results_matrix_gemini_gemini-3.6-flash_oracle_evidence_structured_trace.jsonl"),
        ("Prompt: Direct Answer", "results_matrix_gemini_gemini-3.6-flash_native_video_direct.jsonl"),
        ("Prompt: Multi-Turn", "results_matrix_gemini_gemini-3.6-flash_native_video_multi_turn_verification.jsonl"),
        ("Prompt: Thinking / CoT", "results_matrix_gemini_gemini-3.6-flash_native_video_thinking.jsonl"),
        ("Prompt: Role Prompting", "results_matrix_gemini_gemini-3.6-flash_native_video_role_prompting.jsonl"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 19.5), dpi=300)
    axes_flat = axes.flatten()
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    for idx, (title, fname) in enumerate(heatmap_interventions):
        fpath = outputs_dir / fname
        recs = []
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("task") == "bounce_ball" and r.get("exact_match_result") is not None:
                            n, freq = parse_n_and_f(r["sample_id"])
                            r["N_count"] = n
                            r["F_hz"] = freq
                            recs.append(r)
        if recs:
            df_h = pd.DataFrame(recs)
            pivot_h = df_h.pivot_table(
                index="F_hz",
                columns="N_count",
                values="exact_match_result",
                aggfunc="mean"
            )
            if 0 in pivot_h.columns:
                n0_mean = pivot_h[0].dropna().mean()
                if not np.isnan(n0_mean):
                    pivot_h[0] = pivot_h[0].fillna(n0_mean)

            sns.heatmap(
                pivot_h,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=axes_flat[idx],
                vmin=0.0,
                vmax=1.0,
                cbar=(idx == 11),
                cbar_kws={"label": "Final Answer Accuracy"} if idx == 11 else None
            )
            axes_flat[idx].set_title(title, fontsize=14, fontweight="bold", pad=8)
            axes_flat[idx].set_xlabel("Event Count N", fontsize=11, fontweight="bold")
            axes_flat[idx].set_ylabel("Event Frequency F (Hz)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_4h_png = paper_dir / "fig_4_heatmaps.png"
    fig_4h_pdf = paper_dir / "fig_4_heatmaps.pdf"
    plt.savefig(fig_4h_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_4h_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_4h_png} and {fig_4h_pdf}")

    # Copy to AAAI_27 figs if exists
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    if aaai_figs.exists():
        os.system(f"cp '{fig_4h_png}' '{aaai_figs}/'")
        os.system(f"cp '{fig_4h_pdf}' '{aaai_figs}/'")
        print(f"Synced {fig_4h_pdf.name} to AAAI_27/figs")

if __name__ == "__main__":
    main()
