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
    n_val = int(n_match.group(1)) if n_match else None
    f_val = float(f_match.group(1)) if f_match else None
    return n_val, f_val

def main():
    root_dir = Path(__file__).resolve().parent.parent
    outputs_dir = root_dir / "outputs"
    paper_dir = outputs_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating Paper Artifacts in {paper_dir} ===")

    # 1. Load Aggregated Results
    agg_csv = outputs_dir / "aggregated_results.csv"
    if not agg_csv.exists():
        print("Aggregated CSV not found. Running aggregate_results.py...")
        os.system("python3 scripts/aggregate_results.py")

    df = pd.read_csv(agg_csv)

    # Extract N and F from sample_id
    parsed = df["sample_id"].apply(parse_n_and_f)
    df["N_count"] = [p[0] for p in parsed]
    df["F_hz"] = [p[1] for p in parsed]

    # Filter out failed retry instances for clean metric calculation
    valid_df = df[df["exact_match_result"].notnull() & (df["error_message"].isnull())].copy()

    # -------------------------------------------------------------
    # FIGURE 1: Copy Dataset Overview Figure
    # -------------------------------------------------------------
    overview_fig = outputs_dir / "dataset_overview_figure.png"
    if overview_fig.exists():
        os.system(f"cp {overview_fig} {paper_dir}/fig_1_dataset_overview.png")
        print(f"Saved {paper_dir}/fig_1_dataset_overview.png")

    # -------------------------------------------------------------
    # FIGURE 2: 2D N x F Heatmaps across Baseline Domains
    # -------------------------------------------------------------
    baseline_df = valid_df[(valid_df["input_mode"] == "native_video") & 
                           (valid_df["prompt_condition"] == "structured_trace")].copy()

    domains = ["bounce_ball", "blinking", "state_machine"]
    domain_titles = {
        "bounce_ball": "Bounce Ball (Wall Contacts)",
        "blinking": "Blinking (Light Pulses)",
        "state_machine": "State Machine (Transitions)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    for idx, domain in enumerate(domains):
        sub_df = baseline_df[baseline_df["task"] == domain]
        if not sub_df.empty:
            pivot = sub_df.pivot_table(
                index="F_hz",
                columns="N_count",
                values="exact_match_result",
                aggfunc="mean"
            )
            # Broadcast N=0 across frequencies if present
            if 0 in pivot.columns:
                n0_val = pivot[0].dropna().mean()
                if not np.isnan(n0_val):
                    pivot[0] = pivot[0].fillna(n0_val)

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=axes[idx],
                vmin=0.0,
                vmax=1.0,
                cbar=(idx == 2),
                cbar_kws={"label": "Exact Match Accuracy"} if idx == 2 else None
            )
            axes[idx].set_title(domain_titles.get(domain, domain), fontsize=13, fontweight="bold", pad=10)
            axes[idx].set_xlabel("Event Count N", fontsize=11, fontweight="bold")
            axes[idx].set_ylabel("Event Frequency F (Hz)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_2_path = paper_dir / "fig_2_n_x_f_heatmaps.png"
    plt.savefig(fig_2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Diagnostic Interventions (FPS & Prompting Modes)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=300)

    # Panel A: Frame Density Interventions (bounce_ball)
    fps_df = valid_df[(valid_df["task"] == "bounce_ball") & 
                      (valid_df["prompt_condition"] == "structured_trace") &
                      (valid_df["input_mode"].str.startswith("frames_") | (valid_df["input_mode"] == "native_video"))].copy()

    mode_order = ["native_video", "frames_1fps", "frames_2fps", "frames_4fps", "frames_8fps", "frames_10fps", "frames_16fps"]
    mode_labels = ["Native", "1 FPS", "2 FPS", "4 FPS", "8 FPS", "10 FPS", "16 FPS"]

    fps_acc = []
    fps_labels_present = []
    for m, lbl in zip(mode_order, mode_labels):
        m_sub = fps_df[fps_df["input_mode"] == m]
        if not m_sub.empty:
            fps_acc.append(m_sub["exact_match_result"].mean())
            fps_labels_present.append(lbl)

    bars1 = ax1.bar(fps_labels_present, fps_acc, color="#1f77b4", edgecolor="#111111", linewidth=1.2, alpha=0.85)
    ax1.set_ylim(0, 0.5)
    ax1.axhline(0.80, color="red", linestyle="--", linewidth=1.5, label="Target Reliability (τ = 0.80)")
    ax1.set_title("A. Visual Frame Density Interventions (Bounce Ball)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Exact Match Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Visual Sampling Mode", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel B: Prompting Strategy Interventions (bounce_ball)
    prompt_df = valid_df[(valid_df["task"] == "bounce_ball") & 
                         (valid_df["input_mode"] == "native_video")].copy()

    p_order = ["direct", "structured_trace", "multi_turn_verification", "thinking", "role_prompting"]
    p_labels = ["Direct Answer", "Structured Trace", "Multi-Turn", "Thinking/CoT", "Role Prompting"]

    p_acc = []
    p_labels_present = []
    for p, lbl in zip(p_order, p_labels):
        p_sub = prompt_df[prompt_df["prompt_condition"] == p]
        if not p_sub.empty:
            p_acc.append(p_sub["exact_match_result"].mean())
            p_labels_present.append(lbl)

    bars2 = ax2.bar(p_labels_present, p_acc, color="#2ca02c", edgecolor="#111111", linewidth=1.2, alpha=0.85)
    ax2.set_ylim(0, 0.5)
    ax2.axhline(0.80, color="red", linestyle="--", linewidth=1.5, label="Target Reliability (τ = 0.80)")
    ax2.set_title("B. Reasoning & Prompt Format Interventions (Bounce Ball)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Exact Match Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Prompting Condition", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig_3_path = paper_dir / "fig_3_diagnostic_interventions.png"
    plt.savefig(fig_3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_3_path}")

    # -------------------------------------------------------------
    # TABLE 1: Resource Usage & Interventions LaTeX Table
    # -------------------------------------------------------------
    res_csv = outputs_dir / "mode_resource_summary.csv"
    if res_csv.exists():
        res_df = pd.read_csv(res_csv)

        tex_rows = []
        for _, r in res_df.iterrows():
            acc = f"{r['exact_match_accuracy'] * 100:.1f}\\%"
            frames = f"{r['mean_num_frames']:.0f}"
            p_tokens = f"{r['mean_prompt_tokens']:,.0f}"
            cost = f"\\${r['total_cost_usd']:.2f}"
            lat = f"{r['mean_latency_sec']:.1f}s"
            tex_rows.append(f"\\texttt{{{r['input_mode']}}} & \\texttt{{{r['prompt_condition']}}} & {r['sample_count']} & {frames} & {p_tokens} & {lat} & {cost} & \\textbf{{{acc}}} \\\\")

        tab_latex = """\\begin{table*}[t]
\\centering
\\small
\\caption{\\textbf{EventLapse Evaluation Resource, Token, and Latency Summary across Input Modes and Prompt Conditions for Gemini 3.6 Flash.}}
\\label{tab:resource_summary}
\\begin{tabular}{llrrrrrr}
\\toprule
\\textbf{Input Mode} & \\textbf{Prompt Condition} & \\textbf{Samples} & \\textbf{Avg Frames} & \\textbf{Avg Prompt Tokens} & \\textbf{Avg Latency} & \\textbf{Total Cost} & \\textbf{Accuracy} \\\\
\\midrule
""" + "\n".join(tex_rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
        tab1_path = paper_dir / "tab_2_resource_and_interventions.tex"
        with open(tab1_path, "w") as f:
            f.write(tab_latex)
        print(f"Saved {tab1_path}")

    # -------------------------------------------------------------
    # Note: tab_1_trace_grounded_evaluation.tex is dynamically generated by evaluate_paper_metrics.py

    # Copy generated PNGs and LaTeX tables to AAAI_27 directory
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    aaai_tabs = root_dir.parent / "morse papers/morse_profile/AAAI_27/tables-tex"
    if aaai_figs.exists():
        os.system(f"cp {paper_dir}/*.png '{aaai_figs}/'")
    if aaai_tabs.exists():
        os.system(f"cp {paper_dir}/*.tex '{aaai_tabs}/'")

    print("\n=== All Paper Plots and LaTeX Tables Successfully Generated and Synced! ===")

if __name__ == "__main__":
    main()
