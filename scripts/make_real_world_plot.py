#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def main():
    root_dir = Path(__file__).resolve().parent.parent
    real_world_dir = root_dir / "outputs" / "real world results"
    paper_dir = root_dir / "outputs" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("Zero-Shot", "results_repcount_gemini_gemini-3.6-flash_zero_shot.jsonl", "#1f77b4", "o"),
        ("Direct", "results_repcount_gemini_gemini-3.6-flash_direct.jsonl", "#ff7f0e", "s"),
        ("Chain of Thought (CoT)", "results_repcount_gemini_gemini-3.6-flash_cot.jsonl", "#2ca02c", "^")
    ]

    dfs = {}
    for label, fname, color, marker in files:
        fpath = real_world_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                recs = [json.loads(line) for line in f if line.strip()]
            dfs[label] = (pd.DataFrame(recs), color, marker)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Panel A: Discrete N (N = 1 to 15)
    for label, (df, color, marker) in dfs.items():
        sub_df = df[(df["gt_count"] >= 1) & (df["gt_count"] <= 15)]
        n_grouped = sub_df.groupby("gt_count")["exact_match"].mean()
        ax1.plot(n_grouped.index, n_grouped.values, marker=marker, label=label, color=color, linewidth=2.2, markersize=6)

    ax1.set_title("A. Accuracy vs. Event Count N (RepCount Dataset, N ≤ 15)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Ground-Truth Event Count N", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 0.70)
    ax1.set_xticks(range(1, 16))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    # Panel B: Binned Ranges
    bins = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,15), (16,20), (21,30), (31,100)]
    bin_labels = ["1-2", "3-4", "5-6", "7-8", "9-10", "11-15", "16-20", "21-30", ">30"]

    for label, (df, color, marker) in dfs.items():
        b_accs = []
        for low, high in bins:
            sub = df[(df["gt_count"] >= low) & (df["gt_count"] <= high)]
            acc = sub["exact_match"].mean() if len(sub) > 0 else 0.0
            b_accs.append(acc)
        ax2.plot(bin_labels, b_accs, marker=marker, label=label, color=color, linewidth=2.2, markersize=6)

    ax2.set_title("B. Accuracy vs. Event Count Bins (Real-World RepCount Transfer)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Event Count Bins (N)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 0.70)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    
    # Save output to both real world results and paper folder
    out1 = real_world_dir / "fig_5_real_world_transfer.png"
    out2 = paper_dir / "fig_5_real_world_transfer.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out1}")
    print(f"Saved {out2}")

    # Copy to AAAI_27 figs if exists
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    if aaai_figs.exists():
        os.system(f"cp '{out2}' '{aaai_figs}/'")
        print(f"Synced {out2} to AAAI_27/figs")

if __name__ == "__main__":
    main()
