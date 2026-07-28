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

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Panel B alone: Binned Ranges
    bins = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,15), (16,20), (21,30), (31,100)]
    bin_labels = ["1-2", "3-4", "5-6", "7-8", "9-10", "11-15", "16-20", "21-30", ">30"]

    for label, (df, color, marker) in dfs.items():
        b_accs = []
        for low, high in bins:
            sub = df[(df["gt_count"] >= low) & (df["gt_count"] <= high)]
            acc = sub["exact_match"].mean() if len(sub) > 0 else 0.0
            b_accs.append(acc)
        ax.plot(bin_labels, b_accs, marker=marker, label=label, color=color, linewidth=2.2, markersize=7)

    ax.set_title("Real-World Transfer Performance (RepCount Dataset)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Event Count Bins (N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, 0.70)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5, loc="upper right")

    plt.tight_layout()
    
    # Save PDF and PNG
    out_pdf = paper_dir / "fig_5_real_world_transfer.pdf"
    out_png = paper_dir / "fig_5_real_world_transfer.png"
    out_rw_pdf = real_world_dir / "fig_5_real_world_transfer.pdf"
    out_rw_png = real_world_dir / "fig_5_real_world_transfer.png"

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_rw_pdf, bbox_inches="tight")
    plt.savefig(out_rw_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")

    # Copy to AAAI_27 figs if exists
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    if aaai_figs.exists():
        os.system(f"cp '{out_pdf}' '{aaai_figs}/'")
        os.system(f"cp '{out_png}' '{aaai_figs}/'")
        print(f"Synced {out_pdf.name} to AAAI_27/figs")

if __name__ == "__main__":
    main()
