#!/usr/bin/env python3
import json
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_heatmap_from_jsonl(jsonl_path: str, output_png: str = None):
    jsonl_file = Path(jsonl_path).resolve()
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found at {jsonl_file}")

    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    records = []
    for d in data:
        sample_id = d.get("sample_id", "")
        task = d.get("task", "bounce_ball")
        em = d.get("exact_match_result", False)
        
        m_f = re.search(r"_F([\d.]+)_", sample_id)
        m_n = re.search(r"_N(\d+)_", sample_id)
        
        if m_n and m_f:
            n_val = int(m_n.group(1))
            f_val = float(m_f.group(1))
            records.append({
                "N": n_val,
                "F": f_val,
                "em": 1.0 if em else 0.0,
                "task": task
            })

    df = pd.DataFrame(records)

    # Pivot with F as index (Y-axis) and N as columns (X-axis)
    pivot = df.pivot_table(index="F", columns="N", values="em", aggfunc="mean")

    # Broadcast N=0 accuracy across all frequencies if N=0 is invariant
    if 0 in pivot.columns:
        n0_mean = pivot[0].dropna().mean()
        if not np.isnan(n0_mean):
            pivot[0] = pivot[0].fillna(n0_mean)

    pivot_asc = pivot.sort_index(ascending=True)

    plt.figure(figsize=(10, 6.5), dpi=300)
    sns.set_theme(style="white", font="sans-serif")

    ax = sns.heatmap(
        pivot_asc,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Exact Match Accuracy", "shrink": 0.85}
    )

    run_title = jsonl_file.stem.replace("results_matrix_", "").replace("_", " ").title()
    plt.title(f"{run_title} — Accuracy Matrix (N × F)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Event Count N (X-axis)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Event Frequency F (Hz) (Y-axis)", fontsize=12, fontweight="bold", labelpad=10)

    ax.set_xticklabels([f"{c}" for c in pivot_asc.columns], fontsize=11)
    ax.set_yticklabels([f"{f:.1f}" for f in pivot_asc.index], fontsize=11, rotation=0)

    plt.tight_layout()

    if output_png is None:
        output_png = jsonl_file.parent / f"heatmap_{jsonl_file.stem}.png"
    else:
        output_png = Path(output_png)

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    output_svg = output_png.with_suffix(".svg")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()

    print(f"Heatmap figure saved to:\n  - PNG: {output_png}\n  - SVG: {output_svg}")
    return output_png, pivot_asc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot N x F heatmap from EventLapse result jsonl file")
    parser.add_argument("jsonl_path", type=str, help="Path to jsonl results file")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()
    
    plot_heatmap_from_jsonl(args.jsonl_path, args.output)
