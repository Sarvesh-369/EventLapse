#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_outputs_dir

@click.command()
@click.option("--csv-file", default=None, help="Input aggregated CSV file")
@click.option("--output-dir", default=None, help="Directory to save heatmaps")
def main(csv_file: str, output_dir: str):
    """
    Generates N x F matrix accuracy heatmaps for temporal & spatial reasoning tasks.
    """
    out_dir = Path(output_dir) if output_dir else get_outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(csv_file) if csv_file else out_dir / "aggregated_results.csv"

    if not input_path.exists():
        print(f"Input aggregated CSV not found at {input_path}. Creating sample N x F matrix heatmap template...")
        # Create a sample 4x6 N x F matrix heatmap template
        counts = [2, 4, 8, 12]
        freqs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        dummy_matrix = np.array([
            [1.00, 0.95, 0.90, 0.85, 0.70, 0.50],
            [0.95, 0.90, 0.82, 0.75, 0.60, 0.40],
            [0.85, 0.78, 0.70, 0.62, 0.45, 0.30],
            [0.70, 0.60, 0.52, 0.40, 0.25, 0.15],
        ])

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            dummy_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=[f"{f} Hz" for f in freqs],
            yticklabels=[f"N={c}" for c in counts],
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Exact Match Accuracy"}
        )
        plt.title("N x F Parametric Matrix Accuracy (Event Count N vs Event Frequency F)", fontsize=14, pad=15)
        plt.xlabel("Event Frequency F (Hz)", fontsize=12)
        plt.ylabel("Event Count N", fontsize=12)
        plt.tight_layout()

        out_path = out_dir / "n_x_f_matrix_heatmap_template.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved N x F matrix heatmap template figure to {out_path}")
        return

    df = pd.read_csv(input_path)

    # Check if control parameter columns exist
    if "task" not in df.columns or "exact_match_result" not in df.columns:
        print("Required columns ('task', 'exact_match_result') not found in CSV.")
        return

    tasks = df["task"].unique()

    for task_name in tasks:
        task_df = df[df["task"] == task_name]
        if task_df.empty:
            continue

        # Check for control parameters or multi-axis parameters
        pivot = task_df.pivot_table(
            index="control_value",
            columns="input_mode",
            values="exact_match_result",
            aggfunc="mean"
        )

        if not pivot.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                cbar_kws={"label": "Exact Match Accuracy"}
            )
            plt.title(f"{task_name.replace('_', ' ').title()} - N x F Accuracy Matrix", fontsize=14, pad=15)
            plt.xlabel("Input Mode / Sampling Rate", fontsize=12)
            plt.ylabel("Control Parameter Value", fontsize=12)
            plt.tight_layout()

            out_path = out_dir / f"heatmap_matrix_{task_name}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved N x F heatmap for {task_name} to {out_path}")

if __name__ == "__main__":
    main()
