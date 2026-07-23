#!/usr/bin/env python3
import sys
import json
import pandas as pd
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eventlapse.utils.paths import get_outputs_dir

@click.command()
@click.option("--output-csv", default=None, help="Path for aggregated CSV file")
def main(output_csv: str):
    """
    Aggregates experiment results JSONL files into consolidated summary CSV tables,
    including Input-token usage, API cost, latency, supplied frames, and accuracy per mode.
    """
    outputs_dir = get_outputs_dir()
    jsonl_files = list(outputs_dir.glob("results_*.jsonl"))

    if not jsonl_files:
        print("No result JSONL files found in outputs directory.")
        return

    records = []
    for jf in jsonl_files:
        with open(jf, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Ensure required resource fields exist
    for col in ["num_frames", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd", "latency_sec"]:
        if col not in df.columns:
            df[col] = 0

    out_path = Path(output_csv) if output_csv else outputs_dir / "aggregated_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} experiment rows to {out_path}")

    # Generate Resource & Cost Summary Table
    group_cols = ["provider", "requested_model", "input_mode", "prompt_condition"]
    valid_cols = [c for c in group_cols if c in df.columns]

    if valid_cols:
        summary_df = df.groupby(valid_cols).agg(
            sample_count=("sample_id", "count"),
            mean_num_frames=("num_frames", "mean"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
            total_cost_usd=("estimated_cost_usd", "sum"),
            mean_cost_usd_per_sample=("estimated_cost_usd", "mean"),
            exact_match_accuracy=("exact_match_result", "mean")
        ).reset_index()

        summary_df["mean_num_frames"] = summary_df["mean_num_frames"].round(1)
        summary_df["mean_prompt_tokens"] = summary_df["mean_prompt_tokens"].round(1)
        summary_df["mean_completion_tokens"] = summary_df["mean_completion_tokens"].round(1)
        summary_df["mean_total_tokens"] = summary_df["mean_total_tokens"].round(1)
        summary_df["mean_latency_sec"] = summary_df["mean_latency_sec"].round(2)
        summary_df["total_cost_usd"] = summary_df["total_cost_usd"].round(6)
        summary_df["mean_cost_usd_per_sample"] = summary_df["mean_cost_usd_per_sample"].round(6)
        summary_df["exact_match_accuracy"] = summary_df["exact_match_accuracy"].round(4)

        summary_path = outputs_dir / "mode_resource_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Generated resource & cost summary table at {summary_path}")
        print("\n=== RESOURCE & COST SUMMARY TABLE ===")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
