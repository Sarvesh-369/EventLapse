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
    Aggregates experiment results JSONL files into a consolidated summary CSV table.
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
    out_path = Path(output_csv) if output_csv else outputs_dir / "aggregated_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} experiment rows to {out_path}")

if __name__ == "__main__":
    main()
