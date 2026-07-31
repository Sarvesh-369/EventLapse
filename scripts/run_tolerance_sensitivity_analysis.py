#!/usr/bin/env python3
import json
import re

import numpy as np
import pandas as pd
from pathlib import Path

def parse_n_and_f(sample_id: str):
    n_match = re.search(r"_N(\d+)_", str(sample_id))
    f_match = re.search(r"_F([\d\.]+)_", str(sample_id))
    n_val = int(n_match.group(1)) if n_match else 0
    f_val = float(f_match.group(1)) if f_match else 0.0
    return n_val, f_val

def extract_timestamps(text):
    if not text:
        return []
    matches_mmss = re.findall(r"\b(\d{1,2}):(\d{2}(?:\.\d+)?)\b", str(text))
    results = []
    for m, s in matches_mmss:
        sec = float(m) * 60 + float(s)
        if 0.0 <= sec <= 30.0:
            results.append(sec)
    matches_sec = re.findall(r"\b(\d{1,2}\.\d+)\s*s\b", str(text), re.IGNORECASE)
    for s in matches_sec:
        sec = float(s)
        if 0.0 <= sec <= 30.0 and sec not in results:
            results.append(sec)
    return results

def load_gt_events(sample_id, domain, root_dir):
    trace_path = root_dir / f"data/traces/{domain}/{sample_id}_trace.json"
    if trace_path.exists():
        with open(trace_path) as f:
            data = json.load(f)
            return [e["timestamp"] for e in data.get("events", [])]
    return []

def compute_metrics_with_tolerance(records, root_dir, delta_fn):
    total = len(records)
    if total == 0:
        return {"acc": 0.0, "vor": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "acr": 0.0, "rfr": 0.0}

    em_hits = 0
    vor_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    f1_sum = 0.0
    acr_count = 0
    rfr_count = 0

    for r in records:
        is_em = bool(r.get("exact_match_result", False))
        if is_em:
            em_hits += 1

        gt_ts = load_gt_events(r["sample_id"], r["task"], root_dir)
        model_ts = extract_timestamps(r.get("raw_model_response"))

        N = len(gt_ts)
        M = len(model_ts)
        _, F_hz = parse_n_and_f(r["sample_id"])

        delta_t = delta_fn(F_hz)

        vor = (M / N) if N > 0 else (0.0 if M == 0 else 1.0)
        vor_sum += vor

        matched_model = 0
        gt_used = set()
        for m_t in model_ts:
            best_idx = None
            best_diff = delta_t + 1e-5
            for idx, g_t in enumerate(gt_ts):
                if idx not in gt_used:
                    diff = abs(m_t - g_t)
                    if diff <= delta_t and diff < best_diff:
                        best_diff = diff
                        best_idx = idx
            if best_idx is not None:
                matched_model += 1
                gt_used.add(best_idx)

        precision = (matched_model / M) if M > 0 else (1.0 if N == 0 else 0.0)
        recall = (len(gt_used) / N) if N > 0 else (1.0 if M == 0 else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else (1.0 if (N == 0 and M == 0) else 0.0)

        p_sum += precision
        r_sum += recall
        f1_sum += f1

        if is_em and (f1 < 0.80):
            acr_count += 1

        if (f1 >= 0.80) and (not is_em):
            rfr_count += 1

    return {
        "acc": (em_hits / total) * 100.0,
        "vor": vor_sum / total,
        "p": (p_sum / total) * 100.0,
        "r": (r_sum / total) * 100.0,
        "f1": (f1_sum / total) * 100.0,
        "acr": (acr_count / total) * 100.0,
        "rfr": (rfr_count / total) * 100.0
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    outputs_dir = root_dir / "outputs"

    baseline_file = outputs_dir / "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"
    baseline_records = []
    if baseline_file.exists():
        with open(baseline_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("exact_match_result") is not None:
                        n, freq = parse_n_and_f(rec["sample_id"])
                        rec["N_count"] = n
                        rec["F_hz"] = freq
                        baseline_records.append(rec)

    tolerances = [
        (r"Fixed \(\delta = 1.0\,\text{s}\)", lambda F: 1.0),
        (r"Fixed \(\delta = 0.5\,\text{s}\)", lambda F: 0.5),
        (r"Fixed \(\delta = 0.25\,\text{s}\)", lambda F: 0.25),
        (r"Rate-Relative \(\delta = \frac{1}{2F}\)", lambda F: 1.0 / (2.0 * F) if F > 0 else 1.0),
    ]

    print("=== SENSITIVITY ANALYSIS OVER MATCHING TOLERANCES (MACRO BASELINE) ===")
    results_table = []
    for tol_name, delta_fn in tolerances:
        m_macro = compute_metrics_with_tolerance(baseline_records, root_dir, delta_fn)
        results_table.append({
            "Tolerance Strategy": tol_name,
            "Accuracy (%)": f"{m_macro['acc']:.1f}",
            "VOR": f"{m_macro['vor']:.2f}",
            "Precision (%)": f"{m_macro['p']:.1f}",
            "Recall (%)": f"{m_macro['r']:.1f}",
            "F1 Score (%)": f"{m_macro['f1']:.1f}",
            "ACR (%)": f"{m_macro['acr']:.1f}",
            "RFR (%)": f"{m_macro['rfr']:.1f}"
        })

    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

    print("\n=== SENSITIVITY ANALYSIS OVER HIGH-FREQUENCY REGION (F >= 2.5 Hz) ===")
    hf_records = [r for r in baseline_records if r["F_hz"] >= 2.5]
    hf_table = []
    for tol_name, delta_fn in tolerances:
        m_hf = compute_metrics_with_tolerance(hf_records, root_dir, delta_fn)
        hf_table.append({
            "Tolerance Strategy": tol_name,
            "Accuracy (%)": f"{m_hf['acc']:.1f}",
            "VOR": f"{m_hf['vor']:.2f}",
            "Precision (%)": f"{m_hf['p']:.1f}",
            "Recall (%)": f"{m_hf['r']:.1f}",
            "F1 Score (%)": f"{m_hf['f1']:.1f}",
            "ACR (%)": f"{m_hf['acr']:.1f}",
            "RFR (%)": f"{m_hf['rfr']:.1f}"
        })
    df_hf = pd.DataFrame(hf_table)
    print(df_hf.to_string(index=False))

    # Generate TeX snippet for paper appendix/table
    tex_content = r"""\begin{table}[h]
\centering
\small
\caption{\textbf{Sensitivity Analysis of Trace Metrics Across Timestamp Matching Tolerances} ($\delta$). Metrics are reported for the macro baseline ($N \in [0..12], F \in [0.5..4.0]$\,Hz). The rate-relative tolerance $\delta(F) = \frac{1}{2F}$ bounds matching within each event's unique inter-event half-period.}
\label{tab:tolerance_sensitivity}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Matching Tolerance Strategy} & \textbf{Acc (\%)} & \textbf{VOR} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{Trace $F_1$ (\%)} & \textbf{ACR (\%)} \\
\midrule
"""
    for row in results_table:
        tex_content += f"{row['Tolerance Strategy']} & {row['Accuracy (%)']} & {row['VOR']} & {row['Precision (%)']} & {row['Recall (%)']} & {row['F1 Score (%)']} & {row['ACR (%)']} \\\\\n"
    tex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_out = root_dir / "outputs/paper/tab_app_tolerance_sensitivity.tex"
    with open(tex_out, "w") as f:
        f.write(tex_content)
    print(f"\nSaved LaTeX table to {tex_out}")

if __name__ == "__main__":
    main()
