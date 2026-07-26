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

def compute_metrics(records, root_dir, delta_t=1.0):
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
    paper_dir = outputs_dir / "paper"

    # Clean paper_dir completely
    if paper_dir.exists():
        for item in paper_dir.glob("*"):
            if item.is_file():
                item.unlink()
    paper_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating Required Paper Artifacts in {paper_dir} ===")

    # -------------------------------------------------------------
    # FIGURE 1: Generate Dataset Overview Figure
    # -------------------------------------------------------------
    os.system("python3 scripts/make_dataset_overview_figure.py")
    overview_fig = outputs_dir / "dataset_overview_figure.png"
    if overview_fig.exists():
        os.system(f"cp {overview_fig} {paper_dir}/fig_1_dataset_overview.png")
        print(f"Saved {paper_dir}/fig_1_dataset_overview.png")

    # Load baseline records for Table 1 and Figure 3
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

    # -------------------------------------------------------------
    # TABLE 1: 3-Part Trace-Grounded Evaluation Table
    # -------------------------------------------------------------
    # Part A: By Task
    tasks = [("Bounce Ball", "bounce_ball"), ("Blinking", "blinking"), ("State Machine", "state_machine")]
    part_a_rows = []
    for label, domain in tasks:
        sub = [r for r in baseline_records if r["task"] == domain]
        m = compute_metrics(sub, root_dir)
        part_a_rows.append(f"  {label:<24} & {m['acc']:5.1f}\\% & {m['vor']:4.2f} & {m['p']:5.1f}\\% & {m['r']:5.1f}\\% & {m['f1']:5.1f}\\% & {m['acr']:4.1f}\\% & {m['rfr']:4.1f}\\% \\\\")

    m_macro = compute_metrics(baseline_records, root_dir)
    part_a_rows.append(f"  \\textbf{{Macro Average}}        & \\textbf{{{m_macro['acc']:5.1f}\\%}} & \\textbf{{{m_macro['vor']:4.2f}}} & \\textbf{{{m_macro['p']:5.1f}\\%}} & \\textbf{{{m_macro['r']:5.1f}\\%}} & \\textbf{{{m_macro['f1']:5.1f}\\%}} & \\textbf{{{m_macro['acr']:4.1f}\\%}} & \\textbf{{{m_macro['rfr']:4.1f}\\%}} \\\\")

    # Part B: Region Subsets
    regions = [
        ("Low Count, Low Freq", lambda r: r["N_count"] <= 3 and r["F_hz"] <= 2.0),
        ("High Count, Low Freq", lambda r: r["N_count"] >= 5 and r["F_hz"] <= 2.0),
        ("Low Count, High Freq", lambda r: r["N_count"] <= 3 and r["F_hz"] >= 2.5),
        ("High Count, High Freq", lambda r: r["N_count"] >= 5 and r["F_hz"] >= 2.5),
    ]
    part_b_rows = []
    for label, cond in regions:
        sub = [r for r in baseline_records if cond(r)]
        m = compute_metrics(sub, root_dir)
        part_b_rows.append(f"  {label:<24} & {m['acc']:5.1f}\\% & {m['vor']:4.2f} & {m['p']:5.1f}\\% & {m['r']:5.1f}\\% & {m['f1']:5.1f}\\% & {m['acr']:4.1f}\\% & {m['rfr']:4.1f}\\% \\\\")

    # Part C: Bounce Ball Interventions
    interventions_files = [
        ("Native + Structured (Baseline)", "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"),
        ("Dense Sampling (4 FPS)", "results_matrix_gemini_gemini-3.6-flash_frames_4fps_structured_trace.jsonl"),
        ("Keyframe Evidence", "results_matrix_gemini_gemini-3.6-flash_oracle_evidence_structured_trace.jsonl"),
        ("Direct Answer", "results_matrix_gemini_gemini-3.6-flash_native_video_direct.jsonl"),
        ("Structured Trace", "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"),
        ("Multi-Turn Verification", "results_matrix_gemini_gemini-3.6-flash_native_video_multi_turn_verification.jsonl"),
        ("Thinking / CoT", "results_matrix_gemini_gemini-3.6-flash_native_video_thinking.jsonl"),
        ("Role Prompting", "results_matrix_gemini_gemini-3.6-flash_native_video_role_prompting.jsonl"),
    ]
    part_c_rows = []
    for label, fname in interventions_files:
        fpath = outputs_dir / fname
        sub = []
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("task") == "bounce_ball" and rec.get("exact_match_result") is not None:
                            sub.append(rec)
        m = compute_metrics(sub, root_dir)
        part_c_rows.append(f"  {label:<32} & {m['acc']:5.1f}\\% & {m['vor']:4.2f} & {m['p']:5.1f}\\% & {m['r']:5.1f}\\% & {m['f1']:5.1f}\\% & {m['acr']:4.1f}\\% & {m['rfr']:4.1f}\\% \\\\")

    tab1_latex = f"""\\begin{{table*}}[t]
\\centering
\\small
\\caption{{\\textbf{{Trace-Grounded Evaluation and Diagnostic Breakdown for EventLapse.}} \\emph{{Part A:}} Trace performance across synthetic domains. \\emph{{Part B:}} Performance across ($N \\times F$) capability regions. \\emph{{Part C:}} Diagnostic interventions on the Bouncing Ball task.}}
\\label{{tab:trace_grounded_evaluation}}
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
\\textbf{{Evaluation Domain / Region}} & \\textbf{{Final Answer Accuracy (\\%)}} & \\textbf{{VOR}} & \\textbf{{Trace P (\\%)}} & \\textbf{{Trace R (\\%)}} & \\textbf{{Trace F1 (\\%)}} & \\textbf{{ACR (\\%)}} & \\textbf{{RFR (\\%)}} \\\\
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Part A: Trace Evaluation by Task Domain}}}} \\\\
{chr(10).join(part_a_rows)}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Part B: Trace Evaluation across ($N \\times F$) Capability Regions}}}} \\\\
{chr(10).join(part_b_rows)}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Part C: Trace Diagnosis of Bounce Ball Interventions}}}} \\\\
\\textbf{{Visual Evidence Family}} & & & & & & & \\\\
{chr(10).join(part_c_rows[:3])}
\\textbf{{Prompting & Reasoning Family}} & & & & & & \\\\
{chr(10).join(part_c_rows[3:])}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    tab1_path = paper_dir / "tab_1_trace_grounded_evaluation.tex"
    with open(tab1_path, "w") as f:
        f.write(tab1_latex)
    print(f"Saved {tab1_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Heatmaps (X-axis: Event Count N, Y-axis: Frequency F)
    # -------------------------------------------------------------
    df_base = pd.DataFrame(baseline_records)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    domain_labels = {"bounce_ball": "Bounce Ball", "blinking": "Blinking", "state_machine": "State Machine"}

    for idx, domain in enumerate(["bounce_ball", "blinking", "state_machine"]):
        sub_df = df_base[df_base["task"] == domain]
        if not sub_df.empty:
            pivot = sub_df.pivot_table(
                index="F_hz",
                columns="N_count",
                values="exact_match_result",
                aggfunc="mean"
            )
            if 0 in pivot.columns:
                n0_mean = pivot[0].dropna().mean()
                if not np.isnan(n0_mean):
                    pivot[0] = pivot[0].fillna(n0_mean)

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=axes[idx],
                vmin=0.0,
                vmax=1.0,
                cbar=(idx == 2),
                cbar_kws={"label": "Final Answer Accuracy"} if idx == 2 else None
            )
            axes[idx].set_title(f"{domain_labels[domain]}", fontsize=14, fontweight="bold", pad=10)
            axes[idx].set_xlabel("Event Count N", fontsize=11, fontweight="bold")
            axes[idx].set_ylabel("Event Frequency F (Hz)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_3_path = paper_dir / "fig_3_n_x_f_heatmaps.png"
    plt.savefig(fig_3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Intervention Analysis Line Plots vs N x F (Panels A, B, C)
    # -------------------------------------------------------------
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel A: Visual Sampling Densities vs N x F
    fps_modes = [("1 FPS", "frames_1fps"), ("2 FPS", "frames_2fps"), ("4 FPS", "frames_4fps"), ("8 FPS", "frames_8fps"), ("16 FPS", "frames_16fps")]
    colors_fps = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (label, mode), color in zip(fps_modes, colors_fps):
        fpath = outputs_dir / f"results_matrix_gemini_gemini-3.6-flash_{mode}_structured_trace.jsonl"
        if fpath.exists():
            recs = []
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("task") == "bounce_ball" and r.get("exact_match_result") is not None:
                            n, freq = parse_n_and_f(r["sample_id"])
                            r["NxF"] = round(n * freq, 2)
                            recs.append(r)
            if recs:
                df_fps = pd.DataFrame(recs)
                nxf_grouped = df_fps.groupby("NxF")["exact_match_result"].mean()
                ax_a.plot(nxf_grouped.index, nxf_grouped.values, marker="o", label=label, color=color, linewidth=2, markersize=4)

    ax_a.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_a.set_title("A. Visual Sampling Densities", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Event Complexity (N × F)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(fontsize=9, loc="upper right")

    # Panel B: Keyframe Evidence vs Native vs 16 FPS vs N x F
    panel_b_modes = [("Native Video", "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl", "#1f77b4"),
                     ("16 FPS", "results_matrix_gemini_gemini-3.6-flash_frames_16fps_structured_trace.jsonl", "#9467bd"),
                     ("Keyframe", "results_matrix_gemini_gemini-3.6-flash_oracle_evidence_structured_trace.jsonl", "#e63946")]

    for label, fname, color in panel_b_modes:
        fpath = outputs_dir / fname
        if fpath.exists():
            recs = []
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("task") == "bounce_ball" and r.get("exact_match_result") is not None:
                            n, freq = parse_n_and_f(r["sample_id"])
                            r["NxF"] = round(n * freq, 2)
                            recs.append(r)
            if recs:
                df_b = pd.DataFrame(recs)
                nxf_grouped = df_b.groupby("NxF")["exact_match_result"].mean()
                ax_b.plot(nxf_grouped.index, nxf_grouped.values, marker="s" if label == "Keyframe" else "o", label=label, color=color, linewidth=2.2, markersize=4)

    ax_b.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_b.set_title("B. Keyframe Evidence vs Native & 16 FPS", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Event Complexity (N × F)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(fontsize=9, loc="upper right")

    # Panel C: Prompting Strategies Line Plot vs N x F
    prompt_strats = [("Direct", "results_matrix_gemini_gemini-3.6-flash_native_video_direct.jsonl", "#1f77b4"),
                     ("Structured Trace", "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl", "#2ca02c"),
                     ("Multi-Turn Verification", "results_matrix_gemini_gemini-3.6-flash_native_video_multi_turn_verification.jsonl", "#d62728"),
                     ("Thinking / CoT", "results_matrix_gemini_gemini-3.6-flash_native_video_thinking.jsonl", "#ff7f0e"),
                     ("Role Prompting", "results_matrix_gemini_gemini-3.6-flash_native_video_role_prompting.jsonl", "#9467bd")]

    for label, fname, color in prompt_strats:
        fpath = outputs_dir / fname
        if fpath.exists():
            recs = []
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("task") == "bounce_ball" and r.get("exact_match_result") is not None:
                            n, freq = parse_n_and_f(r["sample_id"])
                            r["NxF"] = round(n * freq, 2)
                            recs.append(r)
            if recs:
                df_c = pd.DataFrame(recs)
                nxf_grouped = df_c.groupby("NxF")["exact_match_result"].mean()
                ax_c.plot(nxf_grouped.index, nxf_grouped.values, marker="o", label=label, color=color, linewidth=2, markersize=4)

    ax_c.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_c.set_title("C. Prompting & Reasoning Formats", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Event Complexity (N × F)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Final Answer Accuracy", fontsize=11, fontweight="bold")
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_4_path = paper_dir / "fig_4_intervention_analysis.png"
    plt.savefig(fig_4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_4_path}")

    # Copy ONLY the required generated files to AAAI_27 directory
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    aaai_tabs = root_dir.parent / "morse papers/morse_profile/AAAI_27/tables-tex"
    if aaai_figs.exists():
        os.system(f"cp {paper_dir}/*.png '{aaai_figs}/'")
    if aaai_tabs.exists():
        os.system(f"cp {paper_dir}/*.tex '{aaai_tabs}/'")

    print("\n=== Clean Generation Complete: Saved 3 Figures and 1 Table to outputs/paper/ and synced to AAAI_27! ===")

if __name__ == "__main__":
    main()
